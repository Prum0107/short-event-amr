import argparse
import json
import os
from collections import Counter, defaultdict
from xml.sax.saxutils import escape


CATEGORY_ORDER = [
    "good",
    "candidate_exists",
    "boundary_error",
    "evidence_good_decode_bad",
    "semantic_miss",
]
CATEGORY_RANK = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def fmt_pct(value):
    return f"{100.0 * float(value):.2f}%"


def fmt_float(value):
    return f"{float(value):.4f}"


def best_window(windows):
    return windows[0] if windows else None


def migration_status(v1_category, v2_category, top1_delta):
    r1 = CATEGORY_RANK.get(v1_category, 99)
    r2 = CATEGORY_RANK.get(v2_category, 99)
    if r2 < r1:
        return "category_improved"
    if r2 > r1:
        return "category_regressed"
    if top1_delta >= 0.15:
        return "iou_improved"
    if top1_delta <= -0.15:
        return "iou_regressed"
    return "stable"


def build_case_rows(v1_rows, v2_rows, v1_evidence, v2_evidence):
    v1_by_qid = {row["qid"]: row for row in v1_rows}
    v2_by_qid = {row["qid"]: row for row in v2_rows}
    ev1_by_qid = {item["qid"]: item for item in v1_evidence}
    ev2_by_qid = {item["qid"]: item for item in v2_evidence}

    rows = []
    for qid in sorted(set(v1_by_qid) & set(v2_by_qid)):
        one = v1_by_qid[qid]
        two = v2_by_qid[qid]
        top1_delta = float(two.get("top1_iou", 0.0)) - float(one.get("top1_iou", 0.0))
        top5_delta = float(two.get("top5_iou", 0.0)) - float(one.get("top5_iou", 0.0))
        gap_delta = float(two.get("evidence_gap", 0.0)) - float(one.get("evidence_gap", 0.0))
        transition = f"{one.get('category')} -> {two.get('category')}"
        rows.append(
            {
                "qid": qid,
                "vid": one.get("vid", ""),
                "query": one.get("query", ""),
                "gt_windows": one.get("gt_windows", []),
                "v1_category": one.get("category", ""),
                "v2_category": two.get("category", ""),
                "transition": transition,
                "status": migration_status(one.get("category", ""), two.get("category", ""), top1_delta),
                "v1_top1_iou": float(one.get("top1_iou", 0.0)),
                "v2_top1_iou": float(two.get("top1_iou", 0.0)),
                "top1_iou_delta": top1_delta,
                "v1_top5_iou": float(one.get("top5_iou", 0.0)),
                "v2_top5_iou": float(two.get("top5_iou", 0.0)),
                "top5_iou_delta": top5_delta,
                "v1_evidence_gap": float(one.get("evidence_gap", 0.0)),
                "v2_evidence_gap": float(two.get("evidence_gap", 0.0)),
                "evidence_gap_delta": gap_delta,
                "v1_window": best_window(one.get("windows", [])),
                "v2_window": best_window(two.get("windows", [])),
                "v1_windows": one.get("windows", [])[:5],
                "v2_windows": two.get("windows", [])[:5],
                "v1_evidence": ev1_by_qid.get(qid, {}),
                "v2_evidence": ev2_by_qid.get(qid, {}),
            }
        )
    return rows


def summarize_rows(rows):
    transitions = Counter(row["transition"] for row in rows)
    statuses = Counter(row["status"] for row in rows)
    v1_categories = Counter(row["v1_category"] for row in rows)
    v2_categories = Counter(row["v2_category"] for row in rows)
    matrix = {
        src: {dst: 0 for dst in CATEGORY_ORDER}
        for src in CATEGORY_ORDER
    }
    for row in rows:
        if row["v1_category"] in matrix and row["v2_category"] in matrix[row["v1_category"]]:
            matrix[row["v1_category"]][row["v2_category"]] += 1

    semantic_rows = [row for row in rows if row["v1_category"] == "semantic_miss"]
    return {
        "num_samples": len(rows),
        "status_counts": dict(statuses),
        "v1_categories": dict(v1_categories),
        "v2_categories": dict(v2_categories),
        "transition_counts": dict(transitions),
        "transition_matrix": matrix,
        "semantic_miss_v1": len(semantic_rows),
        "semantic_miss_recovered": sum(1 for row in semantic_rows if row["v2_category"] != "semantic_miss"),
        "semantic_miss_to_good": sum(1 for row in semantic_rows if row["v2_category"] == "good"),
        "good_regressed": sum(1 for row in rows if row["v1_category"] == "good" and row["v2_category"] != "good"),
        "avg_top1_iou_delta": sum(row["top1_iou_delta"] for row in rows) / max(len(rows), 1),
        "avg_top5_iou_delta": sum(row["top5_iou_delta"] for row in rows) / max(len(rows), 1),
        "avg_evidence_gap_delta": sum(row["evidence_gap_delta"] for row in rows) / max(len(rows), 1),
    }


def x_window_text(window):
    if not window:
        return "-"
    score = f", {window[2]:.3f}" if len(window) > 2 else ""
    return f"[{window[0]:.1f}, {window[1]:.1f}{score}]"


def rect_for_window(window, x_at, y0, plot_h, color, opacity):
    if not window:
        return ""
    x = x_at(window[0])
    w = max(1.0, x_at(window[1]) - x)
    return f'<rect x="{x:.2f}" y="{y0}" width="{w:.2f}" height="{plot_h}" fill="{color}" opacity="{opacity}"/>'


def panel_svg(item, row, version, label, x0, y0, plot_w, plot_h):
    scores = item.get("evidence_scores", [])
    duration = max(float(item.get("duration", len(scores))), float(len(scores)), 1.0)
    x_scale = plot_w / duration

    def x_at(t):
        return x0 + float(t) * x_scale

    def y_at(v):
        return y0 + plot_h - max(0.0, min(1.0, float(v))) * plot_h

    grid = []
    for val in [0.0, 0.5, 1.0]:
        y = y_at(val)
        grid.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}" stroke="#d7dde8"/>')
        grid.append(f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#64748b">{val:.1f}</text>')

    gt = "\n".join(
        rect_for_window(window, x_at, y0, plot_h, "#22c55e", 0.20)
        for window in row.get("gt_windows", [])
    )
    pred = rect_for_window(
        row.get(f"{version}_window"),
        x_at,
        y0,
        plot_h,
        "#ef4444" if version == "v1" else "#7c3aed",
        0.22,
    )
    points = " ".join(f"{x_at(idx + 0.5):.2f},{y_at(score):.2f}" for idx, score in enumerate(scores))
    polyline = f'<polyline points="{points}" fill="none" stroke="#0f7bc1" stroke-width="2"/>' if points else ""
    label = (
        f"{escape(label)} {escape(row[f'{version}_category'])} "
        f"top1 IoU={row[f'{version}_top1_iou']:.3f} "
        f"gap={row[f'{version}_evidence_gap']:.3f}"
    )
    return f"""
    <g>
      <text x="{x0}" y="{y0 - 14}" font-size="13" font-weight="700" fill="#0f172a">{label}</text>
      <rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#cbd5e1"/>
      {''.join(grid)}
      {gt}
      {pred}
      {polyline}
    </g>
    """


def case_svg(row, label_v1="V1", label_v2="V2", width=1140, panel_h=150):
    x0 = 64
    plot_w = width - 100
    v1_item = row.get("v1_evidence", {})
    v2_item = row.get("v2_evidence", {})
    height = 2 * panel_h + 118
    title = escape(f"{row['qid']}: {row.get('query', '')}"[:160])
    svg = f"""
    <svg viewBox="0 0 {width} {height}" class="case-svg" role="img">
      <text x="{x0}" y="24" font-size="15" font-weight="700" fill="#0f172a">{title}</text>
      <text x="{x0}" y="46" font-size="12" fill="#475569">
        transition: {escape(row['transition'])}; top1 delta={row['top1_iou_delta']:.3f}; top5 delta={row['top5_iou_delta']:.3f}
      </text>
      {panel_svg(v1_item, row, "v1", label_v1, x0, 78, plot_w, panel_h)}
      {panel_svg(v2_item, row, "v2", label_v2, x0, 78 + panel_h + 42, plot_w, panel_h)}
      <text x="{x0}" y="{height - 14}" font-size="12" fill="#475569">
        GT green, {escape(label_v1)} top1 red, {escape(label_v2)} top1 purple, blue line is S(t,q)
      </text>
    </svg>
    """
    return svg


def table(headers, rows):
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def category_table(counts):
    return table(
        ["Category", "Count"],
        [[escape(name), str(counts.get(name, 0))] for name in CATEGORY_ORDER],
    )


def matrix_table(matrix):
    headers = ["v1 \\ v2"] + CATEGORY_ORDER
    rows = []
    for src in CATEGORY_ORDER:
        rows.append([escape(src)] + [str(matrix.get(src, {}).get(dst, 0)) for dst in CATEGORY_ORDER])
    return table(headers, rows)


def case_card(row, label_v1="V1", label_v2="V2"):
    meta = table(
        ["Field", "Value"],
        [
            ["Status", escape(row["status"])],
            ["Transition", escape(row["transition"])],
            [f"{escape(label_v1)} top1", escape(x_window_text(row.get("v1_window")))],
            [f"{escape(label_v2)} top1", escape(x_window_text(row.get("v2_window")))],
            ["Top1 IoU delta", fmt_float(row["top1_iou_delta"])],
            ["Evidence gap delta", fmt_float(row["evidence_gap_delta"])],
        ],
    )
    return f"""
    <article class="case-card">
      {case_svg(row, label_v1=label_v1, label_v2=label_v2)}
      <div class="case-meta">{meta}</div>
    </article>
    """


def pick_cases(rows):
    return {
        "semantic_miss_recovered": sorted(
            [row for row in rows if row["v1_category"] == "semantic_miss" and row["v2_category"] != "semantic_miss"],
            key=lambda row: (CATEGORY_RANK.get(row["v2_category"], 99), -row["top1_iou_delta"]),
        )[:12],
        "semantic_miss_to_good": sorted(
            [row for row in rows if row["v1_category"] == "semantic_miss" and row["v2_category"] == "good"],
            key=lambda row: -row["top1_iou_delta"],
        )[:12],
        "boundary_to_good": sorted(
            [row for row in rows if row["v1_category"] == "boundary_error" and row["v2_category"] == "good"],
            key=lambda row: -row["top1_iou_delta"],
        )[:12],
        "largest_iou_gains": sorted(rows, key=lambda row: row["top1_iou_delta"], reverse=True)[:12],
        "good_regressed": sorted(
            [row for row in rows if row["v1_category"] == "good" and row["v2_category"] != "good"],
            key=lambda row: row["top1_iou_delta"],
        )[:12],
        "largest_iou_losses": sorted(rows, key=lambda row: row["top1_iou_delta"])[:12],
        "still_semantic_miss": sorted(
            [row for row in rows if row["v1_category"] == "semantic_miss" and row["v2_category"] == "semantic_miss"],
            key=lambda row: row["top1_iou_delta"],
        )[:12],
    }


def write_html(output_dir, summary, selected, label_v1="V1", label_v2="V2"):
    sections = []
    section_titles = {
        "semantic_miss_recovered": "V1 Semantic Miss Recovered",
        "semantic_miss_to_good": "V1 Semantic Miss To Good",
        "boundary_to_good": "Boundary Error To Good",
        "largest_iou_gains": "Largest Top1 IoU Gains",
        "good_regressed": "Good Cases Regressed",
        "largest_iou_losses": "Largest Top1 IoU Losses",
        "still_semantic_miss": "Still Semantic Miss",
    }
    for key, rows in selected.items():
        cards = "\n".join(case_card(row, label_v1=label_v1, label_v2=label_v2) for row in rows)
        sections.append(f"<section><h2>{section_titles[key]}</h2>{cards or '<p>No cases.</p>'}</section>")

    status_rows = [
        [escape(key), str(value)]
        for key, value in sorted(summary["status_counts"].items(), key=lambda item: item[0])
    ]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(label_v1)} to {escape(label_v2)} Learned Decoder Migration</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #475569;
      --line: #d7dde8;
      --soft: #f8fafc;
      --accent: #7c3aed;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #ffffff;
      color: var(--ink);
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 28px 24px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 14px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
      line-height: 1.5;
      max-width: 920px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 22px 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--soft);
    }}
    .metric strong {{
      display: block;
      font-size: 22px;
      margin-bottom: 4px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .tables {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      align-items: start;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin: 10px 0 18px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef2f7;
    }}
    .case-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 14px 0;
      overflow-x: auto;
    }}
    .case-svg {{
      width: 100%;
      min-width: 960px;
      height: auto;
      display: block;
    }}
    .case-meta {{
      max-width: 920px;
    }}
    @media (max-width: 900px) {{
      .summary, .tables {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>{escape(label_v1)} to {escape(label_v2)} Learned Decoder Migration</h1>
  <p>
    This report compares learned decoder predictions before and after hard-negative evidence training.
    It asks whether the target system reduces semantic misses, and where the errors move after hard-negative pressure.
  </p>
  <div class="summary">
    <div class="metric"><strong>{summary['num_samples']}</strong><span>validation samples</span></div>
    <div class="metric"><strong>{summary['semantic_miss_recovered']}/{summary['semantic_miss_v1']}</strong><span>{escape(label_v1)} semantic misses recovered</span></div>
    <div class="metric"><strong>{summary['semantic_miss_to_good']}</strong><span>semantic miss to good</span></div>
    <div class="metric"><strong>{fmt_float(summary['avg_top1_iou_delta'])}</strong><span>average top1 IoU delta</span></div>
  </div>
  <div class="tables">
    <section>
      <h2>Category Counts</h2>
      <h3>{escape(label_v1)}</h3>
      {category_table(summary['v1_categories'])}
      <h3>{escape(label_v2)}</h3>
      {category_table(summary['v2_categories'])}
    </section>
    <section>
      <h2>Status Counts</h2>
      {table(['Status', 'Count'], status_rows)}
    </section>
  </div>
  <section>
    <h2>Transition Matrix</h2>
    {matrix_table(summary['transition_matrix'])}
  </section>
  {''.join(sections)}
</main>
</body>
</html>
"""
    path = os.path.join(output_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1_case_rows", default="results/learned_evidence_decoder_v1/learned_case_rows.json")
    parser.add_argument("--v2_case_rows", default="results/learned_evidence_decoder_v2_hn_l02/learned_case_rows.json")
    parser.add_argument(
        "--v1_evidence",
        default="results/evidence_baseline_release_v1/full_val_eval/predictions_evidence_samples.json",
    )
    parser.add_argument(
        "--v2_evidence",
        default="results/evidence_baseline_v2_hn_l02/full_val_eval/predictions_evidence_samples.json",
    )
    parser.add_argument("--output_dir", default="results/decoder_migration_v1_to_v2")
    parser.add_argument("--label_v1", default="V1")
    parser.add_argument("--label_v2", default="V2")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    rows = build_case_rows(
        v1_rows=load_json(args.v1_case_rows),
        v2_rows=load_json(args.v2_case_rows),
        v1_evidence=load_json(args.v1_evidence),
        v2_evidence=load_json(args.v2_evidence),
    )
    rows_for_json = []
    for row in rows:
        copy = dict(row)
        copy.pop("v1_evidence", None)
        copy.pop("v2_evidence", None)
        rows_for_json.append(copy)

    summary = summarize_rows(rows)
    selected = pick_cases(rows)
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(rows_for_json, os.path.join(args.output_dir, "case_migrations.json"))
    save_json(
        {key: [{k: v for k, v in row.items() if not k.endswith("_evidence")} for row in value] for key, value in selected.items()},
        os.path.join(args.output_dir, "selected_cases.json"),
    )
    report_path = write_html(args.output_dir, summary, selected, label_v1=args.label_v1, label_v2=args.label_v2)
    print(json.dumps(summary, indent=2))
    print("saved HTML report to", report_path)


if __name__ == "__main__":
    main()
