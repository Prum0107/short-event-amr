import argparse
import json
import math
import os
from collections import Counter, defaultdict
from xml.sax.saxutils import escape


SIGNAL_ORDER = ["strong", "partial", "weak", "misleading"]
CATEGORY_ORDER = ["good", "candidate_exists", "boundary_error", "evidence_good_decode_bad", "semantic_miss"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * max(0.0, min(100.0, float(pct))) / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(values[lo])
    alpha = k - lo
    return float(values[lo] * (1.0 - alpha) + values[hi] * alpha)


def top_mean(values, frac=0.25):
    if not values:
        return 0.0
    k = max(1, int(math.ceil(len(values) * frac)))
    return mean(sorted(values, reverse=True)[:k])


def smooth_scores(scores, kernel=5):
    if not scores or kernel <= 1:
        return list(scores)
    pad = kernel // 2
    padded = [scores[0]] * pad + list(scores) + [scores[-1]] * pad
    return [sum(padded[idx : idx + kernel]) / kernel for idx in range(len(scores))]


def in_any_window(idx, windows):
    for st, ed in windows:
        if float(st) <= idx < float(ed):
            return True
    return False


def split_gt_out(scores, gt_windows):
    gt_values = []
    out_values = []
    for idx, score in enumerate(scores):
        if in_any_window(idx, gt_windows):
            gt_values.append(score)
        else:
            out_values.append(score)
    return gt_values, out_values


def best_window(row):
    windows = row.get("windows", [])
    return windows[0] if windows else None


def x_window_text(window):
    if not window:
        return "-"
    score = f", {window[2]:.3f}" if len(window) > 2 else ""
    return f"[{window[0]:.1f}, {window[1]:.1f}{score}]"


def diagnose_signal(stats):
    if stats["out_max"] > stats["gt_max"] + 0.10 and stats["gt_mean_margin"] <= 0.0:
        return "misleading"
    if stats["peak_in_gt"] and stats["gt_mean_margin"] > 0.03 and stats["gt_top25_margin"] > 0.03:
        return "strong"
    if stats["gt_max"] >= stats["global_p80"] or stats["gt_mean_margin"] > 0.0 or stats["top5_in_gt_frac"] >= 0.4:
        return "partial"
    return "weak"


def analyze_item(item, case_row):
    raw_scores = [float(value) for value in item.get("evidence_scores", [])]
    scores = smooth_scores(raw_scores, kernel=5)
    gt_windows = item.get("gt_windows", [])
    gt_values, out_values = split_gt_out(scores, gt_windows)
    if not scores:
        scores = [0.0]
    sorted_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    peak_idx = sorted_indices[0]
    top5 = sorted_indices[:5]
    top10 = sorted_indices[:10]

    gt_mean = mean(gt_values)
    out_mean = mean(out_values)
    gt_max = max(gt_values) if gt_values else 0.0
    out_max = max(out_values) if out_values else 0.0
    gt_top25 = top_mean(gt_values)
    out_top25 = top_mean(out_values)
    stats = {
        "qid": item.get("qid", ""),
        "vid": item.get("vid", ""),
        "query": item.get("query", ""),
        "category": case_row.get("category", ""),
        "top1_iou": float(case_row.get("top1_iou", 0.0)),
        "top5_iou": float(case_row.get("top5_iou", 0.0)),
        "gt_mean": gt_mean,
        "out_mean": out_mean,
        "gt_mean_margin": gt_mean - out_mean,
        "gt_max": gt_max,
        "out_max": out_max,
        "gt_max_margin": gt_max - out_max,
        "gt_top25": gt_top25,
        "out_top25": out_top25,
        "gt_top25_margin": gt_top25 - out_top25,
        "global_mean": mean(scores),
        "global_p80": percentile(scores, 80),
        "global_p90": percentile(scores, 90),
        "peak_idx": peak_idx,
        "peak_score": scores[peak_idx],
        "peak_in_gt": in_any_window(peak_idx, gt_windows),
        "top5_in_gt_frac": sum(1 for idx in top5 if in_any_window(idx, gt_windows)) / max(len(top5), 1),
        "top10_in_gt_frac": sum(1 for idx in top10 if in_any_window(idx, gt_windows)) / max(len(top10), 1),
        "pred_window": best_window(case_row),
        "gt_windows": gt_windows,
    }
    stats["signal_bucket"] = diagnose_signal(stats)
    if stats["category"] == "semantic_miss" and stats["signal_bucket"] in {"weak", "misleading"}:
        stats["ceiling_type"] = "representation_ceiling"
    elif stats["category"] == "semantic_miss":
        stats["ceiling_type"] = "decoder_candidate_ceiling"
    elif stats["category"] in {"boundary_error", "candidate_exists", "evidence_good_decode_bad"}:
        stats["ceiling_type"] = "decoding_ceiling"
    else:
        stats["ceiling_type"] = "solved_or_near_solved"
    return stats


def aggregate(rows):
    by_category = defaultdict(list)
    by_signal = defaultdict(list)
    by_ceiling = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_signal[row["signal_bucket"]].append(row)
        by_ceiling[row["ceiling_type"]].append(row)

    def avg(items, key):
        return sum(float(item.get(key, 0.0)) for item in items) / max(len(items), 1)

    category_rows = {}
    for category, items in by_category.items():
        signal_counts = Counter(item["signal_bucket"] for item in items)
        category_rows[category] = {
            "count": len(items),
            "avg_top1_iou": avg(items, "top1_iou"),
            "avg_top5_iou": avg(items, "top5_iou"),
            "avg_gt_mean_margin": avg(items, "gt_mean_margin"),
            "avg_gt_top25_margin": avg(items, "gt_top25_margin"),
            "peak_in_gt_pct": sum(1 for item in items if item["peak_in_gt"]) / max(len(items), 1),
            "signals": {name: signal_counts.get(name, 0) for name in SIGNAL_ORDER},
        }

    semantic_rows = [row for row in rows if row["category"] == "semantic_miss"]
    return {
        "num_samples": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "signal_counts": {name: len(by_signal.get(name, [])) for name in SIGNAL_ORDER},
        "ceiling_counts": dict(Counter(row["ceiling_type"] for row in rows)),
        "category_diagnosis": category_rows,
        "semantic_miss_count": len(semantic_rows),
        "semantic_miss_representation_ceiling": sum(1 for row in semantic_rows if row["ceiling_type"] == "representation_ceiling"),
        "semantic_miss_decoder_candidate_ceiling": sum(1 for row in semantic_rows if row["ceiling_type"] == "decoder_candidate_ceiling"),
        "avg_gt_mean_margin": avg(rows, "gt_mean_margin"),
        "avg_gt_top25_margin": avg(rows, "gt_top25_margin"),
        "peak_in_gt_pct": sum(1 for row in rows if row["peak_in_gt"]) / max(len(rows), 1),
    }


def table(headers, rows):
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def fmt_pct(value):
    return f"{100.0 * float(value):.2f}%"


def fmt_float(value):
    return f"{float(value):.4f}"


def rect_for_window(window, x_at, y0, plot_h, color, opacity):
    if not window:
        return ""
    x = x_at(window[0])
    w = max(1.0, x_at(window[1]) - x)
    return f'<rect x="{x:.2f}" y="{y0}" width="{w:.2f}" height="{plot_h}" fill="{color}" opacity="{opacity}"/>'


def case_svg(item, row, width=1120, height=230):
    scores = smooth_scores([float(value) for value in item.get("evidence_scores", [])], kernel=5)
    duration = max(float(item.get("duration", len(scores))), float(len(scores)), 1.0)
    x0 = 64
    y0 = 74
    plot_w = width - 104
    plot_h = 118
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

    gt_rects = "\n".join(rect_for_window(window, x_at, y0, plot_h, "#22c55e", 0.20) for window in row.get("gt_windows", []))
    pred_rect = rect_for_window(row.get("pred_window"), x_at, y0, plot_h, "#7c3aed", 0.24)
    peak_x = x_at(row.get("peak_idx", 0) + 0.5)
    peak_line = f'<line x1="{peak_x:.2f}" y1="{y0}" x2="{peak_x:.2f}" y2="{y0 + plot_h}" stroke="#ef4444" stroke-width="2" opacity="0.75"/>'
    points = " ".join(f"{x_at(idx + 0.5):.2f},{y_at(score):.2f}" for idx, score in enumerate(scores))
    polyline = f'<polyline points="{points}" fill="none" stroke="#0f7bc1" stroke-width="2"/>' if points else ""
    title = escape(f"{row['qid']}: {row.get('query', '')}"[:150])
    subtitle = escape(
        f"{row['category']} / {row['signal_bucket']} / {row['ceiling_type']} | "
        f"top1_iou={row['top1_iou']:.3f}, gt_mean_margin={row['gt_mean_margin']:.3f}, "
        f"gt_top25_margin={row['gt_top25_margin']:.3f}, pred={x_window_text(row.get('pred_window'))}"
    )
    return f"""
    <svg viewBox="0 0 {width} {height}" class="case-svg" role="img">
      <text x="{x0}" y="26" font-size="15" font-weight="700" fill="#0f172a">{title}</text>
      <text x="{x0}" y="48" font-size="12" fill="#475569">{subtitle}</text>
      <rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#cbd5e1"/>
      {''.join(grid)}
      {gt_rects}
      {pred_rect}
      {peak_line}
      {polyline}
      <text x="{x0}" y="{height - 16}" font-size="12" fill="#475569">
        GT green, predicted top1 purple, global evidence peak red, S(t,q) blue
      </text>
    </svg>
    """


def write_html(output_dir, summary, rows, evidence_by_qid, label, max_cases):
    category_table_rows = []
    category_diag = summary["category_diagnosis"]
    for category in CATEGORY_ORDER:
        stats = category_diag.get(category, {})
        signals = stats.get("signals", {})
        category_table_rows.append(
            [
                escape(category),
                str(stats.get("count", 0)),
                fmt_float(stats.get("avg_top1_iou", 0.0)),
                fmt_float(stats.get("avg_top5_iou", 0.0)),
                fmt_float(stats.get("avg_gt_mean_margin", 0.0)),
                fmt_float(stats.get("avg_gt_top25_margin", 0.0)),
                fmt_pct(stats.get("peak_in_gt_pct", 0.0)),
                " / ".join(str(signals.get(name, 0)) for name in SIGNAL_ORDER),
            ]
        )

    signal_rows = [[name, str(summary["signal_counts"].get(name, 0))] for name in SIGNAL_ORDER]
    ceiling_rows = [[escape(name), str(count)] for name, count in sorted(summary["ceiling_counts"].items())]

    semantic_rows = [row for row in rows if row["category"] == "semantic_miss"]
    likely_rep = sorted(
        [row for row in semantic_rows if row["ceiling_type"] == "representation_ceiling"],
        key=lambda row: (row["gt_top25_margin"], row["gt_mean_margin"]),
    )[:max_cases]
    likely_decode = sorted(
        [row for row in semantic_rows if row["ceiling_type"] == "decoder_candidate_ceiling"],
        key=lambda row: (row["gt_top25_margin"], row["gt_mean_margin"]),
        reverse=True,
    )[:max_cases]

    improved_boundary = sorted(
        [row for row in rows if row["category"] == "boundary_error"],
        key=lambda row: row["gt_top25_margin"],
        reverse=True,
    )[:max_cases]

    def svg_block(case_rows):
        if not case_rows:
            return "<p>No cases.</p>"
        return "\n".join(case_svg(evidence_by_qid[row["qid"]], row) for row in case_rows if row["qid"] in evidence_by_qid)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Feature Ceiling Diagnosis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: #ffffff; border-left: 4px solid #0f7bc1; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eceff3; }}
    .case-svg {{ width: 100%; max-width: 1160px; display: block; margin: 16px 0; border: 1px solid #d8dde6; background: white; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Feature Ceiling Diagnosis</h1>
  <div class="note">
    Label: <b>{escape(label)}</b>. This report separates representation ceiling from decoding ceiling by comparing GT evidence against outside-GT evidence.
  </div>
  <h2>Summary</h2>
  {table(["Metric", "Value"], [
      ["Samples", str(summary["num_samples"])],
      ["Semantic miss", str(summary["semantic_miss_count"])],
      ["Semantic miss likely representation ceiling", str(summary["semantic_miss_representation_ceiling"])],
      ["Semantic miss likely decoder/candidate ceiling", str(summary["semantic_miss_decoder_candidate_ceiling"])],
      ["Avg GT mean margin", fmt_float(summary["avg_gt_mean_margin"])],
      ["Avg GT top25 margin", fmt_float(summary["avg_gt_top25_margin"])],
      ["Global peak inside GT", fmt_pct(summary["peak_in_gt_pct"])],
  ])}
  <h2>Signal Buckets</h2>
  {table(["Signal", "Count"], signal_rows)}
  <h2>Ceiling Buckets</h2>
  {table(["Ceiling Type", "Count"], ceiling_rows)}
  <h2>Diagnosis By Decoder Category</h2>
  {table(["Category", "Count", "Avg Top1 IoU", "Avg Top5 IoU", "GT Mean Margin", "GT Top25 Margin", "Peak In GT", "Signals strong/partial/weak/misleading"], category_table_rows)}
  <h2>Likely Representation Ceiling: Persistent Semantic Miss</h2>
  {svg_block(likely_rep)}
  <h2>Likely Decoder/Candidate Ceiling: Semantic Miss With Usable Evidence</h2>
  {svg_block(likely_decode)}
  <h2>Boundary Errors With Strong Evidence</h2>
  {svg_block(improved_boundary)}
</body>
</html>"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--case_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="diagnosis")
    parser.add_argument("--max_cases", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    evidence = load_json(args.evidence_path)
    case_rows = load_json(args.case_rows_path)
    evidence_by_qid = {item["qid"]: item for item in evidence}
    case_by_qid = {row["qid"]: row for row in case_rows}

    rows = []
    for qid in sorted(set(evidence_by_qid) & set(case_by_qid)):
        rows.append(analyze_item(evidence_by_qid[qid], case_by_qid[qid]))

    summary = aggregate(rows)
    summary["label"] = args.label
    summary["evidence_path"] = args.evidence_path
    summary["case_rows_path"] = args.case_rows_path

    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(rows, os.path.join(args.output_dir, "case_diagnosis_rows.json"))
    write_html(args.output_dir, summary, rows, evidence_by_qid, args.label, args.max_cases)
    print(json.dumps(summary, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
