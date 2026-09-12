import argparse
import json
import os
from html import escape


DEFAULT_CASES = [
    ("semantic_recovery", "jf4iyQPJSvk_3"),
    ("temporal_correction", "q1ivQ_2fddk_1"),
    ("anchor_protection", "5I8lmN8rwDM_4"),
    ("temporal_risk_failure", "cz0FSQDVBMw_3"),
]


HEAD_KEYS = [
    ("Quality", "pred_quality_head"),
    ("Semantic gain", "pred_semantic_gain"),
    ("Temporal gain", "pred_temporal_gain"),
    ("Strict gain", "pred_strict_gain"),
    ("Anchor reliability", "pred_anchor_reliability"),
    ("Anchor risk", "pred_anchor_risk"),
    ("Semantic risk", "pred_semantic_risk"),
    ("Temporal risk", "pred_temporal_risk"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_qid(items):
    return {item["qid"]: item for item in items}


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def first_window(row_or_item):
    windows = row_or_item.get("windows") or row_or_item.get("pred_relevant_windows") or []
    return windows[0] if windows else []


def fmt_span(span):
    if not span or len(span) < 2:
        return "-"
    return f"[{float(span[0]):.1f}, {float(span[1]):.1f}]"


def fmt_value(value):
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def rect_for_span(span, x_at, y, height, color, opacity=0.16, stroke=None):
    if not span or len(span) < 2:
        return ""
    x = x_at(float(span[0]))
    w = max(1.0, x_at(float(span[1])) - x)
    stroke_attr = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{height:.2f}" '
        f'fill="{color}" opacity="{opacity}"{stroke_attr}/>'
    )


def line_for_span(span, x_at, y, height, color, label):
    if not span or len(span) < 2:
        return ""
    x = x_at(float(span[0]))
    w = max(1.0, x_at(float(span[1])) - x)
    text_x = x + min(w + 6.0, 220.0)
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{height:.2f}" '
        f'fill="none" stroke="{color}" stroke-width="3" rx="2"/>'
        f'<text x="{text_x:.2f}" y="{y + height - 5:.2f}" font-size="12" '
        f'font-family="Arial, sans-serif" fill="{color}">{escape(label)}</text>'
    )


def make_polyline(scores, x_at, y_at):
    if not scores:
        return ""
    points = []
    for idx, score in enumerate(scores):
        points.append(f"{x_at(idx + 0.5):.2f},{y_at(clamp(score)):.2f}")
    return " ".join(points)


def top_head(row):
    rows = row.get("top_rows") or []
    return rows[0] if rows else {}


def text_line(x, y, label, value, fill="#111827"):
    return (
        f'<text x="{x}" y="{y}" font-size="13" font-family="Arial, sans-serif" '
        f'fill="{fill}"><tspan font-weight="700">{escape(label)}</tspan>'
        f' {escape(str(value))}</text>'
    )


def render_case(label, qid, ms_item, sbec_item, ms_pred, ecf_row, raes_row):
    duration = max(
        float(ms_item.get("duration", len(ms_item.get("evidence_scores", [])))),
        float(sbec_item.get("duration", len(sbec_item.get("evidence_scores", [])))),
        float(len(ms_item.get("evidence_scores", []))),
        float(len(sbec_item.get("evidence_scores", []))),
        1.0,
    )
    width = 1280
    height = 620
    x0 = 78
    y0 = 94
    plot_w = 860
    plot_h = 270
    span_y = 394
    row_h = 28

    def x_at(t):
        return x0 + float(t) / duration * plot_w

    def y_at(v):
        return y0 + plot_h - clamp(v) * plot_h

    grid = []
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_at(value)
        grid.append(
            f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}" '
            f'stroke="#d1d5db" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" font-family="Arial, sans-serif" fill="#4b5563">{value:.2f}</text>'
        )
    tick_step = 30 if duration <= 180 else 50
    t = 0
    while t <= duration + 1e-6:
        x = x_at(t)
        grid.append(
            f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + plot_h}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{x:.2f}" y="{y0 + plot_h + 18}" text-anchor="middle" '
            f'font-size="11" font-family="Arial, sans-serif" fill="#4b5563">{int(t)}</text>'
        )
        t += tick_step

    ms_points = make_polyline(ms_item.get("evidence_scores", []), x_at, y_at)
    sbec_points = make_polyline(sbec_item.get("evidence_scores", []), x_at, y_at)
    gt_rects = "\n".join(
        rect_for_span(span, x_at, y0, plot_h, "#16a34a", 0.18)
        for span in ms_item.get("gt_windows", [])
    )

    ms_window = first_window(ms_pred)
    ecf_window = first_window(ecf_row)
    raes_window = first_window(raes_row)
    head = top_head(raes_row)

    span_rows = "\n".join(
        [
            line_for_span(ms_window, x_at, span_y, row_h, "#f59e0b", f"MS IoU {fmt_value(raes_row.get('ms_top1_iou'))}"),
            line_for_span(ecf_window, x_at, span_y + 36, row_h, "#7c3aed", f"TSEL-ECF IoU {fmt_value(ecf_row.get('top1_iou'))}"),
            line_for_span(raes_window, x_at, span_y + 72, row_h, "#dc2626", f"TSEL-RAES IoU {fmt_value(raes_row.get('top1_iou'))}"),
        ]
    )

    head_lines = []
    for idx, (name, key) in enumerate(HEAD_KEYS):
        head_lines.append(text_line(984, 210 + idx * 27, name + ":", fmt_value(head.get(key))))

    q = escape(ms_item.get("query", ""))
    if len(q) > 128:
        q = q[:125] + "..."
    title = f"{label.replace('_', ' ').title()} | {qid}"
    gt_text = ", ".join(fmt_span(span) for span in ms_item.get("gt_windows", [])[:3])
    meta_lines = [
        text_line(984, 90, "GT:", gt_text),
        text_line(984, 116, "MS:", fmt_span(ms_window)),
        text_line(984, 142, "RAES:", fmt_span(raes_window)),
    ]
    legend = """
<rect x="78" y="48" width="14" height="14" fill="#16a34a" opacity="0.18"/>
<text x="98" y="60" font-size="12" font-family="Arial, sans-serif" fill="#374151">GT span</text>
<line x1="170" y1="55" x2="212" y2="55" stroke="#2563eb" stroke-width="3"/>
<text x="220" y="60" font-size="12" font-family="Arial, sans-serif" fill="#374151">MS evidence</text>
<line x1="316" y1="55" x2="358" y2="55" stroke="#0f766e" stroke-width="3"/>
<text x="366" y="60" font-size="12" font-family="Arial, sans-serif" fill="#374151">SBEC evidence</text>
<rect x="500" y="44" width="34" height="18" fill="none" stroke="#f59e0b" stroke-width="3"/>
<text x="542" y="60" font-size="12" font-family="Arial, sans-serif" fill="#374151">MS top-1</text>
<rect x="618" y="44" width="34" height="18" fill="none" stroke="#7c3aed" stroke-width="3"/>
<text x="660" y="60" font-size="12" font-family="Arial, sans-serif" fill="#374151">TSEL-ECF</text>
<rect x="746" y="44" width="34" height="18" fill="none" stroke="#dc2626" stroke-width="3"/>
<text x="788" y="60" font-size="12" font-family="Arial, sans-serif" fill="#374151">TSEL-RAES</text>
"""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
<text x="40" y="30" font-size="20" font-family="Arial, sans-serif" fill="#111827" font-weight="700">{escape(title)}</text>
<text x="40" y="78" font-size="15" font-family="Arial, sans-serif" fill="#111827">{q}</text>
{legend}
<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#9ca3af"/>
{gt_rects}
{''.join(grid)}
<polyline points="{ms_points}" fill="none" stroke="#2563eb" stroke-width="2.4"/>
<polyline points="{sbec_points}" fill="none" stroke="#0f766e" stroke-width="2.4" opacity="0.92"/>
{span_rows}
<text x="{x0 + plot_w / 2:.2f}" y="558" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#374151">Time (s)</text>
<text x="22" y="{y0 + plot_h / 2:.2f}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#374151" transform="rotate(-90 22 {y0 + plot_h / 2:.2f})">Evidence score</text>
<rect x="960" y="48" width="280" height="496" fill="#f9fafb" stroke="#d1d5db"/>
<text x="984" y="68" font-size="15" font-family="Arial, sans-serif" fill="#111827" font-weight="700">Case summary</text>
{''.join(meta_lines)}
<text x="984" y="184" font-size="15" font-family="Arial, sans-serif" fill="#111827" font-weight="700">RAES heads</text>
{''.join(head_lines)}
</svg>
"""


def make_case_summary(label, qid, ms_item, ms_pred, ecf_row, raes_row):
    head = top_head(raes_row)
    return {
        "label": label,
        "qid": qid,
        "query": ms_item.get("query", ""),
        "gt_windows": ms_item.get("gt_windows", []),
        "ms_window": first_window(ms_pred),
        "ecf_window": first_window(ecf_row),
        "raes_window": first_window(raes_row),
        "ms_top1_iou": raes_row.get("ms_top1_iou"),
        "ecf_top1_iou": ecf_row.get("top1_iou"),
        "raes_top1_iou": raes_row.get("top1_iou"),
        "raes_candidate_source": raes_row.get("candidate_source", ""),
        "raes_representation": raes_row.get("chosen_representation", ""),
        "raes_heads": {key: head.get(key) for _, key in HEAD_KEYS},
    }


def write_index(output_dir, summaries):
    md_lines = [
        "# TSEL Paper Case Figures",
        "",
        "Generated from frozen test artifacts. These are candidate figures for the paper, not final layout exports.",
        "",
        "| Case | qid | Query | MS IoU | TSEL-ECF IoU | TSEL-RAES IoU | Figure |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>TSEL Paper Cases</title>",
        "<style>body{font-family:Arial,sans-serif;margin:28px;color:#111827} img{max-width:100%;border:1px solid #d1d5db;margin:14px 0 34px} table{border-collapse:collapse}td,th{border:1px solid #d1d5db;padding:6px 8px}</style>",
        "</head><body><h1>TSEL Paper Case Figures</h1>",
    ]
    for item in summaries:
        fig = item["figure"]
        md_lines.append(
            "| {label} | `{qid}` | {query} | {ms:.3f} | {ecf:.3f} | {raes:.3f} | [{fig}]({fig}) |".format(
                label=item["label"],
                qid=item["qid"],
                query=item["query"].replace("|", "/"),
                ms=float(item.get("ms_top1_iou") or 0.0),
                ecf=float(item.get("ecf_top1_iou") or 0.0),
                raes=float(item.get("raes_top1_iou") or 0.0),
                fig=fig,
            )
        )
        html_lines.append(f"<h2>{escape(item['label'])}: {escape(item['qid'])}</h2>")
        html_lines.append(f"<p>{escape(item['query'])}</p>")
        html_lines.append(f"<img src=\"{escape(fig)}\" alt=\"{escape(item['label'])}\">")
    html_lines.append("</body></html>")
    with open(os.path.join(output_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines) + "\n")


def parse_cases(raw_cases):
    if not raw_cases:
        return DEFAULT_CASES
    cases = []
    for item in raw_cases.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError("case entries must be label:qid")
        label, qid = item.split(":", 1)
        cases.append((label.strip(), qid.strip()))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_evidence_path", default="results/evidence_baseline_release_v1/full_test_eval/predictions_evidence_samples.json")
    parser.add_argument("--ms_predictions_path", default="results/test_candidate_level_fusion_temporal_hn_v4_stage_semantic_width/ms_clap_shape_v2_top2_predictions.jsonl")
    parser.add_argument("--sbec_evidence_path", default="results/temporal_evidence_hn_v4_stage_semantic_width/full_test_eval/predictions_evidence_samples.json")
    parser.add_argument("--ecf_rows_path", default="results/test_candidate_level_fusion_temporal_hn_v4_stage_semantic_width/fusion_case_rows.json")
    parser.add_argument("--raes_rows_path", default="results/test_semantic_temporal_candidate_scorer_v2_quality_guard_seed2027_cases/seed2027/case_rows.json")
    parser.add_argument("--output_dir", default="docs/paper_assets/figures/tsel_case_figures")
    parser.add_argument("--cases", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ms_by_qid = by_qid(load_json(args.ms_evidence_path))
    ms_pred_by_qid = by_qid(load_jsonl(args.ms_predictions_path))
    sbec_by_qid = by_qid(load_json(args.sbec_evidence_path))
    ecf_by_qid = by_qid(load_json(args.ecf_rows_path))
    raes_by_qid = by_qid(load_json(args.raes_rows_path))
    summaries = []
    for label, qid in parse_cases(args.cases):
        missing = [
            name
            for name, mapping in [
                ("ms_evidence", ms_by_qid),
                ("ms_predictions", ms_pred_by_qid),
                ("sbec_evidence", sbec_by_qid),
                ("ecf_rows", ecf_by_qid),
                ("raes_rows", raes_by_qid),
            ]
            if qid not in mapping
        ]
        if missing:
            raise KeyError(f"{qid} missing from {', '.join(missing)}")
        svg = render_case(
            label,
            qid,
            ms_by_qid[qid],
            sbec_by_qid[qid],
            ms_pred_by_qid[qid],
            ecf_by_qid[qid],
            raes_by_qid[qid],
        )
        safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_name = f"{safe_label}_{qid}.svg"
        with open(os.path.join(args.output_dir, out_name), "w", encoding="utf-8") as f:
            f.write(svg)
        summary = make_case_summary(label, qid, ms_by_qid[qid], ms_pred_by_qid[qid], ecf_by_qid[qid], raes_by_qid[qid])
        summary["figure"] = out_name
        summaries.append(summary)
    with open(os.path.join(args.output_dir, "case_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    write_index(args.output_dir, summaries)
    print(f"saved {len(summaries)} TSEL paper case figures to {args.output_dir}")


if __name__ == "__main__":
    main()
