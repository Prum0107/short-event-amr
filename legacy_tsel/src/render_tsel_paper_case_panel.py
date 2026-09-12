import argparse
import json
import os
from html import escape


CASE_MESSAGES = {
    "semantic_recovery": "Temporal evidence overrides an unreliable MS anchor.",
    "temporal_correction": "Boundary evidence turns a loose match into a strict hit.",
    "anchor_protection": "Risk-aware scoring preserves a reliable MS anchor.",
    "temporal_risk_failure": "Failure: temporal gain is over-trusted.",
}


DISPLAY_LABELS = {
    "semantic_recovery": "Semantic recovery",
    "temporal_correction": "Temporal correction",
    "anchor_protection": "Anchor protection",
    "temporal_risk_failure": "Temporal-risk failure",
}


HEAD_SELECTION = {
    "semantic_recovery": [
        ("semantic_gain", "pred_semantic_gain"),
        ("anchor_rel", "pred_anchor_reliability"),
    ],
    "temporal_correction": [
        ("temporal_gain", "pred_temporal_gain"),
        ("strict_gain", "pred_strict_gain"),
    ],
    "anchor_protection": [
        ("anchor_rel", "pred_anchor_reliability"),
        ("anchor_risk", "pred_anchor_risk"),
    ],
    "temporal_risk_failure": [
        ("temporal_gain", "pred_temporal_gain"),
        ("temporal_risk", "pred_temporal_risk"),
    ],
}


COLORS = {
    "gt": "#16a34a",
    "ms": "#f59e0b",
    "ecf": "#7c3aed",
    "raes": "#dc2626",
    "axis": "#9ca3af",
    "grid": "#e5e7eb",
    "text": "#111827",
    "muted": "#4b5563",
    "panel": "#f8fafc",
}


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def fmt_span(span):
    if not span or len(span) < 2:
        return "-"
    return f"[{float(span[0]):.0f}, {float(span[1]):.0f}]"


def fmt_float(value):
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def clip_text(text, limit):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def infer_durations(evidence_path):
    durations = {}
    if not evidence_path or not os.path.exists(evidence_path):
        return durations
    for item in load_json(evidence_path):
        qid = item.get("qid")
        if qid:
            duration = item.get("duration")
            if duration is None:
                scores = item.get("evidence_scores") or []
                duration = len(scores)
            durations[qid] = max(float(duration), 1.0)
    return durations


def case_duration(case, durations):
    qid = case.get("qid")
    duration = durations.get(qid, 0.0)
    for key in ["gt_windows"]:
        for span in case.get(key, []) or []:
            if len(span) >= 2:
                duration = max(duration, float(span[1]))
    for key in ["ms_window", "ecf_window", "raes_window"]:
        span = case.get(key) or []
        if len(span) >= 2:
            duration = max(duration, float(span[1]))
    return max(duration, 1.0)


def rect(x, y, w, h, fill, stroke=None, opacity=1.0, rx=0):
    stroke_attr = f' stroke="{stroke}" stroke-width="1.5"' if stroke else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 1.0):.2f}" '
        f'height="{h:.2f}" fill="{fill}" opacity="{opacity:.3f}" '
        f'rx="{rx}"{stroke_attr}/>'
    )


def text(x, y, content, size=13, fill=None, weight=None, anchor=None):
    fill = fill or COLORS["text"]
    attrs = [
        f'x="{x:.2f}"',
        f'y="{y:.2f}"',
        f'font-size="{size}"',
        'font-family="Arial, sans-serif"',
        f'fill="{fill}"',
    ]
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    return f"<text {' '.join(attrs)}>{escape(str(content))}</text>"


def span_bar(span, x_at, y, label, color, iou=None):
    if not span or len(span) < 2:
        return ""
    x = x_at(float(span[0]))
    w = max(2.0, x_at(float(span[1])) - x)
    caption = label if iou is None else f"{label} IoU {fmt_float(iou)}"
    return "\n".join(
        [
            rect(x, y, w, 12, color, opacity=0.22, rx=2),
            f'<line x1="{x:.2f}" y1="{y + 15:.2f}" x2="{x + w:.2f}" y2="{y + 15:.2f}" stroke="{color}" stroke-width="3"/>',
            text(min(x + w + 6, 960), y + 17, caption, size=11, fill=color),
        ]
    )


def render_case(case, durations, x, y, width):
    label = case.get("label", "")
    qid = case.get("qid", "")
    duration = case_duration(case, durations)
    plot_x = x + 300
    plot_y = y + 56
    plot_w = width - 560
    summary_x = plot_x + plot_w + 26

    def x_at(t):
        return plot_x + max(0.0, min(float(t) / duration, 1.0)) * plot_w

    parts = []
    parts.append(rect(x, y, width, 166, COLORS["panel"], stroke="#d1d5db", opacity=1.0, rx=6))
    parts.append(text(x + 22, y + 30, DISPLAY_LABELS.get(label, label), size=17, weight="700"))
    parts.append(text(x + 22, y + 52, qid, size=11, fill=COLORS["muted"]))
    parts.append(text(x + 22, y + 78, clip_text(case.get("query", ""), 58), size=13))
    parts.append(text(x + 22, y + 106, CASE_MESSAGES.get(label, ""), size=12, fill=COLORS["muted"]))

    parts.append(f'<line x1="{plot_x:.2f}" y1="{plot_y:.2f}" x2="{plot_x + plot_w:.2f}" y2="{plot_y:.2f}" stroke="{COLORS["axis"]}" stroke-width="1.5"/>')
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tx = plot_x + frac * plot_w
        parts.append(f'<line x1="{tx:.2f}" y1="{plot_y - 8:.2f}" x2="{tx:.2f}" y2="{plot_y + 94:.2f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(text(tx, plot_y + 112, f"{duration * frac:.0f}s", size=10, fill=COLORS["muted"], anchor="middle"))

    for span in case.get("gt_windows", []) or []:
        if len(span) >= 2:
            parts.append(span_bar(span, x_at, plot_y + 8, "GT", COLORS["gt"]))
    parts.append(span_bar(case.get("ms_window"), x_at, plot_y + 34, "MS", COLORS["ms"], case.get("ms_top1_iou")))
    parts.append(span_bar(case.get("ecf_window"), x_at, plot_y + 60, "ECF", COLORS["ecf"], case.get("ecf_top1_iou")))
    parts.append(span_bar(case.get("raes_window"), x_at, plot_y + 86, "RAES", COLORS["raes"], case.get("raes_top1_iou")))

    heads = case.get("raes_heads") or {}
    parts.append(text(summary_x, y + 34, "Key RAES heads", size=12, weight="700"))
    for idx, (name, key) in enumerate(HEAD_SELECTION.get(label, [])[:2]):
        parts.append(text(summary_x, y + 58 + idx * 22, f"{name}: {fmt_float(heads.get(key))}", size=12, fill=COLORS["muted"]))
    parts.append(text(summary_x, y + 114, f"MS {fmt_span(case.get('ms_window'))}", size=11, fill=COLORS["muted"]))
    parts.append(text(summary_x, y + 134, f"RAES {fmt_span(case.get('raes_window'))}", size=11, fill=COLORS["muted"]))
    return "\n".join(parts)


def render_panel(cases, durations):
    width = 1280
    margin = 34
    row_h = 188
    height = margin * 2 + 86 + row_h * len(cases)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#ffffff"),
        text(margin, 42, "TSEL case analysis: evidence helps, but risk matters", size=24, weight="700"),
        text(
            margin,
            68,
            "Green=GT, amber=MS anchor, purple=TSEL-ECF, red=TSEL-RAES. Rows are normalized by each clip duration.",
            size=13,
            fill=COLORS["muted"],
        ),
    ]
    y = 102
    for case in cases:
        parts.append(render_case(case, durations, margin, y, width - 2 * margin))
        y += row_h
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_usage_md(output_svg, cases):
    lines = [
        "# Paper-Ready TSEL Case Panel",
        "",
        "This figure is a simplified main-paper panel generated from the frozen case summaries.",
        "Use the individual SVGs in `tsel_case_figures/` for appendix evidence curves.",
        "",
        f"- Main panel: `{os.path.basename(output_svg)}`",
        "- Purpose: show four mechanisms without overcrowding the paper figure.",
        "- Color convention: GT green, MS anchor amber, TSEL-ECF purple, TSEL-RAES red.",
        "",
        "## Included Cases",
        "",
        "| Mechanism | qid | Query | Main message |",
        "|---|---|---|---|",
    ]
    for case in cases:
        label = case.get("label", "")
        lines.append(
            "| {label} | `{qid}` | {query} | {msg} |".format(
                label=DISPLAY_LABELS.get(label, label),
                qid=case.get("qid", ""),
                query=str(case.get("query", "")).replace("|", "/"),
                msg=CASE_MESSAGES.get(label, ""),
            )
        )
    lines += [
        "",
        "## Suggested Caption",
        "",
        "Qualitative analysis of TSEL candidate selection. Temporal-semantic evidence recovers missed moments and corrects loose boundaries, but risk-aware scoring is needed to preserve reliable MS-CLAP anchors. The failure row shows that temporal gain can still be over-trusted, motivating better temporal-risk calibration.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python src/render_tsel_paper_case_panel.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_summaries", default="docs/paper_assets/figures/tsel_case_figures/case_summaries.json")
    parser.add_argument("--evidence_path", default="results/evidence_baseline_release_v1/full_test_eval/predictions_evidence_samples.json")
    parser.add_argument("--output_svg", default="docs/paper_assets/figures/tsel_case_figures/paper_case_panel.svg")
    parser.add_argument("--usage_md", default="docs/paper_assets/figures/tsel_case_figures/paper_case_panel_notes.md")
    args = parser.parse_args()

    cases = load_json(args.case_summaries)
    durations = infer_durations(args.evidence_path)
    os.makedirs(os.path.dirname(args.output_svg), exist_ok=True)
    save_text(args.output_svg, render_panel(cases, durations))
    save_text(args.usage_md, render_usage_md(args.output_svg, cases))
    print(f"saved {args.output_svg}")
    print(f"saved {args.usage_md}")


if __name__ == "__main__":
    main()
