import json
import os
import subprocess
from html import escape
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "docs", "paper_assets", "figures", "academic")
CASE_JSON = os.path.join(
    ROOT,
    "docs",
    "paper_assets",
    "figures",
    "tsel_case_figures",
    "case_summaries.json",
)


OKABE_ITO = {
    "gt": "#000000",
    "ms": "#D55E00",
    "ecf": "#0072B2",
    "raes": "#CC79A7",
    "grid": "#D9D9D9",
    "axis": "#4D4D4D",
    "text": "#111111",
    "muted": "#555555",
}


CASE_LABELS = {
    "semantic_recovery": "Semantic recovery",
    "temporal_correction": "Boundary correction",
    "anchor_protection": "Anchor protection",
    "temporal_risk_failure": "Remaining failure",
}


HEADS = {
    "semantic_recovery": [("sem. gain", "pred_semantic_gain"), ("anchor rel.", "pred_anchor_reliability")],
    "temporal_correction": [("temp. gain", "pred_temporal_gain"), ("strict gain", "pred_strict_gain")],
    "anchor_protection": [("anchor rel.", "pred_anchor_reliability"), ("anchor risk", "pred_anchor_risk")],
    "temporal_risk_failure": [("temp. gain", "pred_temporal_gain"), ("temp. risk", "pred_temporal_risk")],
}


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def find_sharp_module():
    home = Path.home()
    roots = [
        home / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "node_modules" / ".pnpm",
        home / "AppData" / "Roaming" / "npm" / "node_modules",
    ]
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(root.glob("sharp@*/node_modules/sharp"), reverse=True)
        if matches:
            return str(matches[0])
    return None


def load_cases():
    with open(CASE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def case_duration(case):
    max_t = 1.0
    for span in case.get("gt_windows", []) or []:
        max_t = max(max_t, float(span[1]))
    for key in ("ms_window", "ecf_window", "raes_window"):
        span = case.get(key) or []
        if len(span) >= 2:
            max_t = max(max_t, float(span[1]))
    return max_t


def fmt(value):
    return f"{float(value):.2f}"


def render_framework():
    dot = r'''
digraph TSEL {
  graph [
    rankdir=LR,
    bgcolor="white",
    margin=0.08,
    nodesep=0.42,
    ranksep=0.55,
    splines=ortho,
    outputorder=edgesfirst
  ];
  node [
    shape=box,
    style="rounded,filled",
    penwidth=1.2,
    color="#333333",
    fillcolor="#F7F7F7",
    fontname="Helvetica",
    fontsize=16,
    margin="0.12,0.08"
  ];
  edge [
    color="#333333",
    arrowsize=0.65,
    penwidth=1.1,
    fontname="Helvetica",
    fontsize=12
  ];

  audio [label="Long audio A"];
  query [label="Text query q"];
  clap [label="Shared MS-CLAP features\n(audio frames + text embedding)", fillcolor="#FFFFFF"];

  mseb [label="MS-EB\nstable semantic evidence", fillcolor="#EDEDED"];
  tsa [label="TSA\nwithin-audio temporal adapter", fillcolor="#EDEDED"];
  sbec [label="SBEC\nsemantic-to-boundary curriculum", fillcolor="#EDEDED"];

  sfp [label="SFP negatives\nsemantic false peaks", shape=note, fillcolor="#FFFFFF"];
  bwel [label="BWEL negatives\nboundary / width errors", shape=note, fillcolor="#FFFFFF"];

  evidence [label="Query-conditioned evidence\nS(t,q)", fillcolor="#FFFFFF"];
  candidates [label="Candidate windows\nC = {[s,e], features}", fillcolor="#FFFFFF"];

  ecf [label="TSEL-ECF\ncandidate-level fusion", fillcolor="#F1F1F1"];
  raes [label="TSEL-RAES\nrisk-aware scoring\nsemantic, temporal, anchor, risk", fillcolor="#F1F1F1"];
  output [label="Ranked moment\n+ evidence explanation", fillcolor="#FFFFFF"];

  audio -> clap;
  query -> clap;
  clap -> mseb;
  clap -> tsa;
  tsa -> sbec;
  sfp -> sbec [style=dashed];
  bwel -> sbec [style=dashed];
  mseb -> evidence;
  sbec -> evidence;
  evidence -> candidates;
  candidates -> ecf;
  candidates -> raes;
  ecf -> output;
  raes -> output;

  {rank=same; audio; query}
  {rank=same; mseb; tsa}
  {rank=same; ecf; raes}
}
'''
    dot_path = os.path.join(OUT_DIR, "fig1_tsel_framework_academic.dot")
    write(dot_path, dot.strip() + "\n")
    for ext in ("svg", "png", "pdf"):
        out_path = os.path.join(OUT_DIR, f"fig1_tsel_framework_academic.{ext}")
        subprocess.run(["dot", f"-T{ext}", dot_path, "-o", out_path], check=True)


def svg_text(x, y, text, size=18, weight="normal", anchor="start", color=None):
    color = color or OKABE_ITO["text"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{escape(str(text))}</text>'
    )


def wrapped_text(parts, x, y, content, limit=34, line_gap=22, size=17, color=None):
    words = str(content).split()
    lines = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) <= limit:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for idx, line in enumerate(lines[:2]):
        parts.append(svg_text(x, y + idx * line_gap, line, size, color=color))


def svg_line(x1, y1, x2, y2, color, width=2, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"{dash_attr}/>'
    )


def interval(parts, x0, x1, y, color, label, plot_right):
    parts.append(svg_line(x0, y, x1, y, color, 7))
    parts.append(svg_line(x0, y - 9, x0, y + 9, color, 2))
    parts.append(svg_line(x1, y - 9, x1, y + 9, color, 2))
    if label:
        if x0 < 470 and x1 > plot_right - 105:
            parts.append(svg_text(x0 + 8, y + 5, label, 13, color=color))
        elif x1 > plot_right - 105:
            parts.append(svg_text(x0 - 8, y + 5, label, 13, anchor="end", color=color))
        else:
            parts.append(svg_text(x1 + 8, y + 5, label, 13, color=color))


def render_cases():
    cases = load_cases()
    width = 1800
    height = 1180
    left = 440
    right = 1415
    top = 120
    row_h = 250
    lane_offsets = {"GT": 48, "MS": 88, "ECF": 128, "RAES": 168}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="1800" height="1180" fill="#FFFFFF"/>',
        svg_text(48, 52, "Qualitative error analysis with normalized temporal windows", 28, "700"),
        svg_text(48, 84, "Each row is normalized by clip duration; intervals show ground truth and top-1 candidates.", 17, color=OKABE_ITO["muted"]),
    ]
    legend_x = 1080
    for idx, (name, color) in enumerate([("GT", OKABE_ITO["gt"]), ("MS anchor", OKABE_ITO["ms"]), ("TSEL-ECF", OKABE_ITO["ecf"]), ("TSEL-RAES", OKABE_ITO["raes"])]):
        x = legend_x + idx * 160
        parts.append(svg_line(x, 76, x + 42, 76, color, 7))
        parts.append(svg_text(x + 52, 81, name, 16, color=OKABE_ITO["text"]))

    for i, case in enumerate(cases):
        y0 = top + i * row_h
        duration = case_duration(case)
        parts.append(svg_line(46, y0 - 26, 1752, y0 - 26, "#E6E6E6", 1))
        parts.append(svg_text(60, y0 + 4, f"({chr(97 + i)}) {CASE_LABELS.get(case['label'], case['label'])}", 20, "700"))
        parts.append(svg_text(60, y0 + 32, case["qid"], 14, color=OKABE_ITO["muted"]))
        wrapped_text(parts, 60, y0 + 60, case["query"], limit=31, size=17)

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = left + frac * (right - left)
            parts.append(svg_line(x, y0 + 28, x, y0 + 188, OKABE_ITO["grid"], 1))
            parts.append(svg_text(x, y0 + 216, f"{frac:.2f}", 13, anchor="middle", color=OKABE_ITO["muted"]))
        parts.append(svg_line(left, y0 + 188, right, y0 + 188, OKABE_ITO["axis"], 1.5))
        parts.append(svg_text((left + right) / 2, y0 + 238, "normalized time", 13, anchor="middle", color=OKABE_ITO["muted"]))

        for lane, offset in lane_offsets.items():
            parts.append(svg_text(left - 46, y0 + offset + 5, lane, 14, anchor="end", color=OKABE_ITO["muted"]))

        def xpos(t):
            return left + min(max(float(t) / duration, 0.0), 1.0) * (right - left)

        for gt_idx, span in enumerate(case.get("gt_windows", []) or []):
            interval(parts, xpos(span[0]), xpos(span[1]), y0 + lane_offsets["GT"], OKABE_ITO["gt"], "", right)

        interval(parts, xpos(case["ms_window"][0]), xpos(case["ms_window"][1]), y0 + lane_offsets["MS"], OKABE_ITO["ms"], "", right)
        interval(parts, xpos(case["ecf_window"][0]), xpos(case["ecf_window"][1]), y0 + lane_offsets["ECF"], OKABE_ITO["ecf"], "", right)
        interval(parts, xpos(case["raes_window"][0]), xpos(case["raes_window"][1]), y0 + lane_offsets["RAES"], OKABE_ITO["raes"], "", right)

        info_x = right + 48
        parts.append(svg_text(info_x, y0 + 44, "RAES diagnostics", 15, "700"))
        parts.append(svg_text(info_x, y0 + 74, f"IoU MS/ECF/RAES: {fmt(case['ms_top1_iou'])}/{fmt(case['ecf_top1_iou'])}/{fmt(case['raes_top1_iou'])}", 14, color=OKABE_ITO["muted"]))
        for j, (label, key) in enumerate(HEADS.get(case["label"], [])):
            parts.append(svg_text(info_x, y0 + 104 + 24 * j, f"{label}: {fmt(case['raes_heads'][key])}", 14, color=OKABE_ITO["muted"]))
        parts.append(svg_text(info_x, y0 + 166, f"MS [{case['ms_window'][0]:.0f}, {case['ms_window'][1]:.0f}]", 13, color=OKABE_ITO["muted"]))
        parts.append(svg_text(info_x, y0 + 190, f"RAES [{case['raes_window'][0]:.0f}, {case['raes_window'][1]:.0f}]", 13, color=OKABE_ITO["muted"]))

    parts.append("</svg>")
    svg_path = os.path.join(OUT_DIR, "fig2_tsel_cases_academic.svg")
    write(svg_path, "\n".join(parts) + "\n")
    sharp_module = find_sharp_module()
    if sharp_module:
        png_path = os.path.join(OUT_DIR, "fig2_tsel_cases_academic.png")
        js = """
const sharp = require(process.argv[1]);
const input = process.argv[2];
const output = process.argv[3];
sharp(input, { density: 220 }).png().toFile(output).catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
        subprocess.run(["node", "-e", js, sharp_module, svg_path, png_path], check=True)
    else:
        print("warning: sharp is unavailable; wrote SVG only for fig2")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    render_framework()
    render_cases()
    notes = """# Academic Figure Assets

Generated by `python src/render_academic_paper_figures.py`.

- `fig1_tsel_framework_academic.pdf/png/svg`: method overview generated with Graphviz.
- `fig2_tsel_cases_academic.svg/png`: normalized temporal case analysis generated from `case_summaries.json`.

The case figure uses normalized time because the four examples have different
clip durations. Numeric IoU values and RAES diagnostics come directly from the
frozen case summaries.
"""
    write(os.path.join(OUT_DIR, "README.md"), notes)
    print(f"saved academic figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
