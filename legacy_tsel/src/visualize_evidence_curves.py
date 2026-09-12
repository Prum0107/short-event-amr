import argparse
import json
import os
from xml.sax.saxutils import escape


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(value, low, high):
    return max(low, min(high, value))


def span_rect(span, x_scale, x0, y0, height, color, opacity):
    if len(span) < 2:
        return ""
    x = x0 + span[0] * x_scale
    width = max(1.0, (span[1] - span[0]) * x_scale)
    return (
        f'<rect x="{x:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{color}" opacity="{opacity}"/>'
    )


def make_svg(item):
    scores = item["evidence_scores"]
    duration = float(item.get("duration", len(scores)))
    duration = max(duration, float(len(scores)))

    width = 1200
    height = 360
    x0 = 70
    y0 = 64
    plot_w = 1080
    plot_h = 230
    x_scale = plot_w / max(duration, 1.0)

    def x_at(t):
        return x0 + t * x_scale

    def y_at(v):
        return y0 + plot_h - clamp(v, 0.0, 1.0) * plot_h

    points = []
    for idx, score in enumerate(scores):
        t = idx + 0.5
        points.append(f"{x_at(t):.2f},{y_at(float(score)):.2f}")

    gt_rects = "\n".join(
        span_rect(span, x_scale, x0, y0, plot_h, "#2ca02c", 0.22)
        for span in item.get("gt_windows", [])
    )
    pred_rects = "\n".join(
        span_rect(span, x_scale, x0, y0, plot_h, "#d62728", 0.18)
        for span in item.get("pred_relevant_windows", [])[:1]
    )

    grid = []
    for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_at(val)
        grid.append(
            f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}" '
            f'stroke="#cccccc" stroke-width="1" opacity="0.55"/>'
        )
        grid.append(
            f'<text x="{x0 - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#555">{val:.2f}</text>'
        )
    tick_step = 50
    for t in range(0, int(duration) + 1, tick_step):
        x = x_at(t)
        grid.append(
            f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + plot_h}" '
            f'stroke="#dddddd" stroke-width="1" opacity="0.45"/>'
        )
        grid.append(
            f'<text x="{x:.2f}" y="{y0 + plot_h + 22}" text-anchor="middle" '
            f'font-size="12" fill="#555">{t}</text>'
        )

    title = f"{item.get('qid', '')}: {item.get('query', '')}"
    if len(title) > 150:
        title = title[:147] + "..."
    title = escape(title)

    legend_y = 26
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
<text x="{x0}" y="28" font-size="15" font-family="Arial, sans-serif" fill="#222">{title}</text>
<rect x="{width - 265}" y="{legend_y - 14}" width="14" height="14" fill="#2ca02c" opacity="0.22"/>
<text x="{width - 245}" y="{legend_y - 3}" font-size="12" font-family="Arial, sans-serif" fill="#333">GT</text>
<rect x="{width - 205}" y="{legend_y - 14}" width="14" height="14" fill="#d62728" opacity="0.18"/>
<text x="{width - 185}" y="{legend_y - 3}" font-size="12" font-family="Arial, sans-serif" fill="#333">Top-1</text>
<line x1="{width - 125}" y1="{legend_y - 8}" x2="{width - 85}" y2="{legend_y - 8}" stroke="#1f77b4" stroke-width="3"/>
<text x="{width - 78}" y="{legend_y - 3}" font-size="12" font-family="Arial, sans-serif" fill="#333">S(t, q)</text>
<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#999"/>
{gt_rects}
{pred_rects}
{chr(10).join(grid)}
<polyline points="{' '.join(points)}" fill="none" stroke="#1f77b4" stroke-width="2"/>
<text x="{x0 + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#333">Time (s)</text>
<text x="18" y="{y0 + plot_h / 2}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#333" transform="rotate(-90 18 {y0 + plot_h / 2})">Evidence</text>
</svg>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_items", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    items = load_items(args.evidence_path)
    for idx, item in enumerate(items[: args.max_items]):
        qid = str(item.get("qid", idx)).replace("/", "_").replace("\\", "_")
        out_path = os.path.join(args.output_dir, f"{idx:03d}_{qid}.svg")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(make_svg(item))
    print(f"saved {min(len(items), args.max_items)} evidence SVG plots to {args.output_dir}")


if __name__ == "__main__":
    main()
