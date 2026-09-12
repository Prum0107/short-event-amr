import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from xml.sax.saxutils import escape

from metrics import best_iou_among_topk, best_iou_for_window, recall_at_1_iou, recall_at_k_iou


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def smooth_scores(scores, kernel=1):
    if kernel <= 1:
        return list(scores)
    pad = kernel // 2
    padded = [scores[0]] * pad + list(scores) + [scores[-1]] * pad
    out = []
    for idx in range(len(scores)):
        out.append(sum(padded[idx : idx + kernel]) / kernel)
    return out


def percentile(values, pct):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * max(0.0, min(100.0, pct)) / 100.0
    low = int(math.floor(k))
    high = int(math.ceil(k))
    if low == high:
        return sorted_values[low]
    alpha = k - low
    return sorted_values[low] * (1 - alpha) + sorted_values[high] * alpha


def window_iou(a, b):
    if not a or not b:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / max(union, 1e-6)


def nms_windows(windows, threshold=0.7, topn=10):
    kept = []
    for window in sorted(windows, key=lambda x: x[2], reverse=True):
        if window[1] <= window[0]:
            continue
        if all(window_iou(window, old) < threshold for old in kept):
            kept.append(window)
        if len(kept) >= topn:
            break
    return kept


def score_values(values, mode):
    if not values:
        return -1e9
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "max":
        return max(values)
    if mode == "top25":
        k = max(1, int(math.ceil(len(values) * 0.25)))
        return sum(sorted(values, reverse=True)[:k]) / k
    if mode == "sum":
        return sum(values)
    raise ValueError(f"unknown score mode: {mode}")


def prefix_sum(scores):
    prefix = [0.0]
    for score in scores:
        prefix.append(prefix[-1] + score)
    return prefix


def range_mean(prefix, st, ed):
    st = max(0, st)
    ed = min(len(prefix) - 1, ed)
    if ed <= st:
        return 0.0
    return (prefix[ed] - prefix[st]) / (ed - st)


def threshold_segments(
    scores,
    threshold,
    topn=10,
    min_len=1,
    max_len=150,
    pad=0,
    close_gap=0,
    score_mode="mean",
    nms=0.7,
):
    positive = [score >= threshold for score in scores]
    if close_gap > 0:
        idx = 0
        while idx < len(positive):
            if positive[idx]:
                idx += 1
                continue
            gap_st = idx
            while idx < len(positive) and not positive[idx]:
                idx += 1
            gap_ed = idx
            left_on = gap_st > 0 and positive[gap_st - 1]
            right_on = idx < len(positive) and positive[idx]
            if left_on and right_on and gap_ed - gap_st <= close_gap:
                for fill_idx in range(gap_st, gap_ed):
                    positive[fill_idx] = True

    windows = []
    st = None
    for idx, flag in enumerate(positive + [False]):
        if flag and st is None:
            st = idx
        if not flag and st is not None:
            ed = idx
            if ed - st >= min_len:
                left = max(0, st - pad)
                right = min(len(scores), ed + pad)
                if max_len > 0 and right - left > max_len:
                    center = (left + right) // 2
                    left = max(0, center - max_len // 2)
                    right = min(len(scores), left + max_len)
                    left = max(0, right - max_len)
                value = score_values(scores[left:right], score_mode)
                windows.append([left, right, value])
            st = None
    return nms_windows(windows, threshold=nms, topn=topn)


def decoder_threshold_mean(scores, topn=10):
    work = smooth_scores(scores, kernel=3)
    mean = sum(work) / max(len(work), 1)
    return threshold_segments(work, threshold=mean, topn=topn, min_len=1, max_len=150, score_mode="top25")


def make_decoder_threshold_percentile(pct):
    def _decoder(scores, topn=10):
        work = smooth_scores(scores, kernel=3)
        return threshold_segments(
            work,
            threshold=percentile(work, pct),
            topn=topn,
            min_len=1,
            max_len=150,
            close_gap=1,
            score_mode="top25",
        )

    return _decoder


def decoder_peak_expand(scores, topn=10, peak_pct=80.0, floor_pct=55.0):
    work = smooth_scores(scores, kernel=5)
    peak_floor = percentile(work, peak_pct)
    expand_floor = percentile(work, floor_pct)
    peak_indices = sorted(range(len(work)), key=lambda idx: work[idx], reverse=True)
    windows = []
    for peak in peak_indices:
        if work[peak] < peak_floor:
            break
        left = peak
        right = peak + 1
        while left > 0 and work[left - 1] >= expand_floor:
            left -= 1
        while right < len(work) and work[right] >= expand_floor:
            right += 1
        if right - left > 150:
            half = 75
            left = max(0, peak - half)
            right = min(len(work), left + 150)
            left = max(0, right - 150)
        score = score_values(work[left:right], "top25") + 0.1 * work[peak]
        windows.append([left, right, score])
    return nms_windows(windows, threshold=0.7, topn=topn)


def decoder_peak_drop(scores, topn=10, drop=0.18, min_floor_pct=45.0):
    work = smooth_scores(scores, kernel=5)
    min_floor = percentile(work, min_floor_pct)
    peak_indices = sorted(range(len(work)), key=lambda idx: work[idx], reverse=True)[: max(40, topn * 8)]
    windows = []
    for peak in peak_indices:
        floor = max(min_floor, work[peak] - drop)
        left = peak
        right = peak + 1
        while left > 0 and work[left - 1] >= floor:
            left -= 1
        while right < len(work) and work[right] >= floor:
            right += 1
        if right - left < 1:
            continue
        score = score_values(work[left:right], "mean") + 0.15 * score_values(work[left:right], "top25")
        windows.append([left, right, score])
    return nms_windows(windows, threshold=0.7, topn=topn)


def make_decoder_multiscale(windows):
    def _decoder(scores, topn=10):
        work = smooth_scores(scores, kernel=3)
        peak_indices = sorted(range(len(work)), key=lambda idx: work[idx], reverse=True)[:40]
        proposals = []
        for peak in peak_indices:
            for window_len in windows:
                left = max(0, peak - window_len // 2)
                right = min(len(work), left + window_len)
                left = max(0, right - window_len)
                values = work[left:right]
                score = score_values(values, "top25") + 0.05 * score_values(values, "mean")
                proposals.append([left, right, score])
        return nms_windows(proposals, threshold=0.7, topn=topn)

    return _decoder


def make_decoder_dense_window(score_mode="mean", length_penalty=0.0, contrast=0.0):
    def _decoder(scores, topn=10):
        work = smooth_scores(scores, kernel=3)
        prefix = prefix_sum(work)
        windows = []
        max_len = min(150, len(work))
        for st in range(len(work)):
            for ed in range(st + 1, min(len(work), st + max_len) + 1):
                values = work[st:ed]
                base = score_values(values, score_mode)
                length = ed - st
                outside_left = range_mean(prefix, max(0, st - length), st)
                outside_right = range_mean(prefix, ed, min(len(work), ed + length))
                outside = 0.5 * (outside_left + outside_right)
                score = base - length_penalty * (length / 150.0) + contrast * (base - outside)
                windows.append([st, ed, score])
        return nms_windows(windows, threshold=0.7, topn=topn)

    return _decoder


DECODERS = {
    "current_start_end": lambda item, topn: item.get("pred_relevant_windows", [])[:topn],
    "threshold_mean": lambda item, topn: decoder_threshold_mean(item["evidence_scores"], topn=topn),
    "threshold_p60": lambda item, topn: make_decoder_threshold_percentile(60)(item["evidence_scores"], topn=topn),
    "threshold_p70": lambda item, topn: make_decoder_threshold_percentile(70)(item["evidence_scores"], topn=topn),
    "peak_expand": lambda item, topn: decoder_peak_expand(item["evidence_scores"], topn=topn),
    "peak_drop": lambda item, topn: decoder_peak_drop(item["evidence_scores"], topn=topn),
    "multiscale_10_20_40_80_150": lambda item, topn: make_decoder_multiscale([10, 20, 40, 80, 150])(
        item["evidence_scores"], topn=topn
    ),
    "dense_mean": lambda item, topn: make_decoder_dense_window("mean")(item["evidence_scores"], topn=topn),
    "dense_contrast": lambda item, topn: make_decoder_dense_window("mean", length_penalty=0.05, contrast=0.4)(
        item["evidence_scores"], topn=topn
    ),
}


def categorize(windows, gt_windows, evidence_gap, iou_good=0.7, iou_partial=0.1):
    top1 = windows[0] if windows else None
    top1_iou = best_iou_for_window(top1, gt_windows) if top1 else 0.0
    top5_iou = max((best_iou_for_window(pred, gt_windows) for pred in windows[:5]), default=0.0)
    overlap = 0.0
    if top1:
        overlap = max(max(0.0, min(top1[1], gt[1]) - max(top1[0], gt[0])) for gt in gt_windows)
    if top1_iou >= iou_good:
        return "good"
    if top5_iou >= iou_good:
        return "candidate_exists"
    if overlap > 0 or top1_iou >= iou_partial:
        return "boundary_error"
    if evidence_gap >= 0.1:
        return "evidence_good_decode_bad"
    return "semantic_miss"


def evidence_gap(item):
    scores = item["evidence_scores"]
    if not scores:
        return 0.0
    mask = [False] * len(scores)
    for st, ed in item.get("gt_windows", []):
        for idx in range(max(0, int(st)), min(len(scores), int(ed))):
            mask[idx] = True
    pos = [score for score, keep in zip(scores, mask) if keep]
    neg = [score for score, keep in zip(scores, mask) if not keep]
    if not pos or not neg:
        return 0.0
    return sum(pos) / len(pos) - sum(neg) / len(neg)


def evaluate_decoder(items, decoder_name, topn):
    rows = []
    totals = defaultdict(float)
    categories = Counter()
    predictions = []
    for item in items:
        gt_windows = item["gt_windows"]
        windows = DECODERS[decoder_name](item, topn)
        gap = evidence_gap(item)
        category = categorize(windows, gt_windows, gap)
        categories[category] += 1
        top1_iou = best_iou_for_window(windows[0], gt_windows) if windows else 0.0
        top5_iou = max((best_iou_for_window(pred, gt_windows) for pred in windows[:5]), default=0.0)
        totals["R1@0.5"] += recall_at_1_iou(windows, gt_windows, threshold=0.5)
        totals["R1@0.7"] += recall_at_1_iou(windows, gt_windows, threshold=0.7)
        totals["R3@0.5"] += recall_at_k_iou(windows, gt_windows, threshold=0.5, k=3)
        totals["R3@0.7"] += recall_at_k_iou(windows, gt_windows, threshold=0.7, k=3)
        totals["R5@0.7"] += recall_at_k_iou(windows, gt_windows, threshold=0.7, k=5)
        totals["best_iou_top5"] += best_iou_among_topk(windows, gt_windows, k=5)
        totals["top1_iou"] += top1_iou
        totals["top5_iou"] += top5_iou
        rows.append(
            {
                "qid": item["qid"],
                "query": item.get("query", ""),
                "vid": item.get("vid", ""),
                "gt_windows": gt_windows,
                "windows": windows,
                "top1_iou": top1_iou,
                "top5_iou": top5_iou,
                "category": category,
                "evidence_gap": gap,
            }
        )
        predictions.append(
            {
                "qid": item["qid"],
                "query": item.get("query", ""),
                "vid": item.get("vid", ""),
                "pred_relevant_windows": windows,
            }
        )
    n = max(len(items), 1)
    metrics = {key: value / n for key, value in totals.items()}
    metrics["num_samples"] = len(items)
    metrics["categories"] = dict(categories)
    return metrics, rows, predictions


def fmt_pct(value):
    return f"{100 * value:.2f}%"


def fmt(value):
    return f"{value:.4f}"


def svg_for_comparison(item, current_windows, best_windows, width=1180, height=320):
    scores = item["evidence_scores"]
    duration = max(float(item.get("duration", len(scores))), float(len(scores)))
    x0, y0, plot_w, plot_h = 66, 56, 1060, 205
    x_scale = plot_w / max(duration, 1.0)

    def x_at(t):
        return x0 + float(t) * x_scale

    def y_at(v):
        return y0 + plot_h - max(0.0, min(1.0, float(v))) * plot_h

    def rect(span, color, opacity):
        if not span:
            return ""
        x = x_at(span[0])
        w = max(1.0, (span[1] - span[0]) * x_scale)
        return f'<rect x="{x:.2f}" y="{y0}" width="{w:.2f}" height="{plot_h}" fill="{color}" opacity="{opacity}"/>'

    points = " ".join(f"{x_at(i + 0.5):.2f},{y_at(score):.2f}" for i, score in enumerate(scores))
    gt = "\n".join(rect(span, "#2ca02c", 0.22) for span in item.get("gt_windows", []))
    current = rect(current_windows[0], "#d62728", 0.16) if current_windows else ""
    best = rect(best_windows[0], "#9467bd", 0.18) if best_windows else ""
    query = escape(str(item.get("query", ""))[:140])
    qid = escape(str(item.get("qid", "")))

    grid = []
    for val in [0.0, 0.5, 1.0]:
        y = y_at(val)
        grid.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}" stroke="#d0d0d0"/>')
        grid.append(f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{val:.1f}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="white"/>
<text x="{x0}" y="24" font-size="14" font-family="Arial" fill="#222">{qid}: {query}</text>
<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#999"/>
{gt}
{current}
{best}
{chr(10).join(grid)}
<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="2"/>
<text x="{x0}" y="{height - 30}" font-size="12" font-family="Arial" fill="#333">GT green, current red, best decoder purple, blue S(t,q)</text>
</svg>"""


def write_comparison_svgs(items_by_qid, current_rows, best_rows, out_dir, max_items=12):
    clean_dir(out_dir)
    current_by_qid = {row["qid"]: row for row in current_rows}
    improvements = []
    regressions = []
    for best in best_rows:
        current = current_by_qid[best["qid"]]
        delta = best["top1_iou"] - current["top1_iou"]
        record = (delta, best["qid"], current, best)
        if delta > 0.2:
            improvements.append(record)
        elif delta < -0.2:
            regressions.append(record)
    for folder, records, reverse in [
        ("improvements", improvements, True),
        ("regressions", regressions, False),
    ]:
        folder_path = os.path.join(out_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        records = sorted(records, key=lambda x: x[0], reverse=reverse)[:max_items]
        for idx, (delta, qid, current, best) in enumerate(records):
            item = items_by_qid[qid]
            path = os.path.join(folder_path, f"{idx:03d}_{qid}.svg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg_for_comparison(item, current["windows"], best["windows"]))


def html_table(headers, rows):
    thead = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html_report(output_dir, metrics_by_decoder, best_name):
    rows = []
    for name, metrics in sorted(metrics_by_decoder.items(), key=lambda x: x[1]["R1@0.7"], reverse=True):
        rows.append(
            [
                f"<b>{escape(name)}</b>" if name == best_name else escape(name),
                fmt_pct(metrics["R1@0.5"]),
                fmt_pct(metrics["R1@0.7"]),
                fmt_pct(metrics["R3@0.7"]),
                fmt_pct(metrics["R5@0.7"]),
                fmt(metrics["best_iou_top5"]),
                fmt(metrics["top1_iou"]),
            ]
        )

    category_names = ["good", "boundary_error", "candidate_exists", "evidence_good_decode_bad", "semantic_miss"]
    cat_rows = []
    for name, metrics in sorted(metrics_by_decoder.items(), key=lambda x: x[1]["R1@0.7"], reverse=True):
        n = max(metrics["num_samples"], 1)
        cat_rows.append(
            [escape(name)]
            + [f"{metrics['categories'].get(cat, 0)} ({100 * metrics['categories'].get(cat, 0) / n:.1f}%)" for cat in category_names]
        )

    improvement_dir = "comparison_svgs/improvements"
    regression_dir = "comparison_svgs/regressions"
    improvement_imgs = []
    regression_imgs = []
    abs_improvement = os.path.join(output_dir, improvement_dir)
    abs_regression = os.path.join(output_dir, regression_dir)
    if os.path.isdir(abs_improvement):
        for name in sorted(os.listdir(abs_improvement))[:12]:
            improvement_imgs.append(f'<img src="{improvement_dir}/{escape(name)}" alt="{escape(name)}"/>')
    if os.path.isdir(abs_regression):
        for name in sorted(os.listdir(abs_regression))[:12]:
            regression_imgs.append(f'<img src="{regression_dir}/{escape(name)}" alt="{escape(name)}"/>')

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Evidence Decoder Experiment V1</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; background: #f7f7f5; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .note {{ background: white; border-left: 4px solid #1f77b4; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eeeeea; }}
    img {{ width: 100%; max-width: 1180px; display: block; margin: 14px 0; border: 1px solid #ddd; background: white; }}
    code {{ background: #eee; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Evidence Decoder Experiment V1</h1>
  <div class="note">
    Best decoder by R1@0.7: <b>{escape(best_name)}</b>. This report compares how different rules convert
    the learned evidence curve <code>S(t,q)</code> into temporal boundaries.
  </div>

  <h2>Metric Comparison</h2>
  {html_table(["Decoder", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Best IoU Top5", "Mean Top1 IoU"], rows)}

  <h2>Failure Category Comparison</h2>
  {html_table(["Decoder"] + category_names, cat_rows)}

  <h2>Improvements Over Current Start/End Decoder</h2>
  <p>Green = GT, red = current decoder top1, purple = best decoder top1, blue = evidence curve.</p>
  {''.join(improvement_imgs) if improvement_imgs else '<p>No large improvements found.</p>'}

  <h2>Regressions Compared With Current Start/End Decoder</h2>
  {''.join(regression_imgs) if regression_imgs else '<p>No large regressions found.</p>'}
</body>
</html>
"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--max_svgs", type=int, default=12)
    args = parser.parse_args()

    clean_dir(args.output_dir)
    items = load_items(args.evidence_path)
    items_by_qid = {item["qid"]: item for item in items}

    metrics_by_decoder = {}
    rows_by_decoder = {}
    for decoder_name in DECODERS:
        metrics, rows, predictions = evaluate_decoder(items, decoder_name, args.topn)
        metrics_by_decoder[decoder_name] = metrics
        rows_by_decoder[decoder_name] = rows
        save_json(metrics, os.path.join(args.output_dir, "metrics", f"{decoder_name}.json"))
        save_json(rows, os.path.join(args.output_dir, "case_rows", f"{decoder_name}.json"))
        save_jsonl(predictions, os.path.join(args.output_dir, "predictions", f"{decoder_name}.jsonl"))

    best_name = max(metrics_by_decoder, key=lambda name: metrics_by_decoder[name]["R1@0.7"])
    write_comparison_svgs(
        items_by_qid=items_by_qid,
        current_rows=rows_by_decoder["current_start_end"],
        best_rows=rows_by_decoder[best_name],
        out_dir=os.path.join(args.output_dir, "comparison_svgs"),
        max_items=args.max_svgs,
    )
    save_json(
        {
            "best_decoder": best_name,
            "metrics_by_decoder": metrics_by_decoder,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "summary.json"),
    )
    write_html_report(args.output_dir, metrics_by_decoder, best_name)

    print(json.dumps({"best_decoder": best_name, "metrics": metrics_by_decoder[best_name]}, indent=2))
    print("saved HTML report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
