import argparse
import json
import os
from collections import Counter, defaultdict
from xml.sax.saxutils import escape

from metrics import best_iou_for_window
from train_learned_evidence_decoder import SOURCE_NAMES, generate_candidates, load_items


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def fmt_pct(value):
    return f"{100.0 * float(value):.2f}%"


def fmt_float(value):
    return f"{float(value):.4f}"


def table(headers, rows):
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def candidate_type(window, gt_windows):
    iou = best_iou_for_window(window, gt_windows)
    st, ed = float(window[0]), float(window[1])
    length = max(ed - st, 1e-6)
    best_gt = None
    best_iou = 0.0
    for gt in gt_windows:
        cur = best_iou_for_window(window, [gt])
        if cur > best_iou:
            best_iou = cur
            best_gt = gt
    if iou >= 0.7:
        return "good"
    if iou < 0.1 or best_gt is None:
        return "semantic_false"
    gt_st, gt_ed = float(best_gt[0]), float(best_gt[1])
    gt_len = max(gt_ed - gt_st, 1e-6)
    inter = max(0.0, min(ed, gt_ed) - max(st, gt_st))
    if inter / gt_len >= 0.75 and length > gt_len * 1.25:
        return "over_wide"
    if inter / length >= 0.75 and length < gt_len * 0.75:
        return "under_wide"
    return "boundary_distractor"


def analyze_items(items, topn_per_source):
    source_rows = defaultdict(list)
    rank_rows = defaultdict(list)
    sample_best = {}
    sample_oracle = {}
    raw_rows = []

    for item in items:
        qid = item["qid"]
        gt_windows = item["gt_windows"]
        candidates = generate_candidates(item, topn_per_source=topn_per_source)
        sample_oracle[qid] = max((best_iou_for_window(cand["window"], gt_windows) for cand in candidates), default=0.0)
        best_for_qid = None
        for cand in candidates:
            iou = best_iou_for_window(cand["window"], gt_windows)
            record = {
                "qid": qid,
                "source": cand["source"],
                "rank": int(cand["source_rank"]),
                "window": cand["window"],
                "iou": iou,
                "candidate_type": candidate_type(cand["window"], gt_windows),
            }
            source_rows[cand["source"]].append(record)
            rank_rows[(cand["source"], int(cand["source_rank"]))].append(record)
            raw_rows.append(record)
            if best_for_qid is None or iou > best_for_qid["iou"]:
                best_for_qid = record
        if best_for_qid is not None:
            sample_best[qid] = best_for_qid

    total_samples = len(items)
    source_summary = {}
    for source in SOURCE_NAMES:
        rows = source_rows.get(source, [])
        by_qid = defaultdict(list)
        for row in rows:
            by_qid[row["qid"]].append(row)
        best_ious = [max(row["iou"] for row in q_rows) for q_rows in by_qid.values()]
        type_counts = Counter(row["candidate_type"] for row in rows)
        source_summary[source] = {
            "candidates": len(rows),
            "samples_with_candidate": len(by_qid),
            "oracle_r1_05": sum(1 for iou in best_ious if iou >= 0.5) / max(total_samples, 1),
            "oracle_r1_07": sum(1 for iou in best_ious if iou >= 0.7) / max(total_samples, 1),
            "avg_best_iou": mean(best_ious),
            "avg_candidate_iou": mean([row["iou"] for row in rows]),
            "good_rate": sum(1 for row in rows if row["iou"] >= 0.7) / max(len(rows), 1),
            "best_source_wins": sum(1 for row in sample_best.values() if row["source"] == source),
            "candidate_types": dict(type_counts),
        }

    rank_summary = {}
    for (source, rank), rows in sorted(rank_rows.items()):
        rank_summary[f"{source}:{rank}"] = {
            "source": source,
            "rank": rank,
            "candidates": len(rows),
            "avg_iou": mean([row["iou"] for row in rows]),
            "r05_rate": sum(1 for row in rows if row["iou"] >= 0.5) / max(len(rows), 1),
            "r07_rate": sum(1 for row in rows if row["iou"] >= 0.7) / max(len(rows), 1),
            "semantic_false_rate": sum(1 for row in rows if row["candidate_type"] == "semantic_false") / max(len(rows), 1),
            "over_wide_rate": sum(1 for row in rows if row["candidate_type"] == "over_wide") / max(len(rows), 1),
            "under_wide_rate": sum(1 for row in rows if row["candidate_type"] == "under_wide") / max(len(rows), 1),
        }

    oracle_ious = list(sample_oracle.values())
    return {
        "num_samples": total_samples,
        "topn_per_source": topn_per_source,
        "oracle_any_source": {
            "R@0.5": sum(1 for iou in oracle_ious if iou >= 0.5) / max(total_samples, 1),
            "R@0.7": sum(1 for iou in oracle_ious if iou >= 0.7) / max(total_samples, 1),
            "avg_best_iou": mean(oracle_ious),
        },
        "source_summary": source_summary,
        "rank_summary": rank_summary,
        "raw_rows": raw_rows,
    }


def recommended_quotas(rank_summary, max_rank):
    quotas = {}
    for source in SOURCE_NAMES:
        keep = 0
        for rank in range(max_rank):
            stats = rank_summary.get(f"{source}:{rank}", {})
            if not stats:
                continue
            useful = (
                stats.get("r07_rate", 0.0) >= 0.025
                or stats.get("r05_rate", 0.0) >= 0.08
                or stats.get("avg_iou", 0.0) >= 0.22
            )
            too_noisy = stats.get("semantic_false_rate", 0.0) >= 0.75 and stats.get("r05_rate", 0.0) < 0.04
            if useful and not too_noisy:
                keep = rank + 1
        quotas[source] = max(1, min(keep, max_rank))
    return quotas


def write_html(path, result, quotas):
    source_rows = []
    for source, stats in sorted(result["source_summary"].items(), key=lambda item: item[1]["oracle_r1_07"], reverse=True):
        source_rows.append(
            [
                escape(source),
                str(stats["candidates"]),
                fmt_pct(stats["oracle_r1_05"]),
                fmt_pct(stats["oracle_r1_07"]),
                fmt_float(stats["avg_best_iou"]),
                fmt_float(stats["avg_candidate_iou"]),
                fmt_pct(stats["good_rate"]),
                str(stats["best_source_wins"]),
                str(quotas.get(source, 0)),
            ]
        )

    rank_rows = []
    for key, stats in sorted(result["rank_summary"].items(), key=lambda item: (item[1]["source"], item[1]["rank"])):
        rank_rows.append(
            [
                escape(stats["source"]),
                str(stats["rank"]),
                str(stats["candidates"]),
                fmt_float(stats["avg_iou"]),
                fmt_pct(stats["r05_rate"]),
                fmt_pct(stats["r07_rate"]),
                fmt_pct(stats["semantic_false_rate"]),
                fmt_pct(stats["over_wide_rate"]),
                fmt_pct(stats["under_wide_rate"]),
            ]
        )

    quota_text = ", ".join(f"{source}={quota}" for source, quota in quotas.items())
    oracle = result["oracle_any_source"]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Candidate Source Diagnosis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eceff3; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Candidate Source Diagnosis</h1>
  <div class="note">
    Any-source oracle R@0.5={fmt_pct(oracle["R@0.5"])}, R@0.7={fmt_pct(oracle["R@0.7"])}, avg best IoU={fmt_float(oracle["avg_best_iou"])}.
    Recommended quota string: <code>{escape(quota_text)}</code>
  </div>
  <h2>Source Oracle Quality</h2>
  {table(["Source", "Candidates", "Oracle R@0.5", "Oracle R@0.7", "Avg Best IoU", "Avg Candidate IoU", "Candidate Good Rate", "Best-Source Wins", "Recommended Quota"], source_rows)}
  <h2>Rank Quality By Source</h2>
  {table(["Source", "Rank", "Candidates", "Avg IoU", "R@0.5 Rate", "R@0.7 Rate", "Semantic False Rate", "Over-Wide Rate", "Under-Wide Rate"], rank_rows)}
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topn_per_source", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    items = load_items(args.evidence_path)
    result = analyze_items(items, topn_per_source=args.topn_per_source)
    quotas = recommended_quotas(result["rank_summary"], args.topn_per_source)
    result["recommended_quotas"] = quotas
    save_json({k: v for k, v in result.items() if k != "raw_rows"}, os.path.join(args.output_dir, "summary.json"))
    save_json(result["raw_rows"], os.path.join(args.output_dir, "candidate_rows.json"))
    write_html(os.path.join(args.output_dir, "report.html"), result, quotas)
    print(json.dumps({k: v for k, v in result.items() if k != "raw_rows"}, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
