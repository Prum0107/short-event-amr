import argparse
import json
import os
import shutil
from collections import Counter

from metrics import best_iou_for_window
from visualize_evidence_curves import make_svg


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def interval_overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def max_overlap(pred, gt_windows):
    if not pred:
        return 0.0
    return max(interval_overlap(pred, gt) for gt in gt_windows)


def mean_score(scores, st, ed):
    st_idx = max(0, int(st))
    ed_idx = min(len(scores), int(ed))
    if ed_idx <= st_idx:
        return 0.0
    values = scores[st_idx:ed_idx]
    return sum(values) / max(len(values), 1)


def gt_evidence_score(item):
    scores = item["evidence_scores"]
    windows = item.get("gt_windows", [])
    if not windows:
        return 0.0
    return max(mean_score(scores, st, ed) for st, ed in windows)


def outside_evidence_score(item):
    scores = item["evidence_scores"]
    windows = item.get("gt_windows", [])
    if not scores:
        return 0.0
    mask = [True] * len(scores)
    for st, ed in windows:
        for idx in range(max(0, int(st)), min(len(scores), int(ed))):
            mask[idx] = False
    values = [score for score, keep in zip(scores, mask) if keep]
    return sum(values) / max(len(values), 1)


def top_prediction(item):
    preds = item.get("pred_relevant_windows", [])
    return preds[0] if preds else None


def case_info(item, iou_good=0.7, iou_partial=0.1, evidence_margin=0.1):
    gt_windows = item.get("gt_windows", [])
    preds = item.get("pred_relevant_windows", [])
    top1 = top_prediction(item)
    top1_iou = best_iou_for_window(top1, gt_windows) if top1 else 0.0
    top5_iou = max((best_iou_for_window(pred, gt_windows) for pred in preds[:5]), default=0.0)
    overlap = max_overlap(top1, gt_windows) if top1 else 0.0
    gt_ev = gt_evidence_score(item)
    out_ev = outside_evidence_score(item)
    ev_gap = gt_ev - out_ev

    if top1_iou >= iou_good:
        category = "good"
    elif top5_iou >= iou_good:
        category = "candidate_exists"
    elif overlap > 0 or top1_iou >= iou_partial:
        category = "boundary_error"
    elif ev_gap >= evidence_margin:
        category = "evidence_good_decode_bad"
    else:
        category = "semantic_miss"

    return {
        "qid": item.get("qid"),
        "query": item.get("query", ""),
        "vid": item.get("vid", ""),
        "category": category,
        "top1_iou": top1_iou,
        "top5_iou": top5_iou,
        "top1_overlap_seconds": overlap,
        "gt_evidence": gt_ev,
        "outside_evidence": out_ev,
        "evidence_gap": ev_gap,
        "gt_windows": gt_windows,
        "top1": top1,
        "top5": preds[:5],
    }


def sort_key(info):
    category = info["category"]
    if category == "good":
        return -info["top1_iou"]
    if category == "candidate_exists":
        return -(info["top5_iou"] - info["top1_iou"])
    if category == "boundary_error":
        return -info["top1_overlap_seconds"]
    if category == "evidence_good_decode_bad":
        return -info["evidence_gap"]
    return info["top1_iou"]


def write_case_svgs(items, infos, output_dir, max_per_category):
    case_dir = os.path.join(output_dir, "case_sets")
    clean_dir(case_dir)
    qid_to_item = {str(item.get("qid")): item for item in items}
    by_category = {}
    for info in infos:
        by_category.setdefault(info["category"], []).append(info)

    for category, category_infos in by_category.items():
        category_dir = os.path.join(case_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        for idx, info in enumerate(sorted(category_infos, key=sort_key)[:max_per_category]):
            item = qid_to_item[str(info["qid"])]
            qid = str(info["qid"]).replace("/", "_").replace("\\", "_")
            out_path = os.path.join(category_dir, f"{idx:03d}_{qid}.svg")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(make_svg(item))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_per_category", type=int, default=12)
    parser.add_argument("--iou_good", type=float, default=0.7)
    parser.add_argument("--iou_partial", type=float, default=0.1)
    parser.add_argument("--evidence_margin", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    items = load_items(args.evidence_path)
    infos = [
        case_info(
            item,
            iou_good=args.iou_good,
            iou_partial=args.iou_partial,
            evidence_margin=args.evidence_margin,
        )
        for item in items
    ]
    counts = Counter(info["category"] for info in infos)
    summary = {
        "num_samples": len(infos),
        "counts": dict(counts),
        "rates": {key: value / max(len(infos), 1) for key, value in counts.items()},
        "mean_top1_iou": sum(info["top1_iou"] for info in infos) / max(len(infos), 1),
        "mean_top5_iou": sum(info["top5_iou"] for info in infos) / max(len(infos), 1),
        "mean_evidence_gap": sum(info["evidence_gap"] for info in infos) / max(len(infos), 1),
        "args": vars(args),
    }

    save_json(summary, os.path.join(args.output_dir, "case_summary.json"))
    save_json(infos, os.path.join(args.output_dir, "case_analysis.json"))
    write_case_svgs(items, infos, args.output_dir, args.max_per_category)

    print(json.dumps(summary, indent=2))
    print("saved case analysis to", args.output_dir)


if __name__ == "__main__":
    main()
