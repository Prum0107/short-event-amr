import argparse
import json
import math
import os
from collections import Counter, defaultdict

from evidence_decoder_experiment import DECODERS, score_values
from metrics import iou_1d


DEFAULT_SOURCES = [
    "current_start_end",
    "peak_drop",
    "dense_contrast",
    "threshold_mean",
    "threshold_p60",
    "threshold_p70",
    "multiscale_10_20_40_80_150",
]

TYPE_ORDER = [
    "semantic_false_peak",
    "boundary_distractor",
    "over_wide",
    "under_wide",
]


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def normalize_window(window):
    if not window or len(window) < 2:
        return None
    st = float(window[0])
    ed = float(window[1])
    if ed <= st:
        return None
    return [st, ed]


def window_score(scores, st, ed, mode):
    left = max(0, int(math.floor(float(st))))
    right = min(len(scores), int(math.ceil(float(ed))))
    if right <= left:
        return 0.0
    return float(score_values(scores[left:right], mode))


def best_gt_match(window, gt_windows):
    best = None
    best_iou = -1.0
    best_overlap = 0.0
    for gt in gt_windows:
        overlap = max(0.0, min(window[1], gt[1]) - max(window[0], gt[0]))
        iou = iou_1d(window[0], window[1], gt[0], gt[1])
        if iou > best_iou:
            best = gt
            best_iou = iou
            best_overlap = overlap
    return best, max(best_iou, 0.0), best_overlap


def classify_candidate(window, gt_windows, args):
    matched_gt, iou, overlap = best_gt_match(window, gt_windows)
    if matched_gt is None:
        return None, None

    pred_len = max(window[1] - window[0], 1e-6)
    gt_len = max(matched_gt[1] - matched_gt[0], 1e-6)
    gt_coverage = overlap / gt_len
    pred_coverage = overlap / pred_len

    if iou < args.semantic_max_iou:
        candidate_type = "semantic_false_peak"
    elif iou >= args.positive_iou:
        return None, None
    elif gt_coverage >= args.coverage_threshold and pred_len >= gt_len * args.over_wide_ratio:
        candidate_type = "over_wide"
    elif pred_coverage >= args.coverage_threshold and pred_len <= gt_len * args.under_wide_ratio:
        candidate_type = "under_wide"
    elif iou >= args.boundary_min_iou:
        candidate_type = "boundary_distractor"
    else:
        return None, None

    details = {
        "type": candidate_type,
        "matched_gt": [float(matched_gt[0]), float(matched_gt[1])],
        "iou": float(iou),
        "overlap": float(overlap),
        "pred_len": float(pred_len),
        "gt_len": float(gt_len),
        "gt_coverage": float(gt_coverage),
        "pred_coverage": float(pred_coverage),
    }
    return candidate_type, details


def mine_item(item, args):
    scores = item.get("evidence_scores", [])
    gt_windows = item.get("gt_windows", [])
    by_type = {name: {} for name in TYPE_ORDER}

    for source in args.sources:
        decoder = DECODERS[source]
        for rank, raw_window in enumerate(decoder(item, args.topn_per_source), start=1):
            window = normalize_window(raw_window)
            if window is None:
                continue
            candidate_type, details = classify_candidate(window, gt_windows, args)
            if candidate_type is None:
                continue

            score = window_score(scores, window[0], window[1], args.score_mode)
            if score < args.min_score:
                continue

            key = (round(window[0], 3), round(window[1], 3))
            candidate = {
                "window": window,
                "score": score,
                "source": source,
                "rank": rank,
                "decoder_score": float(raw_window[2]) if len(raw_window) > 2 else 0.0,
                **details,
            }
            old = by_type[candidate_type].get(key)
            if old is None or candidate["score"] > old["score"]:
                by_type[candidate_type][key] = candidate

    typed = {}
    for name in TYPE_ORDER:
        limit = getattr(args, f"topk_{name}")
        typed[name] = sorted(by_type[name].values(), key=lambda item: item["score"], reverse=True)[:limit]
    return typed


def summarize(hard_negatives, args):
    type_counts = Counter()
    source_counts = defaultdict(Counter)
    type_scores = defaultdict(list)
    type_ious = defaultdict(list)
    samples_by_type = Counter()

    for typed in hard_negatives.values():
        for name, candidates in typed.items():
            if candidates:
                samples_by_type[name] += 1
            for candidate in candidates:
                type_counts[name] += 1
                source_counts[name][candidate["source"]] += 1
                type_scores[name].append(candidate["score"])
                type_ious[name].append(candidate["iou"])

    total = sum(type_counts.values())
    return {
        "num_samples": args.num_samples,
        "total_hard_negatives": total,
        "avg_hard_negatives_per_sample": total / max(args.num_samples, 1),
        "type_counts": dict(type_counts),
        "samples_by_type": dict(samples_by_type),
        "type_stats": {
            name: {
                "count": type_counts.get(name, 0),
                "samples": samples_by_type.get(name, 0),
                "avg_score": sum(type_scores[name]) / max(len(type_scores[name]), 1),
                "avg_iou": sum(type_ious[name]) / max(len(type_ious[name]), 1),
                "max_iou": max(type_ious[name]) if type_ious[name] else 0.0,
                "source_counts": dict(source_counts[name]),
            }
            for name in TYPE_ORDER
        },
        "settings": {
            "topn_per_source": args.topn_per_source,
            "score_mode": args.score_mode,
            "min_score": args.min_score,
            "semantic_max_iou": args.semantic_max_iou,
            "boundary_min_iou": args.boundary_min_iou,
            "positive_iou": args.positive_iou,
            "coverage_threshold": args.coverage_threshold,
            "over_wide_ratio": args.over_wide_ratio,
            "under_wide_ratio": args.under_wide_ratio,
            "topk_by_type": {name: getattr(args, f"topk_{name}") for name in TYPE_ORDER},
            "sources": args.sources,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence_path",
        default="results/evidence_baseline_v2_hn_l02/full_train_eval/predictions_evidence_samples.json",
    )
    parser.add_argument("--output_path", default="results/typed_hard_negatives_v3/train_typed_hard_negatives.json")
    parser.add_argument("--summary_path", default="results/typed_hard_negatives_v3/summary.json")
    parser.add_argument("--topn_per_source", type=int, default=12)
    parser.add_argument("--score_mode", choices=["mean", "max", "top25", "sum"], default="top25")
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--semantic_max_iou", type=float, default=0.1)
    parser.add_argument("--boundary_min_iou", type=float, default=0.1)
    parser.add_argument("--positive_iou", type=float, default=0.7)
    parser.add_argument("--coverage_threshold", type=float, default=0.8)
    parser.add_argument("--over_wide_ratio", type=float, default=1.4)
    parser.add_argument("--under_wide_ratio", type=float, default=0.7)
    parser.add_argument("--topk_semantic_false_peak", type=int, default=3)
    parser.add_argument("--topk_boundary_distractor", type=int, default=3)
    parser.add_argument("--topk_over_wide", type=int, default=2)
    parser.add_argument("--topk_under_wide", type=int, default=2)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    return parser.parse_args()


def main():
    args = parse_args()
    items = load_items(args.evidence_path)
    args.num_samples = len(items)

    unknown = [source for source in args.sources if source not in DECODERS]
    if unknown:
        raise ValueError(f"unknown decoder sources: {unknown}")

    hard_negatives = {}
    examples = []
    for item in items:
        typed = mine_item(item, args)
        hard_negatives[item["qid"]] = typed
        if len(examples) < 20 and any(typed.values()):
            examples.append(
                {
                    "qid": item["qid"],
                    "query": item.get("query", ""),
                    "gt_windows": item.get("gt_windows", []),
                    "hard_negatives": typed,
                }
            )

    payload = {
        "metadata": summarize(hard_negatives, args),
        "hard_negatives": hard_negatives,
        "examples": examples,
    }
    save_json(payload, args.output_path)
    save_json(payload["metadata"], args.summary_path)
    print(json.dumps(payload["metadata"], indent=2))
    print("saved typed hard negatives to", args.output_path)


if __name__ == "__main__":
    main()
