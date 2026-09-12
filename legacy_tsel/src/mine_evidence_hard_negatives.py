import argparse
import json
import math
import os
from collections import Counter

from evidence_decoder_experiment import DECODERS, score_values
from metrics import best_iou_for_window


DEFAULT_SOURCES = [
    "current_start_end",
    "peak_drop",
    "dense_contrast",
    "threshold_mean",
    "threshold_p60",
    "threshold_p70",
    "multiscale_10_20_40_80_150",
]


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def window_score(scores, st, ed, mode):
    left = max(0, int(math.floor(float(st))))
    right = min(len(scores), int(math.ceil(float(ed))))
    if right <= left:
        return 0.0
    return float(score_values(scores[left:right], mode))


def normalize_window(window):
    if not window or len(window) < 2:
        return None
    st = float(window[0])
    ed = float(window[1])
    if ed <= st:
        return None
    return [st, ed]


def mine_item(item, sources, topn_per_source, max_iou, min_score, score_mode):
    scores = item.get("evidence_scores", [])
    gt_windows = item.get("gt_windows", [])
    by_key = {}

    for source in sources:
        decoder = DECODERS[source]
        for rank, raw_window in enumerate(decoder(item, topn_per_source), start=1):
            window = normalize_window(raw_window)
            if window is None:
                continue
            iou = best_iou_for_window(window, gt_windows)
            score = window_score(scores, window[0], window[1], score_mode)
            if iou >= max_iou or score < min_score:
                continue

            key = (round(window[0], 3), round(window[1], 3))
            old = by_key.get(key)
            candidate = {
                "window": window,
                "score": score,
                "iou": float(iou),
                "source": source,
                "rank": rank,
                "decoder_score": float(raw_window[2]) if len(raw_window) > 2 else 0.0,
            }
            if old is None or candidate["score"] > old["score"]:
                by_key[key] = candidate

    return sorted(by_key.values(), key=lambda item: item["score"], reverse=True)


def summarize(hard_negatives, args):
    source_counts = Counter()
    total = 0
    scores = []
    ious = []
    for candidates in hard_negatives.values():
        total += len(candidates)
        for candidate in candidates:
            source_counts[candidate["source"]] += 1
            scores.append(candidate["score"])
            ious.append(candidate["iou"])

    num_samples = args.num_samples
    samples_with_hn = sum(1 for candidates in hard_negatives.values() if candidates)
    return {
        "num_samples": num_samples,
        "samples_with_hard_negatives": samples_with_hn,
        "coverage": samples_with_hn / max(num_samples, 1),
        "total_hard_negatives": total,
        "avg_hard_negatives_per_sample": total / max(num_samples, 1),
        "avg_score": sum(scores) / max(len(scores), 1),
        "avg_iou": sum(ious) / max(len(ious), 1),
        "max_iou": max(ious) if ious else 0.0,
        "source_counts": dict(source_counts),
        "settings": {
            "topk": args.topk,
            "topn_per_source": args.topn_per_source,
            "max_iou": args.max_iou,
            "min_score": args.min_score,
            "score_mode": args.score_mode,
            "sources": args.sources,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence_path",
        default="results/evidence_baseline_release_v1/full_train_eval/predictions_evidence_samples.json",
    )
    parser.add_argument("--output_path", default="results/hard_negatives_v1/train_hard_negatives.json")
    parser.add_argument("--summary_path", default="results/hard_negatives_v1/summary.json")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--topn_per_source", type=int, default=10)
    parser.add_argument("--max_iou", type=float, default=0.3)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--score_mode", choices=["mean", "max", "top25", "sum"], default="top25")
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
        candidates = mine_item(
            item=item,
            sources=args.sources,
            topn_per_source=args.topn_per_source,
            max_iou=args.max_iou,
            min_score=args.min_score,
            score_mode=args.score_mode,
        )[: args.topk]
        hard_negatives[item["qid"]] = candidates
        if candidates and len(examples) < 20:
            examples.append(
                {
                    "qid": item["qid"],
                    "query": item.get("query", ""),
                    "gt_windows": item.get("gt_windows", []),
                    "hard_negatives": candidates,
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
    print("saved hard negatives to", args.output_path)


if __name__ == "__main__":
    main()
