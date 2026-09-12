import argparse
import json
import math
import os
from collections import Counter, defaultdict

import torch

from metrics import iou_1d
from train_candidate_level_representation_fusion import build_fusion_rows
from train_learned_evidence_decoder import CandidateDataset, load_items, parse_source_quotas, score_rows
from train_two_mode_decoder_selector import apply_decoder, load_decoder_checkpoint
from train_default_anchored_intervention_gate import load_fusion_scorer


TYPE_ORDER = [
    "semantic_false_peak",
    "boundary_distractor",
    "over_wide",
    "under_wide",
]


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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

    pred_len = max(float(window[1]) - float(window[0]), 1e-6)
    gt_len = max(float(matched_gt[1]) - float(matched_gt[0]), 1e-6)
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

    return candidate_type, {
        "type": candidate_type,
        "matched_gt": [float(matched_gt[0]), float(matched_gt[1])],
        "iou": float(iou),
        "overlap": float(overlap),
        "pred_len": float(pred_len),
        "gt_len": float(gt_len),
        "gt_coverage": float(gt_coverage),
        "pred_coverage": float(pred_coverage),
    }


def score_merged_candidates(rows, scorer, device):
    dataset = CandidateDataset(rows, feature_mean=scorer["feature_mean"], feature_std=scorer["feature_std"])
    return score_rows(scorer["model"], dataset, rows, device)


def mine_from_scored_rows(items, scored_rows, args):
    item_by_qid = {item["qid"]: item for item in items}
    grouped = defaultdict(list)
    for row in scored_rows:
        grouped[row["qid"]].append(row)

    hard_negatives = {}
    examples = []
    for qid, rows in grouped.items():
        item = item_by_qid[qid]
        gt_windows = item.get("gt_windows", [])
        by_type = {name: {} for name in TYPE_ORDER}
        ranked = sorted(rows, key=lambda row: row.get("pred_quality", 0.0), reverse=True)[: args.topn_merged]
        for rank, row in enumerate(ranked, start=1):
            if float(row.get("pred_quality", 0.0)) < args.min_pred_quality:
                continue
            window = [float(row["window"][0]), float(row["window"][1])]
            if window[1] <= window[0]:
                continue
            candidate_type, details = classify_candidate(window, gt_windows, args)
            if candidate_type is None:
                continue
            key = (round(window[0], 3), round(window[1], 3), row.get("representation", ""), row.get("candidate_source", ""))
            candidate = {
                "window": window,
                "score": float(row.get("pred_quality", 0.0)),
                "source": f"{row.get('representation', '')}:{row.get('candidate_source', '')}",
                "rank": rank,
                "decoder_score": float(row.get("window", [0.0, 0.0, 0.0])[2]) if len(row.get("window", [])) > 2 else 0.0,
                "representation": row.get("representation", ""),
                "candidate_source": row.get("candidate_source", ""),
                "source_rank": int(row.get("source_rank", 0)),
                **details,
            }
            old = by_type[candidate_type].get(key)
            if old is None or candidate["score"] > old["score"]:
                by_type[candidate_type][key] = candidate

        typed = {}
        for name in TYPE_ORDER:
            limit = getattr(args, f"topk_{name}")
            typed[name] = sorted(by_type[name].values(), key=lambda cand: cand["score"], reverse=True)[:limit]
        hard_negatives[qid] = typed
        if len(examples) < 20 and any(typed.values()):
            examples.append(
                {
                    "qid": qid,
                    "query": item.get("query", ""),
                    "gt_windows": gt_windows,
                    "hard_negatives": typed,
                }
            )
    return hard_negatives, examples


def summarize(hard_negatives, args):
    type_counts = Counter()
    samples_by_type = Counter()
    source_counts = defaultdict(Counter)
    representation_counts = defaultdict(Counter)
    type_scores = defaultdict(list)
    type_ious = defaultdict(list)

    for typed in hard_negatives.values():
        for name, candidates in typed.items():
            if candidates:
                samples_by_type[name] += 1
            for candidate in candidates:
                type_counts[name] += 1
                source_counts[name][candidate["source"]] += 1
                representation_counts[name][candidate.get("representation", "")] += 1
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
                "representation_counts": dict(representation_counts[name]),
            }
            for name in TYPE_ORDER
        },
        "settings": {
            "topn_per_source": args.topn_per_source,
            "topn_merged": args.topn_merged,
            "min_pred_quality": args.min_pred_quality,
            "semantic_max_iou": args.semantic_max_iou,
            "boundary_min_iou": args.boundary_min_iou,
            "positive_iou": args.positive_iou,
            "coverage_threshold": args.coverage_threshold,
            "over_wide_ratio": args.over_wide_ratio,
            "under_wide_ratio": args.under_wide_ratio,
            "topk_by_type": {name: getattr(args, f"topk_{name}") for name in TYPE_ORDER},
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_evidence_path", required=True)
    parser.add_argument("--adapter_evidence_path", required=True)
    parser.add_argument("--ms_decoder_ckpt", required=True)
    parser.add_argument("--adapter_decoder_ckpt", required=True)
    parser.add_argument("--fusion_scorer_ckpt", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--topn_per_source", type=int, default=2)
    parser.add_argument("--source_quotas", default="")
    parser.add_argument("--feature_version", choices=["basic", "shape_v2"], default="shape_v2")
    parser.add_argument("--topn_merged", type=int, default=20)
    parser.add_argument("--min_pred_quality", type=float, default=0.20)
    parser.add_argument("--semantic_max_iou", type=float, default=0.1)
    parser.add_argument("--boundary_min_iou", type=float, default=0.1)
    parser.add_argument("--positive_iou", type=float, default=0.7)
    parser.add_argument("--coverage_threshold", type=float, default=0.8)
    parser.add_argument("--over_wide_ratio", type=float, default=1.4)
    parser.add_argument("--under_wide_ratio", type=float, default=0.7)
    parser.add_argument("--topk_semantic_false_peak", type=int, default=4)
    parser.add_argument("--topk_boundary_distractor", type=int, default=4)
    parser.add_argument("--topk_over_wide", type=int, default=2)
    parser.add_argument("--topk_under_wide", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_quotas = parse_source_quotas(args.source_quotas)
    ms_items = load_items(args.ms_evidence_path)
    adapter_items = load_items(args.adapter_evidence_path)
    args.num_samples = len(ms_items)
    ms_decoder = load_decoder_checkpoint(args.ms_decoder_ckpt, device)
    adapter_decoder = load_decoder_checkpoint(args.adapter_decoder_ckpt, device)
    fusion_scorer = load_fusion_scorer(args.fusion_scorer_ckpt, device)

    print("applying MS decoder...")
    ms_records, _ = apply_decoder(ms_items, ms_decoder, device, "ms_clap")
    print("applying adapter decoder...")
    adapter_records, _ = apply_decoder(adapter_items, adapter_decoder, device, "adapter")
    print("building merged candidates...")
    rows = build_fusion_rows(
        ms_items,
        adapter_items,
        ms_records,
        adapter_records,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    print("scoring merged candidates:", len(rows))
    scored_rows = score_merged_candidates(rows, fusion_scorer, device)
    hard_negatives, examples = mine_from_scored_rows(ms_items, scored_rows, args)
    metadata = summarize(hard_negatives, args)
    payload = {
        "metadata": metadata,
        "hard_negatives": hard_negatives,
        "examples": examples,
    }
    save_json(payload, args.output_path)
    save_json(metadata, args.summary_path)
    print(json.dumps(metadata, indent=2))
    print("saved temporal evidence hard negatives to", args.output_path)


if __name__ == "__main__":
    main()
