import argparse
import json
import os
import statistics
from types import SimpleNamespace

import torch

from evidence_decoder_experiment import save_json, save_jsonl
from evaluate_candidate_level_representation_fusion import records_to_rows
from train_candidate_level_representation_fusion import (
    build_fusion_rows,
    candidate_source_counts,
    mode_counts,
    oracle_from_rows,
    scored_rows_to_records,
    summarize_interventions,
)
from train_learned_evidence_decoder import load_items, parse_source_quotas
from train_semantic_temporal_candidate_scorer import (
    SemanticTemporalCandidateDataset,
    SemanticTemporalCandidateScorer,
    add_semantic_temporal_labels,
    role_counts,
    score_rows,
)
from train_two_mode_decoder_selector import apply_decoder, load_decoder_checkpoint


IOU_THRESHOLDS = [round(0.5 + 0.05 * idx, 2) for idx in range(10)]


def temporal_iou(left, right):
    inter = max(0.0, min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])))
    union = max(float(left[1]), float(right[1])) - min(float(left[0]), float(right[0]))
    return inter / max(union, 1e-12)


def interpolated_ap(precision, recall):
    mprecision = [0.0] + list(precision) + [0.0]
    mrecall = [0.0] + list(recall) + [1.0]
    for idx in range(len(mprecision) - 2, -1, -1):
        mprecision[idx] = max(mprecision[idx], mprecision[idx + 1])
    ap = 0.0
    for idx in range(1, len(mrecall)):
        if mrecall[idx] != mrecall[idx - 1]:
            ap += (mrecall[idx] - mrecall[idx - 1]) * mprecision[idx]
    return ap


def ap_for_query(gt_windows, pred_windows, thresholds=IOU_THRESHOLDS):
    if not gt_windows:
        return [0.0 for _ in thresholds]
    predictions = [
        {
            "window": [float(window[0]), float(window[1])],
            "score": float(window[2]) if len(window) > 2 else 1.0 / (rank + 1),
        }
        for rank, window in enumerate(pred_windows[:10])
    ]
    predictions.sort(key=lambda item: -item["score"])
    if not predictions:
        return [0.0 for _ in thresholds]

    output = []
    for threshold in thresholds:
        locked = [False] * len(gt_windows)
        tp = []
        fp = []
        for pred in predictions:
            ious = [temporal_iou(pred["window"], gt[:2]) for gt in gt_windows]
            order = sorted(range(len(ious)), key=lambda idx: ious[idx], reverse=True)
            matched = False
            for gt_idx in order:
                if ious[gt_idx] < threshold:
                    break
                if locked[gt_idx]:
                    continue
                locked[gt_idx] = True
                matched = True
                break
            tp.append(1.0 if matched else 0.0)
            fp.append(0.0 if matched else 1.0)
        tp_cumsum = []
        fp_cumsum = []
        running_tp = 0.0
        running_fp = 0.0
        for tp_value, fp_value in zip(tp, fp):
            running_tp += tp_value
            running_fp += fp_value
            tp_cumsum.append(running_tp)
            fp_cumsum.append(running_fp)
        recall = [value / max(float(len(gt_windows)), 1.0) for value in tp_cumsum]
        precision = [
            tp_value / max(tp_value + fp_value, 1e-12)
            for tp_value, fp_value in zip(tp_cumsum, fp_cumsum)
        ]
        output.append(interpolated_ap(precision, recall))
    return output


def compute_official_brief(items, rows):
    rows_by_qid = {row["qid"]: row for row in rows}
    ap_values = []
    r1_hits = {threshold: [] for threshold in IOU_THRESHOLDS}
    for item in items:
        qid = item["qid"]
        gt_windows = item.get("gt_windows", item.get("relevant_windows", []))
        pred_windows = rows_by_qid.get(qid, {}).get("windows", [])
        ap_values.append(ap_for_query(gt_windows, pred_windows))
        if pred_windows and gt_windows:
            top_window = pred_windows[0][:2]
            best_iou = max(temporal_iou(top_window, gt[:2]) for gt in gt_windows)
        else:
            best_iou = 0.0
        for threshold in IOU_THRESHOLDS:
            r1_hits[threshold].append(float(best_iou >= threshold))

    ap_by_threshold = {}
    for idx, threshold in enumerate(IOU_THRESHOLDS):
        ap_by_threshold[str(threshold)] = 100.0 * statistics.mean(row[idx] for row in ap_values)
    average = statistics.mean(ap_by_threshold.values())
    r1 = {str(threshold): 100.0 * statistics.mean(r1_hits[threshold]) for threshold in IOU_THRESHOLDS}
    return {
        "MR-full-mAP": float(f"{average:.2f}"),
        "MR-full-mAP@0.5": float(f"{ap_by_threshold['0.5']:.2f}"),
        "MR-full-mAP@0.75": float(f"{ap_by_threshold['0.75']:.2f}"),
        "MR-full-R1@0.5": float(f"{r1['0.5']:.2f}"),
        "MR-full-R1@0.7": float(f"{r1['0.7']:.2f}"),
    }


def load_scorer_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    feature_mean = ckpt["feature_mean"].cpu()
    feature_std = ckpt["feature_std"].cpu()
    model = SemanticTemporalCandidateScorer(
        input_dim=int(feature_mean.numel()),
        hidden_dim=int(args.get("hidden_dim", 128)),
        dropout=float(args.get("dropout", 0.12)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "path": path,
        "args": args,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "model": model,
        "seed": ckpt.get("seed", args.get("seed")),
        "epoch": ckpt.get("epoch"),
        "metrics": ckpt.get("metrics", {}),
    }


def namespace_from_checkpoint(args):
    defaults = {
        "quality_alpha": 1.0,
        "semantic_alpha": 0.8,
        "temporal_alpha": 0.8,
        "strict_gain_alpha": 0.5,
        "utility_alpha": 0.35,
        "anchor_risk_alpha": 0.8,
        "semantic_risk_alpha": 0.5,
        "temporal_risk_alpha": 0.6,
        "anchor_guard_alpha": 0.35,
        "score_mode": "quality_guard",
        "quality_target": "fusion",
        "topn_per_source": 2,
        "source_quotas": "",
        "feature_version": "shape_v2",
    }
    merged = dict(defaults)
    merged.update(args)
    return SimpleNamespace(**merged)


def write_predictions(path, rows):
    save_jsonl(
        [
            {
                "qid": row["qid"],
                "query": row.get("query", ""),
                "vid": row.get("vid", ""),
                "pred_relevant_windows": row.get("windows", []),
            }
            for row in rows
        ],
        path,
    )


def collect_row(summary, system_name):
    metrics = summary["metrics"][system_name]
    official = summary["official_metrics"][system_name]
    interventions = summary["interventions"][system_name]
    return {
        "seed": summary.get("seed"),
        "best_epoch": summary.get("checkpoint_epoch"),
        "R1@0.5": metrics["R1@0.5"],
        "R1@0.7": metrics["R1@0.7"],
        "R3@0.7": metrics["R3@0.7"],
        "R5@0.7": metrics["R5@0.7"],
        "top1_iou": metrics["top1_iou"],
        "top5_iou": metrics["top5_iou"],
        "semantic_miss": metrics["categories"].get("semantic_miss", 0),
        "good": metrics["categories"].get("good", 0),
        "mAP": official["MR-full-mAP"],
        "mAP@0.5": official["MR-full-mAP@0.5"],
        "mAP@0.75": official["MR-full-mAP@0.75"],
        "official_R1@0.5": official["MR-full-R1@0.5"],
        "official_R1@0.7": official["MR-full-R1@0.7"],
        "changed": interventions["changed"],
        "improved": interventions["improved"],
        "regressed": interventions["regressed"],
        "semantic_recovered": interventions["recovered_semantic_miss"],
        "good_regressed": interventions["good_regressed"],
        "precision": interventions["intervention_precision"],
    }


def aggregate(rows):
    keys = [key for key in rows[0] if key not in {"seed", "best_epoch"}]
    output = {"rows": rows, "mean": {}, "std": {}}
    for key in keys:
        values = [float(row[key]) for row in rows]
        output["mean"][key] = statistics.mean(values)
        output["std"][key] = statistics.pstdev(values)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_evidence_path", required=True)
    parser.add_argument("--adapter_evidence_path", required=True)
    parser.add_argument("--ms_decoder_ckpt", required=True)
    parser.add_argument("--adapter_decoder_ckpt", required=True)
    parser.add_argument("--scorer_ckpts", required=True, help="comma-separated checkpoint paths")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skip_case_dump", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer_paths = [path.strip() for path in args.scorer_ckpts.split(",") if path.strip()]
    if not scorer_paths:
        raise ValueError("no scorer checkpoints provided")

    items = load_items(args.ms_evidence_path)
    adapter_items = load_items(args.adapter_evidence_path)
    ms_decoder = load_decoder_checkpoint(args.ms_decoder_ckpt, device)
    adapter_decoder = load_decoder_checkpoint(args.adapter_decoder_ckpt, device)

    print("applying MS decoder to test...")
    ms_records, ms_metrics = apply_decoder(items, ms_decoder, device, "ms_clap")
    print("applying adapter decoder to test...")
    adapter_records, adapter_metrics = apply_decoder(adapter_items, adapter_decoder, device, "adapter")
    ms_rows = records_to_rows(items, ms_records)
    adapter_rows = records_to_rows(items, adapter_records)
    ms_official = compute_official_brief(items, ms_rows)
    adapter_official = compute_official_brief(items, adapter_rows)

    first_ckpt = load_scorer_checkpoint(scorer_paths[0], device)
    first_args = namespace_from_checkpoint(first_ckpt["args"])
    source_quotas = parse_source_quotas(first_args.source_quotas)
    print("building merged test candidates once...")
    rows = build_fusion_rows(
        items,
        adapter_items,
        ms_records,
        adapter_records,
        topn_per_source=int(first_args.topn_per_source),
        feature_version=first_args.feature_version,
        source_quotas=source_quotas,
    )
    rows = add_semantic_temporal_labels(rows, ms_records, quality_target=first_args.quality_target)
    print("test candidates:", len(rows))
    merged_oracle_metrics, merged_oracle_rows = oracle_from_rows(items, rows, ms_records, adapter_records)
    merged_oracle_official = compute_official_brief(items, merged_oracle_rows)

    seed_summaries = []
    aggregate_rows = []
    for ckpt_path in scorer_paths:
        scorer = first_ckpt if ckpt_path == scorer_paths[0] else load_scorer_checkpoint(ckpt_path, device)
        scorer_args = namespace_from_checkpoint(scorer["args"])
        dataset = SemanticTemporalCandidateDataset(
            rows,
            feature_mean=scorer["feature_mean"],
            feature_std=scorer["feature_std"],
        )
        scored = score_rows(scorer["model"], dataset, rows, device, scorer_args)
        scorer_metrics, scorer_rows = scored_rows_to_records(items, scored, ms_records, adapter_records)
        scorer_official = compute_official_brief(items, scorer_rows)
        seed = scorer.get("seed")
        seed_dir = os.path.join(args.output_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        summary = {
            "split": "test",
            "seed": seed,
            "num_samples": len(items),
            "test_candidates": len(rows),
            "scorer_checkpoint": ckpt_path,
            "checkpoint_epoch": scorer.get("epoch"),
            "checkpoint_val_metrics": scorer.get("metrics", {}),
            "metrics": {
                "ms_clap_shape_v2_top2": dict(ms_metrics, official_brief=ms_official),
                "adapter_shape_v2_top2": dict(adapter_metrics, official_brief=adapter_official),
                "semantic_temporal_candidate_scorer_v2": dict(scorer_metrics, official_brief=scorer_official),
                "merged_candidate_oracle": dict(merged_oracle_metrics, official_brief=merged_oracle_official),
            },
            "official_metrics": {
                "ms_clap_shape_v2_top2": ms_official,
                "adapter_shape_v2_top2": adapter_official,
                "semantic_temporal_candidate_scorer_v2": scorer_official,
                "merged_candidate_oracle": merged_oracle_official,
            },
            "interventions": {
                "semantic_temporal_candidate_scorer_v2": summarize_interventions(scorer_rows),
                "merged_candidate_oracle": summarize_interventions(merged_oracle_rows),
            },
            "role_counts": role_counts(scorer_rows),
            "mode_counts": {
                "semantic_temporal_candidate_scorer_v2": mode_counts(scorer_rows),
                "merged_candidate_oracle": mode_counts(merged_oracle_rows),
            },
            "candidate_source_counts": {
                "semantic_temporal_candidate_scorer_v2": candidate_source_counts(scorer_rows),
                "merged_candidate_oracle": candidate_source_counts(merged_oracle_rows),
            },
        }
        save_json(summary, os.path.join(seed_dir, "summary.json"))
        write_predictions(os.path.join(seed_dir, "predictions.jsonl"), scorer_rows)
        if not args.skip_case_dump:
            save_json(scorer_rows, os.path.join(seed_dir, "case_rows.json"))
        seed_summaries.append(summary)
        aggregate_rows.append(collect_row(summary, "semantic_temporal_candidate_scorer_v2"))
        print(json.dumps({"seed": seed, "metrics": scorer_metrics, "official": scorer_official}, indent=2))

    aggregate_summary = aggregate(aggregate_rows)
    aggregate_summary["split"] = "test"
    aggregate_summary["num_samples"] = len(items)
    aggregate_summary["test_candidates"] = len(rows)
    aggregate_summary["scorer_checkpoints"] = scorer_paths
    aggregate_summary["reference_official_metrics"] = {
        "ms_clap_shape_v2_top2": ms_official,
        "adapter_shape_v2_top2": adapter_official,
        "merged_candidate_oracle": merged_oracle_official,
    }
    save_json(aggregate_summary, os.path.join(args.output_dir, "summary_5seed.json"))
    print(json.dumps(aggregate_summary, indent=2))
    print("saved", os.path.join(args.output_dir, "summary_5seed.json"))


if __name__ == "__main__":
    main()
