import argparse
import json
import os

import torch

from evidence_decoder_experiment import save_json, save_jsonl
from train_candidate_level_representation_fusion import (
    apply_candidate_gate,
    build_fusion_rows,
    candidate_source_counts,
    mode_counts,
    oracle_from_rows,
    scored_rows_to_records,
    summarize_interventions,
)
from train_learned_evidence_decoder import CandidateDataset, CandidateScorer, load_items, parse_source_quotas, score_rows
from train_two_mode_decoder_selector import apply_decoder, load_decoder_checkpoint


def load_fusion_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    feature_mean = ckpt["feature_mean"].cpu()
    feature_std = ckpt["feature_std"].cpu()
    model = CandidateScorer(
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
        "epoch": ckpt.get("epoch"),
        "metrics": ckpt.get("metrics", {}),
    }


def load_gate_params(summary_path):
    if not summary_path or not os.path.exists(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return {
        "balanced_gate_params": summary.get("balanced_gate_params"),
        "precision_gate_params": summary.get("precision_gate_params"),
    }


def ground_truth_from_items(items):
    return [
        {
            "qid": item["qid"],
            "query": item.get("query", ""),
            "vid": item.get("vid", ""),
            "duration": item.get("duration", 0),
            "relevant_windows": item.get("gt_windows", item.get("relevant_windows", [])),
        }
        for item in items
    ]


def rows_to_submission(rows):
    return [
        {
            "qid": row["qid"],
            "query": row.get("query", ""),
            "vid": row.get("vid", ""),
            "pred_relevant_windows": row.get("windows", []),
        }
        for row in rows
    ]


def records_to_rows(items, records):
    rows = []
    for item in items:
        qid = item["qid"]
        record = records.get(qid, {})
        rows.append(
            {
                "qid": qid,
                "query": item.get("query", ""),
                "vid": item.get("vid", ""),
                "windows": record.get("windows", []),
            }
        )
    return rows


def compute_official_brief(items, rows):
    from standalone_eval.eval import compute_mr_ap, compute_mr_r1

    submission = rows_to_submission(rows)
    ground_truth = ground_truth_from_items(items)
    ap = compute_mr_ap(submission, ground_truth, num_workers=1, chunksize=50)
    r1 = compute_mr_r1(submission, ground_truth)
    return {
        "MR-full-mAP": ap["average"],
        "MR-full-mAP@0.5": ap["0.5"],
        "MR-full-mAP@0.75": ap["0.75"],
        "MR-full-R1@0.5": r1["0.5"],
        "MR-full-R1@0.7": r1["0.7"],
    }


def attach_official(metrics, official):
    if metrics is not None:
        metrics["official_brief"] = official
    return metrics


def write_predictions(path, rows):
    save_jsonl(
        rows_to_submission(rows),
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_evidence_path", required=True)
    parser.add_argument("--adapter_evidence_path", required=True)
    parser.add_argument("--ms_decoder_ckpt", required=True)
    parser.add_argument("--adapter_decoder_ckpt", required=True)
    parser.add_argument("--fusion_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fusion_summary_path", default="")
    parser.add_argument("--skip_case_dump", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = load_items(args.ms_evidence_path)
    adapter_items = load_items(args.adapter_evidence_path)
    ms_decoder = load_decoder_checkpoint(args.ms_decoder_ckpt, device)
    adapter_decoder = load_decoder_checkpoint(args.adapter_decoder_ckpt, device)
    fusion = load_fusion_checkpoint(args.fusion_ckpt, device)

    print("applying MS decoder to test...")
    ms_records, ms_metrics = apply_decoder(items, ms_decoder, device, "ms_clap")
    print("applying adapter decoder to test...")
    adapter_records, adapter_metrics = apply_decoder(adapter_items, adapter_decoder, device, "adapter")
    ms_rows = records_to_rows(items, ms_records)
    adapter_rows = records_to_rows(items, adapter_records)

    fusion_args = fusion["args"]
    source_quotas = parse_source_quotas(fusion_args.get("source_quotas", ""))
    print("building merged test candidates...")
    rows = build_fusion_rows(
        items,
        adapter_items,
        ms_records,
        adapter_records,
        topn_per_source=int(fusion_args.get("topn_per_source", 2)),
        feature_version=fusion_args.get("feature_version", "shape_v2"),
        source_quotas=source_quotas,
    )
    print("test candidates:", len(rows))
    dataset = CandidateDataset(rows, feature_mean=fusion["feature_mean"], feature_std=fusion["feature_std"])
    scored = score_rows(fusion["model"], dataset, rows, device)
    fusion_metrics, fusion_rows = scored_rows_to_records(items, scored, ms_records, adapter_records)

    gate_params = load_gate_params(args.fusion_summary_path)
    balanced_metrics = balanced_rows = None
    precision_metrics = precision_rows = None
    if gate_params.get("balanced_gate_params"):
        balanced_metrics, balanced_rows = apply_candidate_gate(
            items,
            fusion_rows,
            ms_records,
            gate_params["balanced_gate_params"],
        )
    if gate_params.get("precision_gate_params"):
        precision_metrics, precision_rows = apply_candidate_gate(
            items,
            fusion_rows,
            ms_records,
            gate_params["precision_gate_params"],
        )

    ms_oracle_metrics, ms_oracle_rows = oracle_from_rows(items, scored, ms_records, adapter_records, representation="ms_clap")
    adapter_oracle_metrics, adapter_oracle_rows = oracle_from_rows(
        items,
        scored,
        ms_records,
        adapter_records,
        representation="adapter",
    )
    merged_oracle_metrics, merged_oracle_rows = oracle_from_rows(items, scored, ms_records, adapter_records)

    official_metrics = {
        "ms_clap_shape_v2_top2": compute_official_brief(items, ms_rows),
        "adapter_shape_v2_top2": compute_official_brief(items, adapter_rows),
        "candidate_level_fusion": compute_official_brief(items, fusion_rows),
        "candidate_level_balanced_gate": compute_official_brief(items, balanced_rows) if balanced_rows else None,
        "candidate_level_precision_gate": compute_official_brief(items, precision_rows) if precision_rows else None,
        "ms_candidate_oracle": compute_official_brief(items, ms_oracle_rows),
        "adapter_candidate_oracle": compute_official_brief(items, adapter_oracle_rows),
        "merged_candidate_oracle": compute_official_brief(items, merged_oracle_rows),
    }

    metrics = {
        "ms_clap_shape_v2_top2": attach_official(ms_metrics, official_metrics["ms_clap_shape_v2_top2"]),
        "adapter_shape_v2_top2": attach_official(adapter_metrics, official_metrics["adapter_shape_v2_top2"]),
        "candidate_level_fusion": attach_official(fusion_metrics, official_metrics["candidate_level_fusion"]),
        "candidate_level_balanced_gate": attach_official(
            balanced_metrics,
            official_metrics["candidate_level_balanced_gate"],
        ),
        "candidate_level_precision_gate": attach_official(
            precision_metrics,
            official_metrics["candidate_level_precision_gate"],
        ),
        "ms_candidate_oracle": attach_official(ms_oracle_metrics, official_metrics["ms_candidate_oracle"]),
        "adapter_candidate_oracle": attach_official(
            adapter_oracle_metrics,
            official_metrics["adapter_candidate_oracle"],
        ),
        "merged_candidate_oracle": attach_official(
            merged_oracle_metrics,
            official_metrics["merged_candidate_oracle"],
        ),
    }
    interventions = {
        "candidate_level_fusion": summarize_interventions(fusion_rows),
        "candidate_level_balanced_gate": summarize_interventions(balanced_rows) if balanced_rows else None,
        "candidate_level_precision_gate": summarize_interventions(precision_rows) if precision_rows else None,
        "merged_candidate_oracle": summarize_interventions(merged_oracle_rows),
    }
    summary = {
        "split": "test",
        "num_samples": len(items),
        "test_candidates": len(rows),
        "fusion_checkpoint": args.fusion_ckpt,
        "fusion_checkpoint_epoch": fusion.get("epoch"),
        "fusion_checkpoint_val_metrics": fusion.get("metrics"),
        "metrics": metrics,
        "official_metrics": official_metrics,
        "interventions": interventions,
        "mode_counts": {
            "candidate_level_fusion": mode_counts(fusion_rows),
            "candidate_level_balanced_gate": mode_counts(balanced_rows) if balanced_rows else None,
            "candidate_level_precision_gate": mode_counts(precision_rows) if precision_rows else None,
        },
        "candidate_source_counts": {
            "candidate_level_fusion": candidate_source_counts(fusion_rows),
            "candidate_level_balanced_gate": candidate_source_counts(balanced_rows) if balanced_rows else None,
            "candidate_level_precision_gate": candidate_source_counts(precision_rows) if precision_rows else None,
        },
        "gate_params": gate_params,
    }

    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(official_metrics, os.path.join(args.output_dir, "official_metrics.json"))
    if not args.skip_case_dump:
        save_json(fusion_rows, os.path.join(args.output_dir, "fusion_case_rows.json"))
        save_json(merged_oracle_rows, os.path.join(args.output_dir, "merged_oracle_case_rows.json"))
        if balanced_rows:
            save_json(balanced_rows, os.path.join(args.output_dir, "balanced_gate_case_rows.json"))
        if precision_rows:
            save_json(precision_rows, os.path.join(args.output_dir, "precision_gate_case_rows.json"))
    write_predictions(os.path.join(args.output_dir, "ms_clap_shape_v2_top2_predictions.jsonl"), ms_rows)
    write_predictions(os.path.join(args.output_dir, "adapter_shape_v2_top2_predictions.jsonl"), adapter_rows)
    write_predictions(os.path.join(args.output_dir, "fusion_predictions.jsonl"), fusion_rows)
    if balanced_rows:
        write_predictions(os.path.join(args.output_dir, "balanced_gate_predictions.jsonl"), balanced_rows)
    if precision_rows:
        write_predictions(os.path.join(args.output_dir, "precision_gate_predictions.jsonl"), precision_rows)
    print(json.dumps({"fusion_metrics": fusion_metrics, "adapter_metrics": adapter_metrics, "ms_metrics": ms_metrics}, indent=2))
    print("saved summary to", os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
