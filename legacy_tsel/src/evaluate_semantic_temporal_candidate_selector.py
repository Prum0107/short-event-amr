import argparse
import json
import os

import torch

from evidence_decoder_experiment import save_json, save_jsonl
from evaluate_candidate_level_representation_fusion import compute_official_brief, records_to_rows
from train_candidate_level_representation_fusion import build_fusion_rows, summarize_interventions
from train_default_anchored_intervention_gate import apply_fusion_scorer, load_fusion_scorer
from train_learned_evidence_decoder import load_items, parse_source_quotas
from train_semantic_temporal_candidate_selector import (
    SelectorDataset,
    SemanticTemporalSelector,
    apply_selector,
    build_selector_rows,
    predict,
)
from train_two_mode_decoder_selector import apply_decoder, load_decoder_checkpoint


def load_selector(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    feature_mean = ckpt["feature_mean"].cpu()
    feature_std = ckpt["feature_std"].cpu()
    model = SemanticTemporalSelector(
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
        "seed": ckpt.get("seed"),
        "balanced_params": ckpt.get("balanced_params"),
        "precision_params": ckpt.get("precision_params"),
        "metrics": ckpt.get("metrics", {}),
    }


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_evidence_path", required=True)
    parser.add_argument("--adapter_evidence_path", required=True)
    parser.add_argument("--ms_decoder_ckpt", required=True)
    parser.add_argument("--adapter_decoder_ckpt", required=True)
    parser.add_argument("--fusion_scorer_ckpt", required=True)
    parser.add_argument("--selector_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = load_items(args.ms_evidence_path)
    adapter_items = load_items(args.adapter_evidence_path)
    ms_decoder = load_decoder_checkpoint(args.ms_decoder_ckpt, device)
    adapter_decoder = load_decoder_checkpoint(args.adapter_decoder_ckpt, device)
    fusion_scorer = load_fusion_scorer(args.fusion_scorer_ckpt, device)
    selector = load_selector(args.selector_ckpt, device)

    print("applying MS decoder...")
    ms_records, ms_metrics = apply_decoder(items, ms_decoder, device, "ms_clap")
    print("applying adapter decoder...")
    adapter_records, adapter_metrics = apply_decoder(adapter_items, adapter_decoder, device, "adapter")

    fusion_args = fusion_scorer["args"]
    source_quotas = parse_source_quotas(fusion_args.get("source_quotas", ""))
    print("building merged candidates...")
    candidate_rows = build_fusion_rows(
        items,
        adapter_items,
        ms_records,
        adapter_records,
        topn_per_source=int(fusion_args.get("topn_per_source", 2)),
        feature_version=fusion_args.get("feature_version", "shape_v2"),
        source_quotas=source_quotas,
    )
    print("scoring with frozen fusion scorer...")
    fusion_metrics, fusion_rows = apply_fusion_scorer(
        items,
        candidate_rows,
        ms_records,
        adapter_records,
        fusion_scorer,
        device,
    )
    candidate_records = {row["qid"]: row for row in fusion_rows}
    selector_rows = build_selector_rows(items, candidate_records, ms_records)
    dataset = SelectorDataset(
        selector_rows,
        feature_mean=selector["feature_mean"],
        feature_std=selector["feature_std"],
    )
    predictions = predict(selector["model"], dataset, selector_rows, device)

    balanced_metrics, balanced_rows = apply_selector(
        items,
        predictions,
        candidate_records,
        ms_records,
        selector["balanced_params"],
    )
    precision_metrics, precision_rows = apply_selector(
        items,
        predictions,
        candidate_records,
        ms_records,
        selector["precision_params"],
    )

    ms_rows = records_to_rows(items, ms_records)
    adapter_rows = records_to_rows(items, adapter_records)
    official_metrics = {
        "ms_clap_shape_v2_top2": compute_official_brief(items, ms_rows),
        "adapter_shape_v2_top2": compute_official_brief(items, adapter_rows),
        "frozen_candidate_fusion": compute_official_brief(items, fusion_rows),
        "semantic_temporal_balanced_selector": compute_official_brief(items, balanced_rows),
        "semantic_temporal_precision_selector": compute_official_brief(items, precision_rows),
    }
    for name, metrics in [
        ("ms_clap_shape_v2_top2", ms_metrics),
        ("adapter_shape_v2_top2", adapter_metrics),
        ("frozen_candidate_fusion", fusion_metrics),
        ("semantic_temporal_balanced_selector", balanced_metrics),
        ("semantic_temporal_precision_selector", precision_metrics),
    ]:
        metrics["official_brief"] = official_metrics[name]

    summary = {
        "split": "test",
        "num_samples": len(items),
        "selector_checkpoint": args.selector_ckpt,
        "selector_seed": selector.get("seed"),
        "selector_val_metrics": selector.get("metrics"),
        "candidate_count": len(candidate_rows),
        "metrics": {
            "ms_clap_shape_v2_top2": ms_metrics,
            "adapter_shape_v2_top2": adapter_metrics,
            "frozen_candidate_fusion": fusion_metrics,
            "semantic_temporal_balanced_selector": balanced_metrics,
            "semantic_temporal_precision_selector": precision_metrics,
        },
        "official_metrics": official_metrics,
        "interventions": {
            "frozen_candidate_fusion": summarize_interventions(fusion_rows),
            "semantic_temporal_balanced_selector": summarize_interventions(balanced_rows),
            "semantic_temporal_precision_selector": summarize_interventions(precision_rows),
        },
        "balanced_params": selector["balanced_params"],
        "precision_params": selector["precision_params"],
    }
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(balanced_rows, os.path.join(args.output_dir, "balanced_case_rows.json"))
    save_json(precision_rows, os.path.join(args.output_dir, "precision_case_rows.json"))
    write_predictions(os.path.join(args.output_dir, "balanced_predictions.jsonl"), balanced_rows)
    write_predictions(os.path.join(args.output_dir, "precision_predictions.jsonl"), precision_rows)
    print(json.dumps(summary["metrics"], indent=2))
    print("saved", os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
