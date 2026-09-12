import argparse
import json
import os
import random
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evidence_decoder_experiment import save_json, save_jsonl
from train_candidate_level_representation_fusion import (
    build_fusion_rows,
    candidate_source_counts,
    mode_counts,
    oracle_from_rows,
    scored_rows_to_records,
    summarize_interventions,
)
from train_learned_evidence_decoder import load_items, model_selection_key, parse_source_quotas
from train_semantic_temporal_candidate_scorer import (
    SemanticTemporalCandidateDataset,
    SemanticTemporalCandidateScorer,
    add_semantic_temporal_labels,
    label_counts,
    pos_weight,
    role_counts,
    score_rows,
    train_epoch,
)
from train_two_mode_decoder_selector import apply_decoder, load_decoder_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ms_evidence_path", required=True)
    parser.add_argument("--val_ms_evidence_path", required=True)
    parser.add_argument("--train_adapter_evidence_path", required=True)
    parser.add_argument("--val_adapter_evidence_path", required=True)
    parser.add_argument("--ms_decoder_ckpt", required=True)
    parser.add_argument("--adapter_decoder_ckpt", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lambda_pairwise", type=float, default=0.2)
    parser.add_argument("--quality_loss_weight", type=float, default=0.8)
    parser.add_argument("--anchor_reliable_loss_weight", type=float, default=1.0)
    parser.add_argument("--semantic_loss_weight", type=float, default=1.0)
    parser.add_argument("--temporal_loss_weight", type=float, default=1.0)
    parser.add_argument("--anchor_risk_loss_weight", type=float, default=1.4)
    parser.add_argument("--semantic_risk_loss_weight", type=float, default=0.9)
    parser.add_argument("--temporal_risk_loss_weight", type=float, default=1.0)
    parser.add_argument("--strict_gain_loss_weight", type=float, default=0.8)
    parser.add_argument("--utility_loss_weight", type=float, default=0.5)
    parser.add_argument("--quality_alpha", type=float, default=1.0)
    parser.add_argument("--semantic_alpha", type=float, default=0.8)
    parser.add_argument("--temporal_alpha", type=float, default=0.8)
    parser.add_argument("--strict_gain_alpha", type=float, default=0.5)
    parser.add_argument("--utility_alpha", type=float, default=0.35)
    parser.add_argument("--anchor_risk_alpha", type=float, default=0.8)
    parser.add_argument("--semantic_risk_alpha", type=float, default=0.5)
    parser.add_argument("--temporal_risk_alpha", type=float, default=0.6)
    parser.add_argument("--anchor_guard_alpha", type=float, default=0.35)
    parser.add_argument("--score_mode", choices=["hybrid", "quality", "quality_guard"], default="quality_guard")
    parser.add_argument("--quality_target", choices=["evidence", "fusion"], default="fusion")
    parser.add_argument("--topn_per_source", type=int, default=2)
    parser.add_argument("--source_quotas", default="")
    parser.add_argument("--feature_version", choices=["basic", "shape_v2"], default="shape_v2")
    parser.add_argument("--write_cases", action="store_true")
    return parser.parse_args()


def make_losses(train_dataset, device):
    return {
        "anchor_reliable": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.anchor_reliable).to(device)),
        "semantic": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.semantic_gain).to(device)),
        "temporal": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.temporal_gain).to(device)),
        "anchor_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.anchor_risk).to(device)),
        "semantic_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.semantic_risk).to(device)),
        "temporal_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.temporal_risk).to(device)),
        "strict_gain": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.strict_gain).to(device)),
    }


def run_seed(args, seed, cached, device):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_dataset = cached["train_dataset"]
    val_dataset = cached["val_dataset"]
    train_rows = cached["train_rows"]
    val_rows = cached["val_rows"]
    val_ms_items = cached["val_ms_items"]
    val_ms_records = cached["val_ms_records"]
    val_adapter_records = cached["val_adapter_records"]
    val_ms_metrics = cached["val_ms_metrics"]
    val_adapter_metrics = cached["val_adapter_metrics"]

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model = SemanticTemporalCandidateScorer(
        input_dim=int(train_dataset.features.shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    losses = make_losses(train_dataset, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = f"{args.output_prefix}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)
    best_metrics = None
    best_epoch = 0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, losses, device, args)
        scored_val = score_rows(model, val_dataset, val_rows, device, args)
        val_metrics, _ = scored_rows_to_records(val_ms_items, scored_val, val_ms_records, val_adapter_records)
        print(
            f"[seed {seed} epoch {epoch + 1}] loss={train_metrics['loss']:.4f} "
            f"R1@0.7={100 * val_metrics['R1@0.7']:.2f} "
            f"R1@0.5={100 * val_metrics['R1@0.5']:.2f}",
            flush=True,
        )
        if best_metrics is None or model_selection_key(val_metrics) > model_selection_key(best_metrics):
            best_metrics = val_metrics
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": train_dataset.feature_mean,
                    "feature_std": train_dataset.feature_std,
                    "args": vars(args),
                    "seed": seed,
                    "epoch": best_epoch,
                    "metrics": best_metrics,
                },
                os.path.join(output_dir, "best.pt"),
            )

    ckpt = torch.load(os.path.join(output_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    scored_val = score_rows(model, val_dataset, val_rows, device, args)
    scorer_metrics, scorer_rows = scored_rows_to_records(val_ms_items, scored_val, val_ms_records, val_adapter_records)
    merged_oracle_metrics, merged_oracle_rows = oracle_from_rows(val_ms_items, val_rows, val_ms_records, val_adapter_records)
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "train_candidates": len(train_rows),
        "val_candidates": len(val_rows),
        "feature_dim": int(train_dataset.features.shape[1]),
        "label_counts": {"train": label_counts(train_dataset), "val": label_counts(val_dataset)},
        "metrics": {
            "ms_clap_shape_v2_top2": val_ms_metrics,
            "adapter_shape_v2_top2": val_adapter_metrics,
            "semantic_temporal_candidate_scorer_v2": scorer_metrics,
            "merged_candidate_oracle": merged_oracle_metrics,
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
        "args": vars(args),
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in scorer_rows],
        os.path.join(output_dir, "predictions.jsonl"),
    )
    if args.write_cases:
        save_json(scorer_rows, os.path.join(output_dir, "case_rows.json"))
        save_json(merged_oracle_rows, os.path.join(output_dir, "merged_oracle_case_rows.json"))
    print(json.dumps({"seed": seed, "best_epoch": best_epoch, "metrics": scorer_metrics}, indent=2), flush=True)
    return summary


def aggregate(summaries, output_prefix):
    rows = []
    for summary in summaries:
        seed = summary["seed"]
        m = summary["metrics"]["semantic_temporal_candidate_scorer_v2"]
        it = summary["interventions"]["semantic_temporal_candidate_scorer_v2"]
        rows.append(
            {
                "seed": seed,
                "best_epoch": summary["best_epoch"],
                "R1@0.5": m["R1@0.5"],
                "R1@0.7": m["R1@0.7"],
                "R3@0.7": m["R3@0.7"],
                "R5@0.7": m["R5@0.7"],
                "top1_iou": m["top1_iou"],
                "top5_iou": m["top5_iou"],
                "semantic_miss": m["categories"].get("semantic_miss", 0),
                "good": m["categories"].get("good", 0),
                "changed": it["changed"],
                "improved": it["improved"],
                "regressed": it["regressed"],
                "semantic_recovered": it["recovered_semantic_miss"],
                "good_regressed": it["good_regressed"],
                "precision": it["intervention_precision"],
            }
        )
    keys = [key for key in rows[0] if key not in {"seed", "best_epoch"}]
    result = {"rows": rows, "mean": {}, "std": {}}
    for key in keys:
        vals = [float(row[key]) for row in rows]
        result["mean"][key] = statistics.mean(vals)
        result["std"][key] = statistics.pstdev(vals)
    save_json(result, f"{output_prefix}_5seed_summary.json")
    return result


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    source_quotas = parse_source_quotas(args.source_quotas)

    print("loading evidence and decoders...", flush=True)
    train_ms_items = load_items(args.train_ms_evidence_path)
    val_ms_items = load_items(args.val_ms_evidence_path)
    train_adapter_items = load_items(args.train_adapter_evidence_path)
    val_adapter_items = load_items(args.val_adapter_evidence_path)
    ms_decoder = load_decoder_checkpoint(args.ms_decoder_ckpt, device)
    adapter_decoder = load_decoder_checkpoint(args.adapter_decoder_ckpt, device)

    print("applying decoders once...", flush=True)
    train_ms_records, _ = apply_decoder(train_ms_items, ms_decoder, device, "ms_clap")
    val_ms_records, val_ms_metrics = apply_decoder(val_ms_items, ms_decoder, device, "ms_clap")
    train_adapter_records, _ = apply_decoder(train_adapter_items, adapter_decoder, device, "adapter")
    val_adapter_records, val_adapter_metrics = apply_decoder(val_adapter_items, adapter_decoder, device, "adapter")

    print("building candidates once...", flush=True)
    train_rows = build_fusion_rows(
        train_ms_items,
        train_adapter_items,
        train_ms_records,
        train_adapter_records,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    val_rows = build_fusion_rows(
        val_ms_items,
        val_adapter_items,
        val_ms_records,
        val_adapter_records,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    train_rows = add_semantic_temporal_labels(train_rows, train_ms_records, quality_target=args.quality_target)
    val_rows = add_semantic_temporal_labels(val_rows, val_ms_records, quality_target=args.quality_target)
    train_dataset = SemanticTemporalCandidateDataset(train_rows)
    val_dataset = SemanticTemporalCandidateDataset(val_rows, feature_mean=train_dataset.feature_mean, feature_std=train_dataset.feature_std)
    print("feature dim:", int(train_dataset.features.shape[1]), flush=True)
    print("label counts:", json.dumps({"train": label_counts(train_dataset), "val": label_counts(val_dataset)}, indent=2), flush=True)

    cached = {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "val_ms_items": val_ms_items,
        "val_ms_records": val_ms_records,
        "val_adapter_records": val_adapter_records,
        "val_ms_metrics": val_ms_metrics,
        "val_adapter_metrics": val_adapter_metrics,
    }
    summaries = []
    for seed in seeds:
        print(f"===== cached seed {seed} start =====", flush=True)
        summaries.append(run_seed(args, seed, cached, device))
        print(f"===== cached seed {seed} done =====", flush=True)
    agg = aggregate(summaries, args.output_prefix)
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
