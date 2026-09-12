import argparse
import json
import math
import os
import random
from collections import Counter
from xml.sax.saxutils import escape

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_conservative_transformer_gate import (
    SmallGate,
    apply_interventions,
    build_gate_dataset,
    gate_feature_row,
    gate_stats_for_item,
    summarize_interventions,
    transformer_better,
)
from train_two_mode_decoder_selector import fmt_float, fmt_pct, safe_div, save_json, save_jsonl, table


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rows_by_qid(rows):
    return {row["qid"]: row for row in rows}


def records_to_dict(mode_records):
    return {mode: rows_by_qid(rows) for mode, rows in mode_records.items()}


def quality_flags(record):
    top1 = float(record.get("top1_iou", 0.0))
    return {
        "strict": top1 >= 0.7,
        "loose": top1 >= 0.5,
        "top1": top1,
        "top5": float(record.get("top5_iou", 0.0)),
        "category": record.get("category", ""),
    }


def utility_label(default, transformer):
    d = quality_flags(default)
    t = quality_flags(transformer)
    top1_delta = t["top1"] - d["top1"]
    top5_delta = t["top5"] - d["top5"]
    semantic_recovery = d["category"] == "semantic_miss" and t["category"] != "semantic_miss"
    semantic_jump_harm = d["category"] != "semantic_miss" and t["category"] == "semantic_miss"
    good_regression = d["category"] == "good" and t["category"] != "good"
    to_good = d["category"] != "good" and t["category"] == "good"
    strict_cross = (not d["strict"]) and t["strict"]
    loose_cross = (not d["loose"]) and t["loose"]

    utility = 0.0
    utility += 1.4 if strict_cross else 0.0
    utility += 0.8 if loose_cross else 0.0
    utility += 0.6 if to_good else 0.0
    utility += 0.5 if semantic_recovery else 0.0
    utility += max(top1_delta, 0.0)
    utility += 0.15 * max(top5_delta, 0.0)
    utility -= 1.5 if good_regression else 0.0
    utility -= 1.2 if semantic_jump_harm else 0.0
    utility -= max(-top1_delta, 0.0)
    utility -= 0.10 * max(-top5_delta, 0.0)

    if strict_cross or to_good or semantic_recovery or top1_delta >= 0.15:
        label = "positive"
    elif good_regression or semantic_jump_harm or top1_delta <= -0.12:
        label = "negative"
    else:
        label = "neutral"

    # If the aggregate utility strongly disagrees with the heuristic label, let utility decide.
    if utility >= 0.45:
        label = "positive"
    elif utility <= -0.35:
        label = "negative"

    return {
        "label": label,
        "utility": utility,
        "top1_delta": top1_delta,
        "top5_delta": top5_delta,
        "strict_cross": strict_cross,
        "loose_cross": loose_cross,
        "to_good": to_good,
        "semantic_recovery": semantic_recovery,
        "semantic_jump_harm": semantic_jump_harm,
        "good_regression": good_regression,
    }


def build_utility_dataset(items, records_by_mode):
    features = []
    labels = []
    weights = []
    qids = []
    metadata = []
    for item in items:
        qid = item["qid"]
        records = {mode: records_by_mode[mode][qid] for mode in ["coverage", "precision", "transformer", "default"]}
        meta = utility_label(records["default"], records["transformer"])
        metadata.append({"qid": qid, **meta})
        if meta["label"] == "neutral":
            continue
        features.append(gate_feature_row(item, records))
        labels.append(1.0 if meta["label"] == "positive" else 0.0)
        weights.append(1.0 + min(abs(float(meta["utility"])), 2.0))
        qids.append(qid)
    if not features:
        raise RuntimeError("No non-neutral training examples were found.")
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
        qids,
        metadata,
    )


def build_all_features(items, records_by_mode):
    features = []
    qids = []
    metadata = []
    for item in items:
        qid = item["qid"]
        records = {mode: records_by_mode[mode][qid] for mode in ["coverage", "precision", "transformer", "default"]}
        features.append(gate_feature_row(item, records))
        metadata.append({"qid": qid, **utility_label(records["default"], records["transformer"])})
        qids.append(qid)
    return torch.tensor(features, dtype=torch.float32), qids, metadata


def train_gate(x_train, y_train, sample_weights, args):
    torch.manual_seed(args.seed)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp(min=1e-6)
    x_norm = (x_train - feature_mean) / feature_std
    loader = DataLoader(TensorDataset(x_norm, y_train, sample_weights), batch_size=args.batch_size, shuffle=True)
    model = SmallGate(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
    pos = float(y_train.sum().item())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([safe_div(neg, pos) if pos > 0 else 1.0], dtype=torch.float32)
    for _ in range(args.epochs):
        model.train()
        for x_batch, y_batch, weight_batch in loader:
            logits = model(x_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch, pos_weight=pos_weight, reduction="none")
            loss = (loss * weight_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
    return model, feature_mean, feature_std


@torch.no_grad()
def predict_probs(model, feature_mean, feature_std, x):
    model.eval()
    return torch.sigmoid(model((x - feature_mean) / feature_std)).cpu().tolist()


def select_qids(items, records_by_mode, probs, threshold, min_agreement, max_center_delta, budget):
    candidates = []
    for item, prob in zip(items, probs):
        qid = item["qid"]
        records = {mode: records_by_mode[mode][qid] for mode in ["coverage", "precision", "transformer", "default"]}
        stats = gate_stats_for_item(item, records)
        if (
            prob >= threshold
            and stats["tr_best_agreement"] >= min_agreement
            and stats["center_delta"] <= max_center_delta
            and records["default"].get("category", "") != "good"
        ):
            candidates.append(
                {
                    "qid": qid,
                    "score": float(prob) + 0.15 * float(stats["tr_best_agreement"]) - 0.1 * float(stats["center_delta"]),
                }
            )
    candidates = sorted(candidates, key=lambda row: row["score"], reverse=True)
    max_changes = int(math.ceil(float(budget) * len(items)))
    return set(row["qid"] for row in candidates[:max_changes])


def metric_key(metrics, intervention_summary):
    return (
        metrics["R1@0.7"],
        metrics["R1@0.5"],
        metrics["top1_iou"],
        -metrics["categories"].get("semantic_miss", 0),
        -intervention_summary["good_regressed"],
        intervention_summary["intervention_precision"],
        -intervention_summary["changed"],
    )


def tune_gate(items, records_by_mode, probs):
    best = None
    for threshold in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        for min_agreement in [0.2, 0.35, 0.5, 0.65]:
            for max_center_delta in [0.25, 0.4, 0.6, 0.85]:
                for budget in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15]:
                    qids = select_qids(items, records_by_mode, probs, threshold, min_agreement, max_center_delta, budget)
                    metrics, rows = apply_interventions(items, records_by_mode, qids, "utility_aware")
                    summary = summarize_interventions(rows)
                    key = metric_key(metrics, summary)
                    if best is None or key > best["key"]:
                        best = {
                            "key": key,
                            "params": {
                                "threshold": threshold,
                                "min_agreement": min_agreement,
                                "max_center_delta": max_center_delta,
                                "budget": budget,
                            },
                            "metrics": metrics,
                            "rows": rows,
                            "intervention_summary": summary,
                        }
    return best


def oracle_utility(items, records_by_mode, budget=None):
    candidates = []
    for item in items:
        qid = item["qid"]
        meta = utility_label(records_by_mode["default"][qid], records_by_mode["transformer"][qid])
        if meta["utility"] > 0:
            candidates.append((qid, meta["utility"]))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    if budget is not None:
        candidates = candidates[: int(math.ceil(float(budget) * len(items)))]
    qids = set(qid for qid, _ in candidates)
    metrics, rows = apply_interventions(items, records_by_mode, qids, "utility_oracle")
    return metrics, rows, summarize_interventions(rows), len(qids)


def label_summary(metadata):
    counts = Counter(row["label"] for row in metadata)
    return {
        "counts": dict(counts),
        "avg_utility_by_label": {
            label: sum(row["utility"] for row in metadata if row["label"] == label) / max(counts.get(label, 0), 1)
            for label in ["positive", "neutral", "negative"]
        },
    }


def write_html(output_dir, summary):
    metric_rows = []
    for name, metrics in summary["metrics"].items():
        metric_rows.append(
            [
                escape(name),
                fmt_pct(metrics["R1@0.5"]),
                fmt_pct(metrics["R1@0.7"]),
                fmt_pct(metrics["R3@0.7"]),
                fmt_pct(metrics["R5@0.7"]),
                fmt_float(metrics["top1_iou"]),
                fmt_float(metrics["best_iou_top5"]),
                str(metrics["categories"].get("semantic_miss", 0)),
                str(metrics["categories"].get("good", 0)),
            ]
        )
    intervention_rows = []
    for name, data in summary["interventions"].items():
        intervention_rows.append(
            [
                escape(name),
                str(data["changed"]),
                str(data["improved"]),
                str(data["regressed"]),
                str(data["same"]),
                str(data["recovered_semantic_miss"]),
                str(data["good_regressed"]),
                fmt_pct(data["intervention_precision"]),
            ]
        )
    label_rows = []
    for split in ["train", "val"]:
        counts = summary[f"{split}_label_summary"]["counts"]
        label_rows.append(
            [
                split,
                str(counts.get("positive", 0)),
                str(counts.get("neutral", 0)),
                str(counts.get("negative", 0)),
                fmt_float(summary[f"{split}_label_summary"]["avg_utility_by_label"].get("positive", 0.0)),
                fmt_float(summary[f"{split}_label_summary"]["avg_utility_by_label"].get("negative", 0.0)),
            ]
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Utility-Aware Transformer Gate V5.3</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #7c2d12; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #f1e8e2; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Utility-Aware Transformer Gate V5.3</h1>
  <div class="note">
    This gate trains only on non-neutral interventions. Strong positives include strict/loose crossings, semantic recovery,
    or large top1 gains. Strong negatives include good regressions, semantic jumps, and large top1 losses.
    Tuned params: <code>{escape(json.dumps(summary["params"]))}</code>.
  </div>
  <h2>Validation Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Interventions</h2>
  {table(["Gate", "Changed", "Improved", "Regressed", "Same", "Recovered Semantic Miss", "Good Regressed", "Precision"], intervention_rows)}
  <h2>Utility Labels</h2>
  {table(["Split", "Positive", "Neutral", "Negative", "Avg Positive Utility", "Avg Negative Utility"], label_rows)}
</body>
</html>"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_evidence_path", required=True)
    parser.add_argument("--val_evidence_path", required=True)
    parser.add_argument("--train_mode_records_path", required=True)
    parser.add_argument("--val_mode_records_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_items = load_json(args.train_evidence_path)
    val_items = load_json(args.val_evidence_path)
    train_records = records_to_dict(load_json(args.train_mode_records_path))
    val_records = records_to_dict(load_json(args.val_mode_records_path))

    x_train, y_train, weights, train_qids, train_meta_non_neutral = build_utility_dataset(train_items, train_records)
    x_train_all, _, train_meta_all = build_all_features(train_items, train_records)
    x_val_all, val_qids, val_meta_all = build_all_features(val_items, val_records)

    model, feature_mean, feature_std = train_gate(x_train, y_train, weights, args)
    train_probs = predict_probs(model, feature_mean, feature_std, x_train_all)
    val_probs = predict_probs(model, feature_mean, feature_std, x_val_all)
    best = tune_gate(train_items, train_records, train_probs)

    val_qids_selected = select_qids(
        val_items,
        val_records,
        val_probs,
        best["params"]["threshold"],
        best["params"]["min_agreement"],
        best["params"]["max_center_delta"],
        best["params"]["budget"],
    )
    val_metrics, val_rows = apply_interventions(val_items, val_records, val_qids_selected, "utility_aware")
    val_intervention_summary = summarize_interventions(val_rows)
    utility_oracle_metrics, utility_oracle_rows, utility_oracle_summary, utility_oracle_changed = oracle_utility(
        val_items,
        val_records,
        budget=None,
    )
    utility_budget_metrics, utility_budget_rows, utility_budget_summary, utility_budget_changed = oracle_utility(
        val_items,
        val_records,
        budget=0.15,
    )

    default_metrics = load_json(args.val_mode_records_path.replace("val_mode_records.json", "val_mode_metrics.json"))[
        "two_mode_default"
    ]
    transformer_metrics = load_json(args.val_mode_records_path.replace("val_mode_records.json", "val_mode_metrics.json"))[
        "transformer_source_gate"
    ]
    summary = {
        "params": best["params"],
        "train_label_summary": label_summary(train_meta_all),
        "val_label_summary": label_summary(val_meta_all),
        "non_neutral_train_examples": len(train_qids),
        "metrics": {
            "two_mode_default": default_metrics,
            "transformer_direct": transformer_metrics,
            "utility_aware_gate_v53": val_metrics,
            "utility_oracle": utility_oracle_metrics,
            "utility_oracle_budget_15pct": utility_budget_metrics,
        },
        "train_metrics": {
            "utility_aware_gate_v53": best["metrics"],
        },
        "interventions": {
            "utility_aware_val": val_intervention_summary,
            "utility_oracle_val": utility_oracle_summary,
            "utility_oracle_budget_15pct_val": utility_budget_summary,
        },
        "utility_oracle_changed": utility_oracle_changed,
        "utility_oracle_budget_15pct_changed": utility_budget_changed,
        "args": vars(args),
    }
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(val_rows, os.path.join(args.output_dir, "utility_case_rows.json"))
    save_json(utility_oracle_rows, os.path.join(args.output_dir, "utility_oracle_case_rows.json"))
    save_json(val_meta_all, os.path.join(args.output_dir, "val_utility_labels.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in val_rows],
        os.path.join(args.output_dir, "utility_predictions.jsonl"),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "params": best["params"],
            "args": vars(args),
        },
        os.path.join(args.output_dir, "utility_gate.pt"),
    )
    write_html(args.output_dir, summary)
    print(json.dumps(summary, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
