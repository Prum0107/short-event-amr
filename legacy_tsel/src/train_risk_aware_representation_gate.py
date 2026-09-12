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

from train_learned_evidence_decoder import load_items
from train_representation_fusion_selector import (
    align_items,
    mode_counts,
    oracle_select,
    quality_key,
    representation_feature_row,
    select_predictions,
    summarize_interventions,
)
from train_two_mode_decoder_selector import (
    apply_decoder,
    fmt_float,
    fmt_pct,
    load_decoder_checkpoint,
    safe_div,
    save_json,
    save_jsonl,
    table,
)


def utility_delta(ms_record, adapter_record):
    ms_top1 = float(ms_record.get("top1_iou", 0.0))
    adapter_top1 = float(adapter_record.get("top1_iou", 0.0))
    delta = adapter_top1 - ms_top1
    delta += 0.50 * (float(adapter_top1 >= 0.7) - float(ms_top1 >= 0.7))
    delta += 0.25 * (float(adapter_top1 >= 0.5) - float(ms_top1 >= 0.5))
    if ms_record.get("category") == "semantic_miss" and adapter_record.get("category") != "semantic_miss":
        delta += 0.20
    if ms_record.get("category") == "good" and adapter_record.get("category") != "good":
        delta -= 0.30
    return float(max(-1.5, min(1.5, delta)))


def build_gate_dataset(ms_items, adapter_items, ms_records, adapter_records):
    features = []
    gain_labels = []
    risk_labels = []
    good_risk_labels = []
    utility_targets = []
    qids = []
    for ms_item, adapter_item in align_items(ms_items, adapter_items):
        qid = ms_item["qid"]
        if qid not in ms_records or qid not in adapter_records:
            continue
        ms_record = ms_records[qid]
        adapter_record = adapter_records[qid]
        ms_key = quality_key(ms_record)
        adapter_key = quality_key(adapter_record)
        features.append(representation_feature_row(ms_item, adapter_item, ms_record, adapter_record))
        gain_labels.append(1.0 if adapter_key > ms_key else 0.0)
        risk_labels.append(1.0 if adapter_key < ms_key else 0.0)
        good_risk_labels.append(1.0 if ms_record.get("category") == "good" and adapter_record.get("category") != "good" else 0.0)
        utility_targets.append(utility_delta(ms_record, adapter_record))
        qids.append(qid)
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(gain_labels, dtype=torch.float32),
        torch.tensor(risk_labels, dtype=torch.float32),
        torch.tensor(good_risk_labels, dtype=torch.float32),
        torch.tensor(utility_targets, dtype=torch.float32),
        qids,
    )


class RiskAwareRepresentationGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=48, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.good_risk_head = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        hidden = self.shared(x)
        return {
            "gain_logit": self.gain_head(hidden).squeeze(-1),
            "risk_logit": self.risk_head(hidden).squeeze(-1),
            "good_risk_logit": self.good_risk_head(hidden).squeeze(-1),
            "utility": self.utility_head(hidden).squeeze(-1),
        }


def pos_weight(labels):
    pos = float(labels.sum().item())
    neg = float(len(labels) - pos)
    return torch.tensor([safe_div(neg, pos) if pos > 0 else 1.0], dtype=torch.float32)


def train_gate(x_train, gain_y, risk_y, good_risk_y, utility_y, args):
    torch.manual_seed(args.seed)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp(min=1e-6)
    x_norm = (x_train - feature_mean) / feature_std
    loader = DataLoader(
        TensorDataset(x_norm, gain_y, risk_y, good_risk_y, utility_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = RiskAwareRepresentationGate(x_train.shape[1], args.hidden_dim, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gain_weight = pos_weight(gain_y)
    risk_weight = pos_weight(risk_y)
    good_risk_weight = pos_weight(good_risk_y)
    for _ in range(args.epochs):
        model.train()
        for x_batch, gain_batch, risk_batch, good_risk_batch, utility_batch in loader:
            outputs = model(x_batch)
            gain_loss = F.binary_cross_entropy_with_logits(outputs["gain_logit"], gain_batch, pos_weight=gain_weight)
            risk_loss = F.binary_cross_entropy_with_logits(outputs["risk_logit"], risk_batch, pos_weight=risk_weight)
            good_risk_loss = F.binary_cross_entropy_with_logits(
                outputs["good_risk_logit"],
                good_risk_batch,
                pos_weight=good_risk_weight,
            )
            utility_loss = F.smooth_l1_loss(outputs["utility"], utility_batch)
            loss = (
                gain_loss
                + args.risk_loss_weight * risk_loss
                + args.good_risk_loss_weight * good_risk_loss
                + args.utility_loss_weight * utility_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
    return model, feature_mean, feature_std


@torch.no_grad()
def predict_gate(model, feature_mean, feature_std, x):
    model.eval()
    outputs = model((x - feature_mean) / feature_std)
    gain = torch.sigmoid(outputs["gain_logit"]).cpu().tolist()
    risk = torch.sigmoid(outputs["risk_logit"]).cpu().tolist()
    good_risk = torch.sigmoid(outputs["good_risk_logit"]).cpu().tolist()
    utility = outputs["utility"].cpu().tolist()
    return [
        {
            "gain": float(g),
            "risk": float(r),
            "good_risk": float(gr),
            "utility": float(u),
            "score": float(g - r - 0.5 * gr + 0.5 * u),
        }
        for g, r, gr, u in zip(gain, risk, good_risk, utility)
    ]


def gate_qids(items, predictions, params):
    candidates = []
    for item, pred in zip(items, predictions):
        if (
            pred["gain"] >= params["min_gain"]
            and pred["risk"] <= params["max_risk"]
            and pred["good_risk"] <= params["max_good_risk"]
            and pred["utility"] >= params["min_utility"]
        ):
            score = (
                pred["gain"]
                - params["risk_alpha"] * pred["risk"]
                - params["good_risk_alpha"] * pred["good_risk"]
                + params["utility_alpha"] * pred["utility"]
            )
            candidates.append((item["qid"], score))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    budget = int(math.ceil(float(params["budget"]) * len(items)))
    return set(qid for qid, _ in candidates[:budget])


def apply_gate(items, ms_records, adapter_records, predictions, params):
    qids = gate_qids(items, predictions, params)
    probs = [1.0 if item["qid"] in qids else 0.0 for item in items]
    return select_predictions(items, ms_records, adapter_records, probs, threshold=0.5)


def tune_gate(items, ms_records, adapter_records, predictions, objective):
    grid = []
    for min_gain in [0.45, 0.55, 0.65, 0.75, 0.85]:
        for max_risk in [0.15, 0.25, 0.35, 0.45, 0.55]:
            for max_good_risk in [0.08, 0.15, 0.25, 0.35, 0.5]:
                for min_utility in [-0.10, 0.0, 0.05, 0.10, 0.20]:
                    for budget in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
                        grid.append(
                            {
                                "min_gain": min_gain,
                                "max_risk": max_risk,
                                "max_good_risk": max_good_risk,
                                "min_utility": min_utility,
                                "risk_alpha": 1.0,
                                "good_risk_alpha": 0.8,
                                "utility_alpha": 0.4,
                                "budget": budget,
                            }
                        )

    best = None
    for params in grid:
        metrics, rows = apply_gate(items, ms_records, adapter_records, predictions, params)
        summary = summarize_interventions(rows)
        if objective == "balanced":
            key = (
                metrics["R1@0.7"],
                metrics["R1@0.5"],
                metrics["top1_iou"],
                -summary["good_regressed"],
                -summary["regressed"],
                summary["intervention_precision"],
                -summary["changed"],
            )
        elif objective == "precision":
            if summary["changed"] < 20:
                continue
            key = (
                -summary["good_regressed"],
                -summary["regressed"],
                summary["intervention_precision"],
                metrics["R1@0.7"],
                metrics["top1_iou"],
                summary["changed"],
            )
        else:
            raise ValueError(f"unknown objective: {objective}")
        if best is None or key > best["key"]:
            best = {
                "params": params,
                "metrics": metrics,
                "rows": rows,
                "summary": summary,
                "key": key,
            }
    return best


def gate_prediction_summary(predictions, gain_y, risk_y, good_risk_y, utility_y):
    def avg(values):
        values = list(values)
        return sum(values) / max(len(values), 1)

    adapter_better = [pred for pred, label in zip(predictions, gain_y.tolist()) if label > 0.5]
    ms_better = [pred for pred, label in zip(predictions, risk_y.tolist()) if label > 0.5]
    good_risk = [pred for pred, label in zip(predictions, good_risk_y.tolist()) if label > 0.5]
    return {
        "avg_gain_prob": avg(pred["gain"] for pred in predictions),
        "avg_risk_prob": avg(pred["risk"] for pred in predictions),
        "avg_good_risk_prob": avg(pred["good_risk"] for pred in predictions),
        "avg_utility_pred": avg(pred["utility"] for pred in predictions),
        "adapter_better_avg_gain": avg(pred["gain"] for pred in adapter_better),
        "adapter_better_avg_risk": avg(pred["risk"] for pred in adapter_better),
        "ms_better_avg_gain": avg(pred["gain"] for pred in ms_better),
        "ms_better_avg_risk": avg(pred["risk"] for pred in ms_better),
        "good_risk_avg_good_risk": avg(pred["good_risk"] for pred in good_risk),
        "avg_true_utility": avg(utility_y.tolist()),
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
    for name, stats in summary["interventions"].items():
        intervention_rows.append(
            [
                escape(name),
                str(stats["changed"]),
                str(stats["improved"]),
                str(stats["regressed"]),
                str(stats["recovered_semantic_miss"]),
                str(stats["good_regressed"]),
                fmt_pct(stats["intervention_precision"]),
            ]
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Risk-Aware Representation Gate V2</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #b45309; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #f1ede6; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Risk-Aware Representation Gate V2</h1>
  <div class="note">
    Default is MS-CLAP. Adapter intervention is modeled as a gain/risk decision.
    Balanced params: <code>{escape(json.dumps(summary["balanced_params"]))}</code>.
    Precision params: <code>{escape(json.dumps(summary["precision_params"]))}</code>.
  </div>
  <h2>Validation Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Adapter Interventions</h2>
  {table(["System", "Changed", "Improved", "Regressed", "Recovered Semantic Miss", "Good Regressed", "Precision"], intervention_rows)}
</body>
</html>"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ms_evidence_path", required=True)
    parser.add_argument("--val_ms_evidence_path", required=True)
    parser.add_argument("--train_adapter_evidence_path", required=True)
    parser.add_argument("--val_adapter_evidence_path", required=True)
    parser.add_argument("--ms_decoder_ckpt", required=True)
    parser.add_argument("--adapter_decoder_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--risk_loss_weight", type=float, default=1.2)
    parser.add_argument("--good_risk_loss_weight", type=float, default=1.6)
    parser.add_argument("--utility_loss_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ms_items = load_items(args.train_ms_evidence_path)
    val_ms_items = load_items(args.val_ms_evidence_path)
    train_adapter_items = load_items(args.train_adapter_evidence_path)
    val_adapter_items = load_items(args.val_adapter_evidence_path)
    ms_decoder = load_decoder_checkpoint(args.ms_decoder_ckpt, device)
    adapter_decoder = load_decoder_checkpoint(args.adapter_decoder_ckpt, device)

    print("applying MS-CLAP decoder to train...")
    train_ms_records, train_ms_metrics = apply_decoder(train_ms_items, ms_decoder, device, "ms_clap")
    print("applying adapter decoder to train...")
    train_adapter_records, train_adapter_metrics = apply_decoder(train_adapter_items, adapter_decoder, device, "adapter")
    print("applying MS-CLAP decoder to val...")
    val_ms_records, val_ms_metrics = apply_decoder(val_ms_items, ms_decoder, device, "ms_clap")
    print("applying adapter decoder to val...")
    val_adapter_records, val_adapter_metrics = apply_decoder(val_adapter_items, adapter_decoder, device, "adapter")

    x_train, gain_train, risk_train, good_risk_train, utility_train, _ = build_gate_dataset(
        train_ms_items,
        train_adapter_items,
        train_ms_records,
        train_adapter_records,
    )
    x_val, gain_val, risk_val, good_risk_val, utility_val, _ = build_gate_dataset(
        val_ms_items,
        val_adapter_items,
        val_ms_records,
        val_adapter_records,
    )
    print(
        "gate labels:",
        {
            "train_gain": int(gain_train.sum().item()),
            "train_risk": int(risk_train.sum().item()),
            "train_good_risk": int(good_risk_train.sum().item()),
            "val_gain": int(gain_val.sum().item()),
            "val_risk": int(risk_val.sum().item()),
            "val_good_risk": int(good_risk_val.sum().item()),
        },
    )

    gate, feature_mean, feature_std = train_gate(
        x_train,
        gain_train,
        risk_train,
        good_risk_train,
        utility_train,
        args,
    )
    train_predictions = predict_gate(gate, feature_mean, feature_std, x_train)
    val_predictions = predict_gate(gate, feature_mean, feature_std, x_val)

    balanced_train = tune_gate(
        train_ms_items,
        train_ms_records,
        train_adapter_records,
        train_predictions,
        objective="balanced",
    )
    precision_train = tune_gate(
        train_ms_items,
        train_ms_records,
        train_adapter_records,
        train_predictions,
        objective="precision",
    )
    balanced_metrics, balanced_rows = apply_gate(
        val_ms_items,
        val_ms_records,
        val_adapter_records,
        val_predictions,
        balanced_train["params"],
    )
    precision_metrics, precision_rows = apply_gate(
        val_ms_items,
        val_ms_records,
        val_adapter_records,
        val_predictions,
        precision_train["params"],
    )
    balanced_summary = summarize_interventions(balanced_rows)
    precision_summary = summarize_interventions(precision_rows)
    oracle_metrics, oracle_rows, oracle_summary = oracle_select(val_ms_items, val_ms_records, val_adapter_records)
    oracle_budget_metrics, oracle_budget_rows, oracle_budget_summary = oracle_select(
        val_ms_items,
        val_ms_records,
        val_adapter_records,
        budget=0.2,
    )

    summary = {
        "balanced_params": balanced_train["params"],
        "precision_params": precision_train["params"],
        "label_counts": {
            "train_gain": int(gain_train.sum().item()),
            "train_risk": int(risk_train.sum().item()),
            "train_good_risk": int(good_risk_train.sum().item()),
            "val_gain": int(gain_val.sum().item()),
            "val_risk": int(risk_val.sum().item()),
            "val_good_risk": int(good_risk_val.sum().item()),
        },
        "prediction_summary": {
            "train": gate_prediction_summary(train_predictions, gain_train, risk_train, good_risk_train, utility_train),
            "val": gate_prediction_summary(val_predictions, gain_val, risk_val, good_risk_val, utility_val),
        },
        "metrics": {
            "ms_clap_shape_v2_top2": val_ms_metrics,
            "adapter_shape_v2_top2": val_adapter_metrics,
            "risk_aware_balanced_gate": balanced_metrics,
            "risk_aware_precision_gate": precision_metrics,
            "oracle_ms_or_adapter": oracle_metrics,
            "oracle_budget_20pct": oracle_budget_metrics,
        },
        "interventions": {
            "risk_aware_balanced_gate": balanced_summary,
            "risk_aware_precision_gate": precision_summary,
            "oracle_ms_or_adapter": oracle_summary,
            "oracle_budget_20pct": oracle_budget_summary,
        },
        "mode_counts": {
            "balanced": mode_counts(balanced_rows),
            "precision": mode_counts(precision_rows),
            "oracle": mode_counts(oracle_rows),
            "oracle_budget_20pct": mode_counts(oracle_budget_rows),
        },
        "args": vars(args),
    }

    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(balanced_rows, os.path.join(args.output_dir, "balanced_case_rows.json"))
    save_json(precision_rows, os.path.join(args.output_dir, "precision_case_rows.json"))
    save_json(oracle_rows, os.path.join(args.output_dir, "oracle_case_rows.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in balanced_rows],
        os.path.join(args.output_dir, "balanced_predictions.jsonl"),
    )
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in precision_rows],
        os.path.join(args.output_dir, "precision_predictions.jsonl"),
    )
    torch.save(
        {
            "model_state_dict": gate.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "balanced_params": balanced_train["params"],
            "precision_params": precision_train["params"],
            "args": vars(args),
        },
        os.path.join(args.output_dir, "gate.pt"),
    )
    write_html(args.output_dir, summary)
    print(json.dumps(summary, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
