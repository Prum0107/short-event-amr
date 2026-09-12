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

from train_conservative_transformer_gate import apply_interventions, gate_feature_row, gate_stats_for_item, summarize_interventions
from train_two_mode_decoder_selector import fmt_float, fmt_pct, safe_div, save_json, save_jsonl, table
from train_utility_aware_transformer_gate import records_to_dict, utility_label


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def target_row(default, transformer):
    meta = utility_label(default, transformer)
    top1_delta = float(meta["top1_delta"])
    gain = (
        meta["strict_cross"]
        or meta["loose_cross"]
        or meta["to_good"]
        or meta["semantic_recovery"]
        or top1_delta >= 0.15
        or float(meta["utility"]) >= 0.45
    )
    risk = (
        meta["good_regression"]
        or meta["semantic_jump_harm"]
        or top1_delta <= -0.12
        or float(meta["utility"]) <= -0.35
    )
    gain_weight = 0.35 + max(float(meta["utility"]), 0.0)
    risk_weight = 0.35 + max(-float(meta["utility"]), 0.0)
    if meta["good_regression"] or meta["semantic_jump_harm"]:
        risk_weight += 1.0
    return {
        **meta,
        "gain_target": 1.0 if gain else 0.0,
        "risk_target": 1.0 if risk else 0.0,
        "gain_weight": gain_weight,
        "risk_weight": risk_weight,
    }


def build_dataset(items, records_by_mode):
    features = []
    gain_targets = []
    risk_targets = []
    gain_weights = []
    risk_weights = []
    metadata = []
    qids = []
    for item in items:
        qid = item["qid"]
        records = {mode: records_by_mode[mode][qid] for mode in ["coverage", "precision", "transformer", "default"]}
        meta = target_row(records["default"], records["transformer"])
        features.append(gate_feature_row(item, records))
        gain_targets.append(meta["gain_target"])
        risk_targets.append(meta["risk_target"])
        gain_weights.append(meta["gain_weight"])
        risk_weights.append(meta["risk_weight"])
        metadata.append({"qid": qid, **meta})
        qids.append(qid)
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(gain_targets, dtype=torch.float32),
        torch.tensor(risk_targets, dtype=torch.float32),
        torch.tensor(gain_weights, dtype=torch.float32),
        torch.tensor(risk_weights, dtype=torch.float32),
        qids,
        metadata,
    )


class DualHeadGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=24, dropout=0.08):
        super().__init__()
        self.backbone = nn.Sequential(
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

    def forward(self, x):
        hidden = self.backbone(x)
        return self.gain_head(hidden).squeeze(-1), self.risk_head(hidden).squeeze(-1)


def weighted_bce(logits, targets, weights, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    return (loss * weights).mean()


def train_gate(x_train, gain_train, risk_train, gain_weights, risk_weights, args):
    torch.manual_seed(args.seed)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp(min=1e-6)
    x_norm = (x_train - feature_mean) / feature_std
    loader = DataLoader(
        TensorDataset(x_norm, gain_train, risk_train, gain_weights, risk_weights),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = DualHeadGate(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-4)

    gain_pos = float(gain_train.sum().item())
    gain_neg = float(len(gain_train) - gain_pos)
    risk_pos = float(risk_train.sum().item())
    risk_neg = float(len(risk_train) - risk_pos)
    gain_pos_weight = torch.tensor([safe_div(gain_neg, gain_pos) if gain_pos > 0 else 1.0], dtype=torch.float32)
    risk_pos_weight = torch.tensor([safe_div(risk_neg, risk_pos) if risk_pos > 0 else 1.0], dtype=torch.float32)

    for _ in range(args.epochs):
        model.train()
        for x_batch, gain_batch, risk_batch, gain_w_batch, risk_w_batch in loader:
            gain_logits, risk_logits = model(x_batch)
            gain_loss = weighted_bce(gain_logits, gain_batch, gain_w_batch, gain_pos_weight)
            risk_loss = weighted_bce(risk_logits, risk_batch, risk_w_batch, risk_pos_weight)
            loss = gain_loss + args.risk_loss_weight * risk_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
    return model, feature_mean, feature_std


@torch.no_grad()
def predict(model, feature_mean, feature_std, x):
    model.eval()
    gain_logits, risk_logits = model((x - feature_mean) / feature_std)
    return torch.sigmoid(gain_logits).cpu().tolist(), torch.sigmoid(risk_logits).cpu().tolist()


def select_qids(items, records_by_mode, gain_probs, risk_probs, params):
    candidates = []
    for item, gain, risk in zip(items, gain_probs, risk_probs):
        qid = item["qid"]
        records = {mode: records_by_mode[mode][qid] for mode in ["coverage", "precision", "transformer", "default"]}
        stats = gate_stats_for_item(item, records)
        score = float(gain) - float(params["lambda_risk"]) * float(risk)
        if (
            score >= float(params["score_threshold"])
            and stats["tr_best_agreement"] >= float(params["min_agreement"])
            and stats["center_delta"] <= float(params["max_center_delta"])
            and (not params["protect_good"] or records["default"].get("category", "") != "good")
        ):
            candidates.append(
                {
                    "qid": qid,
                    "score": score + 0.05 * float(stats["tr_best_agreement"]) - 0.05 * float(stats["center_delta"]),
                    "gain": float(gain),
                    "risk": float(risk),
                }
            )
    candidates = sorted(candidates, key=lambda row: row["score"], reverse=True)
    max_changes = int(math.ceil(float(params["budget"]) * len(items)))
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


def tune_params(items, records_by_mode, gain_probs, risk_probs):
    best = None
    for lambda_risk in [0.7, 1.0, 1.3, 1.6]:
        for score_threshold in [-0.05, 0.0, 0.1, 0.2, 0.3]:
            for min_agreement in [0.0, 0.2, 0.4]:
                for max_center_delta in [0.4, 0.7]:
                    for budget in [0.05, 0.1, 0.15]:
                        for protect_good in [True, False]:
                            params = {
                                "lambda_risk": lambda_risk,
                                "score_threshold": score_threshold,
                                "min_agreement": min_agreement,
                                "max_center_delta": max_center_delta,
                                "budget": budget,
                                "protect_good": protect_good,
                            }
                            qids = select_qids(items, records_by_mode, gain_probs, risk_probs, params)
                            metrics, rows = apply_interventions(items, records_by_mode, qids, "dual_head")
                            summary = summarize_interventions(rows)
                            key = metric_key(metrics, summary)
                            if best is None or key > best["key"]:
                                best = {
                                    "key": key,
                                    "params": params,
                                    "metrics": metrics,
                                    "rows": rows,
                                    "intervention_summary": summary,
                                }
    return best


def label_summary(metadata):
    counts = Counter()
    for row in metadata:
        if row["gain_target"] > 0.5 and row["risk_target"] > 0.5:
            counts["gain_and_risk"] += 1
        elif row["gain_target"] > 0.5:
            counts["gain"] += 1
        elif row["risk_target"] > 0.5:
            counts["risk"] += 1
        else:
            counts["neutral"] += 1
    return {
        "counts": dict(counts),
        "avg_utility_gain": sum(row["utility"] for row in metadata if row["gain_target"] > 0.5)
        / max(sum(1 for row in metadata if row["gain_target"] > 0.5), 1),
        "avg_utility_risk": sum(row["utility"] for row in metadata if row["risk_target"] > 0.5)
        / max(sum(1 for row in metadata if row["risk_target"] > 0.5), 1),
    }


def add_prob_rows(rows, gain_probs, risk_probs):
    by_qid = {}
    for row, gain, risk in zip(rows, gain_probs, risk_probs):
        by_qid[row["qid"]] = {"gain_prob": float(gain), "risk_prob": float(risk)}
    return by_qid


def oracle_gain_risk(items, records_by_mode, budget=0.15):
    candidates = []
    for item in items:
        qid = item["qid"]
        meta = target_row(records_by_mode["default"][qid], records_by_mode["transformer"][qid])
        score = float(meta["utility"])
        if score > 0:
            candidates.append((qid, score))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    selected = set(qid for qid, _ in candidates[: int(math.ceil(float(budget) * len(items)))])
    metrics, rows = apply_interventions(items, records_by_mode, selected, "dual_head_oracle")
    return metrics, rows, summarize_interventions(rows), len(selected)


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
                str(counts.get("gain", 0)),
                str(counts.get("risk", 0)),
                str(counts.get("gain_and_risk", 0)),
                str(counts.get("neutral", 0)),
                fmt_float(summary[f"{split}_label_summary"]["avg_utility_gain"]),
                fmt_float(summary[f"{split}_label_summary"]["avg_utility_risk"]),
            ]
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Dual-Head Transformer Gate V5.4</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #4338ca; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #e8e9f4; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Dual-Head Transformer Gate V5.4</h1>
  <div class="note">
    Gain and risk are trained as separate heads. Intervention score is
    <code>gain_prob - lambda * risk_prob</code>. Tuned params:
    <code>{escape(json.dumps(summary["params"]))}</code>.
  </div>
  <h2>Validation Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Interventions</h2>
  {table(["Gate", "Changed", "Improved", "Regressed", "Same", "Recovered Semantic Miss", "Good Regressed", "Precision"], intervention_rows)}
  <h2>Gain/Risk Labels</h2>
  {table(["Split", "Gain", "Risk", "Gain+Risk", "Neutral", "Avg Gain Utility", "Avg Risk Utility"], label_rows)}
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
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--hidden_dim", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--risk_loss_weight", type=float, default=1.2)
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
    train_metrics = load_json(args.train_mode_records_path.replace("train_mode_records.json", "train_mode_metrics.json"))
    val_metrics = load_json(args.val_mode_records_path.replace("val_mode_records.json", "val_mode_metrics.json"))

    (
        x_train,
        gain_train,
        risk_train,
        gain_weights,
        risk_weights,
        train_qids,
        train_meta,
    ) = build_dataset(train_items, train_records)
    x_val, gain_val, risk_val, _, _, val_qids, val_meta = build_dataset(val_items, val_records)

    model, feature_mean, feature_std = train_gate(
        x_train,
        gain_train,
        risk_train,
        gain_weights,
        risk_weights,
        args,
    )
    train_gain_probs, train_risk_probs = predict(model, feature_mean, feature_std, x_train)
    val_gain_probs, val_risk_probs = predict(model, feature_mean, feature_std, x_val)
    best = tune_params(train_items, train_records, train_gain_probs, train_risk_probs)

    selected_val = select_qids(val_items, val_records, val_gain_probs, val_risk_probs, best["params"])
    val_dual_metrics, val_rows = apply_interventions(val_items, val_records, selected_val, "dual_head")
    val_summary = summarize_interventions(val_rows)
    oracle_metrics, oracle_rows, oracle_summary, oracle_changed = oracle_gain_risk(val_items, val_records, budget=0.15)

    prob_by_qid = add_prob_rows([{"qid": qid} for qid in val_qids], val_gain_probs, val_risk_probs)
    for row in val_rows:
        row.update(prob_by_qid.get(row["qid"], {}))

    summary = {
        "params": best["params"],
        "train_label_summary": label_summary(train_meta),
        "val_label_summary": label_summary(val_meta),
        "metrics": {
            "two_mode_default": val_metrics["two_mode_default"],
            "transformer_direct": val_metrics["transformer_source_gate"],
            "dual_head_gate_v54": val_dual_metrics,
            "gain_risk_oracle_budget_15pct": oracle_metrics,
        },
        "train_metrics": {
            "two_mode_default": train_metrics["two_mode_default"],
            "dual_head_gate_v54": best["metrics"],
        },
        "interventions": {
            "dual_head_val": val_summary,
            "gain_risk_oracle_budget_15pct_val": oracle_summary,
        },
        "oracle_changed": oracle_changed,
        "args": vars(args),
    }
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(val_rows, os.path.join(args.output_dir, "dual_head_case_rows.json"))
    save_json(oracle_rows, os.path.join(args.output_dir, "oracle_case_rows.json"))
    save_json(val_meta, os.path.join(args.output_dir, "val_gain_risk_labels.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in val_rows],
        os.path.join(args.output_dir, "dual_head_predictions.jsonl"),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "params": best["params"],
            "args": vars(args),
        },
        os.path.join(args.output_dir, "dual_head_gate.pt"),
    )
    write_html(args.output_dir, summary)
    print(json.dumps(summary, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
