import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from xml.sax.saxutils import escape

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_candidate_level_representation_fusion import (
    build_fusion_rows,
    candidate_source_counts,
    mode_counts,
    oracle_from_rows,
    scored_rows_to_records,
    summarize_interventions,
)
from train_learned_evidence_decoder import CandidateDataset, CandidateScorer, load_items, parse_source_quotas, score_rows
from train_representation_fusion_selector import quality_key, top_window
from train_two_mode_decoder_selector import apply_decoder, evaluate_predictions, fmt_float, fmt_pct, load_decoder_checkpoint, safe_div, table, window_iou


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def mean(values):
    values = list(values)
    return sum(values) / max(len(values), 1)


def std(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def candidate_key(row):
    value = float(row.get("top1_iou", 0.0))
    return (value >= 0.7, value >= 0.5, value)


def intervention_utility(candidate_record, ms_record):
    ms_iou = float(ms_record.get("top1_iou", 0.0))
    cand_iou = float(candidate_record.get("top1_iou", 0.0))
    utility = cand_iou - ms_iou
    utility += 0.50 * (float(cand_iou >= 0.7) - float(ms_iou >= 0.7))
    utility += 0.25 * (float(cand_iou >= 0.5) - float(ms_iou >= 0.5))
    if ms_record.get("category") == "semantic_miss" and candidate_record.get("category") != "semantic_miss":
        utility += 0.20
    if ms_record.get("category") == "good" and candidate_record.get("category") != "good":
        utility -= 0.40
    if ms_record.get("category") == "boundary_error" and candidate_record.get("category") == "semantic_miss":
        utility -= 0.20
    return max(-1.5, min(1.5, float(utility)))


def load_fusion_scorer(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    feature_mean = ckpt["feature_mean"]
    feature_std = ckpt["feature_std"]
    model = CandidateScorer(
        input_dim=int(feature_mean.numel()),
        hidden_dim=int(args.get("hidden_dim", 128)),
        dropout=float(args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "args": args,
    }


def apply_fusion_scorer(items, rows, ms_records, adapter_records, scorer, device):
    dataset = CandidateDataset(rows, feature_mean=scorer["feature_mean"], feature_std=scorer["feature_std"])
    scored = score_rows(scorer["model"], dataset, rows, device)
    return scored_rows_to_records(items, scored, ms_records, adapter_records)


def top_candidate_features(candidate_record, ms_record):
    top_rows = candidate_record.get("top_rows", [])
    top = top_rows[0] if top_rows else {}
    second = top_rows[1] if len(top_rows) > 1 else {}
    top_score = float(top.get("pred_quality", 0.0))
    second_score = float(second.get("pred_quality", 0.0))
    margin = top_score - second_score
    ms_top = top_window(ms_record)
    cand_top = top_window(candidate_record)
    base_features = list(top.get("features", []))
    return base_features + [
        top_score,
        second_score,
        margin,
        mean(float(row.get("pred_quality", 0.0)) for row in top_rows[:5]),
        std(float(row.get("pred_quality", 0.0)) for row in top_rows[:5]),
        float(candidate_record.get("raw_candidate_count", 0)) / 100.0,
        window_iou(cand_top, ms_top),
        safe_div(abs((float(cand_top[0]) + float(cand_top[1])) * 0.5 - (float(ms_top[0]) + float(ms_top[1])) * 0.5), max(float(ms_top[1]) - float(ms_top[0]), 1.0)),
        safe_div(abs((float(cand_top[1]) - float(cand_top[0])) - (float(ms_top[1]) - float(ms_top[0]))), max(float(ms_top[1]) - float(ms_top[0]), 1.0)),
        float(candidate_record.get("chosen_representation") == "adapter"),
        float(candidate_record.get("changed_from_ms", False)),
        float(top.get("source_rank", 0)) / 10.0,
    ]


def build_gate_rows(items, candidate_records, ms_records):
    rows = []
    for item in items:
        qid = item["qid"]
        if qid not in candidate_records or qid not in ms_records:
            continue
        candidate = candidate_records[qid]
        ms_record = ms_records[qid]
        ms_key = quality_key(ms_record)
        cand_key = candidate_key(candidate)
        utility = intervention_utility(candidate, ms_record)
        rows.append(
            {
                "qid": qid,
                "features": top_candidate_features(candidate, ms_record),
                "target_gain": 1.0 if cand_key > ms_key else 0.0,
                "target_risk": 1.0 if cand_key < ms_key else 0.0,
                "target_good_risk": 1.0 if ms_record.get("category") == "good" and candidate.get("category") != "good" else 0.0,
                "target_semantic_recovery": 1.0
                if ms_record.get("category") == "semantic_miss" and candidate.get("category") != "semantic_miss"
                else 0.0,
                "target_intervene": 1.0 if utility > 0.0 else 0.0,
                "target_utility": utility,
            }
        )
    return rows


class GateDataset(Dataset):
    def __init__(self, rows, feature_mean=None, feature_std=None):
        self.rows = rows
        features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
        if feature_mean is None:
            feature_mean = features.mean(dim=0)
        if feature_std is None:
            feature_std = features.std(dim=0, unbiased=False).clamp(min=1e-6)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.features = (features - feature_mean) / feature_std
        self.gain = torch.tensor([row["target_gain"] for row in rows], dtype=torch.float32)
        self.risk = torch.tensor([row["target_risk"] for row in rows], dtype=torch.float32)
        self.good_risk = torch.tensor([row["target_good_risk"] for row in rows], dtype=torch.float32)
        self.intervene = torch.tensor([row["target_intervene"] for row in rows], dtype=torch.float32)
        self.utility = torch.tensor([row["target_utility"] for row in rows], dtype=torch.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.gain[idx],
            self.risk[idx],
            self.good_risk[idx],
            self.intervene[idx],
            self.utility[idx],
        )


class DefaultAnchoredGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=96, dropout=0.12):
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
        self.intervene_head = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.good_risk_head = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return {
            "intervene_logit": self.intervene_head(h).squeeze(-1),
            "gain_logit": self.gain_head(h).squeeze(-1),
            "risk_logit": self.risk_head(h).squeeze(-1),
            "good_risk_logit": self.good_risk_head(h).squeeze(-1),
            "utility": 1.5 * torch.tanh(self.utility_head(h).squeeze(-1)),
        }


def pos_weight(labels, max_value=20.0):
    pos = float(labels.sum().item())
    neg = float(labels.numel() - pos)
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(min(max_value, neg / pos))


def train_gate(model, loader, optimizer, losses, device, args):
    model.train()
    totals = defaultdict(float)
    steps = 0
    for features, gain_y, risk_y, good_risk_y, intervene_y, utility_y in loader:
        features = features.to(device)
        gain_y = gain_y.to(device)
        risk_y = risk_y.to(device)
        good_risk_y = good_risk_y.to(device)
        intervene_y = intervene_y.to(device)
        utility_y = utility_y.to(device)
        outputs = model(features)
        intervene_loss = losses["intervene"](outputs["intervene_logit"], intervene_y)
        gain_loss = losses["gain"](outputs["gain_logit"], gain_y)
        risk_loss = losses["risk"](outputs["risk_logit"], risk_y)
        good_risk_loss = losses["good_risk"](outputs["good_risk_logit"], good_risk_y)
        utility_loss = F.mse_loss(outputs["utility"], utility_y)
        loss = (
            args.intervene_loss_weight * intervene_loss
            + args.gain_loss_weight * gain_loss
            + args.risk_loss_weight * risk_loss
            + args.good_risk_loss_weight * good_risk_loss
            + args.utility_loss_weight * utility_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.item())
        totals["intervene"] += float(intervene_loss.item())
        totals["gain"] += float(gain_loss.item())
        totals["risk"] += float(risk_loss.item())
        totals["good_risk"] += float(good_risk_loss.item())
        totals["utility"] += float(utility_loss.item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def predict_gate(model, dataset, rows, device):
    model.eval()
    features = dataset.features.to(device)
    predictions = []
    for st in range(0, len(features), 4096):
        outputs = model(features[st : st + 4096])
        batch = {
            "intervene": torch.sigmoid(outputs["intervene_logit"]).cpu().tolist(),
            "gain": torch.sigmoid(outputs["gain_logit"]).cpu().tolist(),
            "risk": torch.sigmoid(outputs["risk_logit"]).cpu().tolist(),
            "good_risk": torch.sigmoid(outputs["good_risk_logit"]).cpu().tolist(),
            "utility": outputs["utility"].cpu().tolist(),
        }
        for idx in range(len(batch["intervene"])):
            row = dict(rows[st + idx])
            row.update(
                {
                    "pred_intervene": float(batch["intervene"][idx]),
                    "pred_gain": float(batch["gain"][idx]),
                    "pred_risk": float(batch["risk"][idx]),
                    "pred_good_risk": float(batch["good_risk"][idx]),
                    "pred_utility": float(batch["utility"][idx]),
                }
            )
            predictions.append(row)
    return predictions


def gate_score(pred, params):
    return (
        float(pred.get("pred_intervene", 0.0))
        + params["gain_alpha"] * float(pred.get("pred_gain", 0.0))
        - params["risk_alpha"] * float(pred.get("pred_risk", 0.0))
        - params["good_risk_alpha"] * float(pred.get("pred_good_risk", 0.0))
        + params["utility_alpha"] * float(pred.get("pred_utility", 0.0))
    )


def apply_gate(items, predictions, candidate_records, ms_records, params):
    pred_by_qid = {row["qid"]: row for row in predictions}
    candidates = []
    for item in items:
        qid = item["qid"]
        pred = pred_by_qid.get(qid)
        candidate = candidate_records.get(qid)
        if not pred or not candidate or not candidate.get("changed_from_ms"):
            continue
        if (
            pred["pred_intervene"] >= params["min_intervene"]
            and pred["pred_gain"] >= params["min_gain"]
            and pred["pred_risk"] <= params["max_risk"]
            and pred["pred_good_risk"] <= params["max_good_risk"]
            and pred["pred_utility"] >= params["min_utility"]
        ):
            candidates.append((qid, gate_score(pred, params)))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    budget = int(math.ceil(float(params["budget"]) * len(items)))
    keep = set(qid for qid, _ in candidates[:budget])

    windows_by_qid = {}
    rows = []
    for item in items:
        qid = item["qid"]
        use_candidate = qid in keep
        chosen = candidate_records[qid] if use_candidate else ms_records[qid]
        pred = pred_by_qid.get(qid, {})
        windows_by_qid[qid] = chosen.get("windows", [])
        rows.append(
            {
                "qid": qid,
                "query": item.get("query", ""),
                "vid": item.get("vid", ""),
                "chosen_representation": chosen.get("chosen_representation", "ms_clap") if use_candidate else "ms_clap",
                "candidate_source": chosen.get("candidate_source", "") if use_candidate else "",
                "candidate_source_rank": chosen.get("candidate_source_rank", -1) if use_candidate else -1,
                "changed_from_ms": use_candidate,
                "pred_intervene": float(pred.get("pred_intervene", 0.0)),
                "pred_gain": float(pred.get("pred_gain", 0.0)),
                "pred_risk": float(pred.get("pred_risk", 0.0)),
                "pred_good_risk": float(pred.get("pred_good_risk", 0.0)),
                "pred_utility": float(pred.get("pred_utility", 0.0)),
                "ms_top1_iou": float(ms_records[qid].get("top1_iou", 0.0)),
                "ms_category": ms_records[qid].get("category", ""),
                "adapter_category": candidate_records[qid].get("adapter_category", ""),
                "windows": chosen.get("windows", []),
            }
        )
    metrics, case_rows = evaluate_predictions(items, windows_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, rows


def tune_gate(items, predictions, candidate_records, ms_records, objective):
    if objective == "balanced":
        grid = {
            "min_intervene": [0.45, 0.55, 0.65],
            "min_gain": [0.45, 0.60],
            "max_risk": [0.35, 0.50],
            "max_good_risk": [0.15, 0.25],
            "min_utility": [-0.10, 0.05],
            "budget": [0.10, 0.20, 0.30],
        }
    elif objective == "precision":
        grid = {
            "min_intervene": [0.60, 0.70, 0.80],
            "min_gain": [0.60, 0.75],
            "max_risk": [0.20, 0.35],
            "max_good_risk": [0.06, 0.12],
            "min_utility": [0.0, 0.10],
            "budget": [0.03, 0.05, 0.10],
        }
    else:
        raise ValueError(f"unknown objective: {objective}")

    best = None
    for min_intervene in grid["min_intervene"]:
        for min_gain in grid["min_gain"]:
            for max_risk in grid["max_risk"]:
                for max_good_risk in grid["max_good_risk"]:
                    for min_utility in grid["min_utility"]:
                        for budget in grid["budget"]:
                            params = {
                                "min_intervene": min_intervene,
                                "min_gain": min_gain,
                                "max_risk": max_risk,
                                "max_good_risk": max_good_risk,
                                "min_utility": min_utility,
                                "gain_alpha": 0.8,
                                "risk_alpha": 1.0,
                                "good_risk_alpha": 0.8,
                                "utility_alpha": 0.4,
                                "budget": budget,
                            }
                            metrics, rows = apply_gate(items, predictions, candidate_records, ms_records, params)
                            stats = summarize_interventions(rows)
                            if objective == "balanced":
                                key = (
                                    metrics["R1@0.7"],
                                    metrics["R1@0.5"],
                                    metrics["top1_iou"],
                                    -stats["good_regressed"],
                                    -stats["regressed"],
                                    stats["intervention_precision"],
                                    -stats["changed"],
                                )
                            else:
                                if stats["changed"] < 10:
                                    continue
                                key = (
                                    -stats["good_regressed"],
                                    -stats["regressed"],
                                    stats["intervention_precision"],
                                    metrics["R1@0.7"],
                                    metrics["top1_iou"],
                                    -stats["changed"],
                                )
                            if best is None or key > best["key"]:
                                best = {"params": params, "metrics": metrics, "rows": rows, "summary": stats, "key": key}
    return best


def metric_average(seed_results, system_name):
    values = [result["metrics"][system_name]["R1@0.7"] for result in seed_results]
    return {
        "mean_R1@0.7": mean(values),
        "std_R1@0.7": std(values),
        "values_R1@0.7": values,
    }


def write_html(output_dir, summary):
    metric_rows = []
    for name, metrics in summary["metrics"].items():
        if not metrics:
            continue
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
    seed_rows = []
    for result in summary["seed_results"]:
        seed_rows.append(
            [
                str(result["seed"]),
                fmt_pct(result["metrics"]["default_anchored_balanced_gate"]["R1@0.7"]),
                fmt_pct(result["metrics"]["default_anchored_precision_gate"]["R1@0.7"]),
                str(result["interventions"]["default_anchored_balanced_gate"]["changed"]),
                str(result["interventions"]["default_anchored_balanced_gate"]["regressed"]),
                str(result["interventions"]["default_anchored_balanced_gate"]["good_regressed"]),
            ]
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Default-Anchored Intervention Gate V3</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #7c3aed; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eee8fb; }}
  </style>
</head>
<body>
  <h1>Default-Anchored Intervention Gate V3</h1>
  <div class="note">
    V1 candidate scorer is frozen. This gate only decides whether its top merged candidate should replace the MS-CLAP default.
  </div>
  <h2>Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Seed Results</h2>
  {table(["Seed", "Balanced R1@0.7", "Precision R1@0.7", "Changed", "Regressed", "Good Regressed"], seed_rows)}
</body>
</html>"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_seeds(text):
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ms_evidence_path", required=True)
    parser.add_argument("--val_ms_evidence_path", required=True)
    parser.add_argument("--train_adapter_evidence_path", required=True)
    parser.add_argument("--val_adapter_evidence_path", required=True)
    parser.add_argument("--ms_decoder_ckpt", required=True)
    parser.add_argument("--adapter_decoder_ckpt", required=True)
    parser.add_argument("--fusion_scorer_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_v1_summary_path", default="")
    parser.add_argument("--reference_v2_summary_path", default="")
    parser.add_argument("--reference_gate_summary_path", default="")
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--intervene_loss_weight", type=float, default=1.0)
    parser.add_argument("--gain_loss_weight", type=float, default=0.8)
    parser.add_argument("--risk_loss_weight", type=float, default=1.1)
    parser.add_argument("--good_risk_loss_weight", type=float, default=1.6)
    parser.add_argument("--utility_loss_weight", type=float, default=0.5)
    parser.add_argument("--topn_per_source", type=int, default=2)
    parser.add_argument("--source_quotas", default="")
    parser.add_argument("--feature_version", choices=["basic", "shape_v2"], default="shape_v2")
    return parser.parse_args()


def load_reference_metrics(args):
    metrics = {}
    if args.reference_v1_summary_path and os.path.exists(args.reference_v1_summary_path):
        with open(args.reference_v1_summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name in ["candidate_level_fusion_v1", "candidate_level_balanced_gate_v1", "merged_candidate_oracle", "merged_candidate_oracle_20pct_budget"]:
            if data.get("metrics", {}).get(name):
                metrics[f"{name}_ref"] = data["metrics"][name]
    if args.reference_v2_summary_path and os.path.exists(args.reference_v2_summary_path):
        with open(args.reference_v2_summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name in ["candidate_risk_reranker_v2", "candidate_risk_balanced_gate_v2"]:
            if data.get("metrics", {}).get(name):
                metrics[f"{name}_ref"] = data["metrics"][name]
    if args.reference_gate_summary_path and os.path.exists(args.reference_gate_summary_path):
        with open(args.reference_gate_summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        src = data.get("metrics", {})
        if src.get("risk_aware_balanced_gate"):
            metrics["representation_gate_v2_ref"] = src["risk_aware_balanced_gate"]
        if src.get("oracle_ms_or_adapter"):
            metrics["whole_prediction_oracle_ref"] = src["oracle_ms_or_adapter"]
    return metrics


def run_seed(seed, args, train_dataset, val_dataset, train_rows, val_rows, train_candidate_records, val_candidate_records, train_ms_items, val_ms_items, train_ms_records, val_ms_records, device):
    random.seed(seed)
    torch.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = DefaultAnchoredGate(train_dataset.features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    losses = {
        "intervene": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.intervene).to(device)),
        "gain": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.gain).to(device)),
        "risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.risk).to(device)),
        "good_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.good_risk).to(device)),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        train_gate(model, train_loader, optimizer, losses, device, args)

    train_preds = predict_gate(model, train_dataset, train_rows, device)
    val_preds = predict_gate(model, val_dataset, val_rows, device)
    balanced = tune_gate(train_ms_items, train_preds, train_candidate_records, train_ms_records, objective="balanced")
    precision = tune_gate(train_ms_items, train_preds, train_candidate_records, train_ms_records, objective="precision") or balanced
    balanced_metrics, balanced_rows = apply_gate(val_ms_items, val_preds, val_candidate_records, val_ms_records, balanced["params"])
    precision_metrics, precision_rows = apply_gate(val_ms_items, val_preds, val_candidate_records, val_ms_records, precision["params"])
    return {
        "seed": seed,
        "balanced_params": balanced["params"],
        "precision_params": precision["params"],
        "metrics": {
            "default_anchored_balanced_gate": balanced_metrics,
            "default_anchored_precision_gate": precision_metrics,
        },
        "interventions": {
            "default_anchored_balanced_gate": summarize_interventions(balanced_rows),
            "default_anchored_precision_gate": summarize_interventions(precision_rows),
        },
        "mode_counts": {
            "default_anchored_balanced_gate": mode_counts(balanced_rows),
            "default_anchored_precision_gate": mode_counts(precision_rows),
        },
        "candidate_source_counts": {
            "default_anchored_balanced_gate": candidate_source_counts(balanced_rows),
            "default_anchored_precision_gate": candidate_source_counts(precision_rows),
        },
        "rows": {
            "balanced": balanced_rows,
            "precision": precision_rows,
        },
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_quotas = parse_source_quotas(args.source_quotas)

    train_ms_items = load_items(args.train_ms_evidence_path)
    val_ms_items = load_items(args.val_ms_evidence_path)
    train_adapter_items = load_items(args.train_adapter_evidence_path)
    val_adapter_items = load_items(args.val_adapter_evidence_path)
    ms_decoder = load_decoder_checkpoint(args.ms_decoder_ckpt, device)
    adapter_decoder = load_decoder_checkpoint(args.adapter_decoder_ckpt, device)
    fusion_scorer = load_fusion_scorer(args.fusion_scorer_ckpt, device)

    print("applying MS-CLAP decoder to train...")
    train_ms_records, train_ms_metrics = apply_decoder(train_ms_items, ms_decoder, device, "ms_clap")
    print("applying adapter decoder to train...")
    train_adapter_records, train_adapter_metrics = apply_decoder(train_adapter_items, adapter_decoder, device, "adapter")
    print("applying MS-CLAP decoder to val...")
    val_ms_records, val_ms_metrics = apply_decoder(val_ms_items, ms_decoder, device, "ms_clap")
    print("applying adapter decoder to val...")
    val_adapter_records, val_adapter_metrics = apply_decoder(val_adapter_items, adapter_decoder, device, "adapter")

    print("building merged train candidates...")
    train_candidate_rows = build_fusion_rows(
        train_ms_items,
        train_adapter_items,
        train_ms_records,
        train_adapter_records,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    print("building merged val candidates...")
    val_candidate_rows = build_fusion_rows(
        val_ms_items,
        val_adapter_items,
        val_ms_records,
        val_adapter_records,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    print("scoring candidates with frozen V1 scorer...")
    train_candidate_metrics, train_candidate_case_rows = apply_fusion_scorer(
        train_ms_items,
        train_candidate_rows,
        train_ms_records,
        train_adapter_records,
        fusion_scorer,
        device,
    )
    val_candidate_metrics, val_candidate_case_rows = apply_fusion_scorer(
        val_ms_items,
        val_candidate_rows,
        val_ms_records,
        val_adapter_records,
        fusion_scorer,
        device,
    )
    train_candidate_records = {row["qid"]: row for row in train_candidate_case_rows}
    val_candidate_records = {row["qid"]: row for row in val_candidate_case_rows}
    train_gate_rows = build_gate_rows(train_ms_items, train_candidate_records, train_ms_records)
    val_gate_rows = build_gate_rows(val_ms_items, val_candidate_records, val_ms_records)
    train_dataset = GateDataset(train_gate_rows)
    val_dataset = GateDataset(val_gate_rows, feature_mean=train_dataset.feature_mean, feature_std=train_dataset.feature_std)
    print("gate feature dim:", int(train_dataset.features.shape[1]))
    print("gate labels:", {
        "train_intervene": int(train_dataset.intervene.sum().item()),
        "train_gain": int(train_dataset.gain.sum().item()),
        "train_risk": int(train_dataset.risk.sum().item()),
        "train_good_risk": int(train_dataset.good_risk.sum().item()),
        "val_intervene": int(val_dataset.intervene.sum().item()),
        "val_gain": int(val_dataset.gain.sum().item()),
        "val_risk": int(val_dataset.risk.sum().item()),
        "val_good_risk": int(val_dataset.good_risk.sum().item()),
    })

    ms_oracle_metrics, ms_oracle_rows = oracle_from_rows(val_ms_items, val_candidate_rows, val_ms_records, val_adapter_records, representation="ms_clap")
    adapter_oracle_metrics, adapter_oracle_rows = oracle_from_rows(
        val_ms_items,
        val_candidate_rows,
        val_ms_records,
        val_adapter_records,
        representation="adapter",
    )
    merged_oracle_metrics, merged_oracle_rows = oracle_from_rows(val_ms_items, val_candidate_rows, val_ms_records, val_adapter_records)

    seed_results = []
    for seed in parse_seeds(args.seeds):
        print("running seed", seed)
        result = run_seed(
            seed,
            args,
            train_dataset,
            val_dataset,
            train_gate_rows,
            val_gate_rows,
            train_candidate_records,
            val_candidate_records,
            train_ms_items,
            val_ms_items,
            train_ms_records,
            val_ms_records,
            device,
        )
        seed_results.append(result)
        print(seed, "balanced R1@0.7", f"{100 * result['metrics']['default_anchored_balanced_gate']['R1@0.7']:.2f}")

    best_seed_result = max(seed_results, key=lambda item: item["metrics"]["default_anchored_balanced_gate"]["R1@0.7"])
    metrics = {
        "ms_clap_shape_v2_top2": val_ms_metrics,
        "adapter_shape_v2_top2": val_adapter_metrics,
        "frozen_candidate_fusion_v1": val_candidate_metrics,
        "default_anchored_balanced_gate_best_seed": best_seed_result["metrics"]["default_anchored_balanced_gate"],
        "default_anchored_precision_gate_best_seed": best_seed_result["metrics"]["default_anchored_precision_gate"],
        "ms_candidate_oracle": ms_oracle_metrics,
        "adapter_candidate_oracle": adapter_oracle_metrics,
        "merged_candidate_oracle": merged_oracle_metrics,
    }
    metrics.update(load_reference_metrics(args))
    summary = {
        "seeds": parse_seeds(args.seeds),
        "best_seed": best_seed_result["seed"],
        "train_candidates": len(train_candidate_rows),
        "val_candidates": len(val_candidate_rows),
        "gate_feature_dim": int(train_dataset.features.shape[1]),
        "label_counts": {
            "train_intervene": int(train_dataset.intervene.sum().item()),
            "train_gain": int(train_dataset.gain.sum().item()),
            "train_risk": int(train_dataset.risk.sum().item()),
            "train_good_risk": int(train_dataset.good_risk.sum().item()),
            "val_intervene": int(val_dataset.intervene.sum().item()),
            "val_gain": int(val_dataset.gain.sum().item()),
            "val_risk": int(val_dataset.risk.sum().item()),
            "val_good_risk": int(val_dataset.good_risk.sum().item()),
        },
        "metrics": metrics,
        "seed_averages": {
            "default_anchored_balanced_gate": metric_average(seed_results, "default_anchored_balanced_gate"),
            "default_anchored_precision_gate": metric_average(seed_results, "default_anchored_precision_gate"),
        },
        "interventions": {
            "default_anchored_balanced_gate_best_seed": best_seed_result["interventions"]["default_anchored_balanced_gate"],
            "default_anchored_precision_gate_best_seed": best_seed_result["interventions"]["default_anchored_precision_gate"],
            "frozen_candidate_fusion_v1": summarize_interventions(val_candidate_case_rows),
            "merged_candidate_oracle": summarize_interventions(merged_oracle_rows),
        },
        "seed_results": [
            {
                "seed": result["seed"],
                "balanced_params": result["balanced_params"],
                "precision_params": result["precision_params"],
                "metrics": result["metrics"],
                "interventions": result["interventions"],
                "mode_counts": result["mode_counts"],
                "candidate_source_counts": result["candidate_source_counts"],
            }
            for result in seed_results
        ],
        "args": vars(args),
    }
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(best_seed_result["rows"]["balanced"], os.path.join(args.output_dir, "best_seed_balanced_case_rows.json"))
    save_json(best_seed_result["rows"]["precision"], os.path.join(args.output_dir, "best_seed_precision_case_rows.json"))
    save_json(val_candidate_case_rows, os.path.join(args.output_dir, "frozen_candidate_fusion_case_rows.json"))
    write_html(args.output_dir, summary)
    print(json.dumps(summary["seed_averages"], indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
