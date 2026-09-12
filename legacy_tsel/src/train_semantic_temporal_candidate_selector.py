import argparse
import json
import math
import os
import random
from collections import defaultdict
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
    summarize_interventions,
)
from train_default_anchored_intervention_gate import apply_fusion_scorer, load_fusion_scorer, top_candidate_features
from train_learned_evidence_decoder import load_items, parse_source_quotas
from train_representation_fusion_selector import quality_key
from train_two_mode_decoder_selector import apply_decoder, evaluate_predictions, fmt_float, fmt_pct, load_decoder_checkpoint, table


POSITIVE_TEMPORAL_CATEGORIES = {"boundary_error", "candidate_exists", "evidence_good_decode_bad"}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def mean(values):
    values = list(values)
    return sum(values) / max(len(values), 1)


def std(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def parse_seeds(text):
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def pos_weight(labels, max_value=20.0):
    pos = float(labels.sum().item())
    neg = float(labels.numel() - pos)
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(min(max_value, neg / pos))


def candidate_quality_key(record):
    value = float(record.get("top1_iou", 0.0))
    return (value >= 0.7, value >= 0.5, value)


def role_labels(candidate_record, ms_record):
    ms_cat = ms_record.get("category", "")
    cand_cat = candidate_record.get("category", "")
    ms_key = quality_key(ms_record)
    cand_key = candidate_quality_key(candidate_record)
    ms_iou = float(ms_record.get("top1_iou", 0.0))
    cand_iou = float(candidate_record.get("top1_iou", 0.0))

    semantic_recovery = ms_cat == "semantic_miss" and cand_cat != "semantic_miss"
    temporal_positive = (
        cand_key > ms_key
        and ms_cat in POSITIVE_TEMPORAL_CATEGORIES
    ) or (ms_cat != "good" and cand_cat == "good")
    anchor_risk = ms_cat == "good" and (cand_cat != "good" or cand_iou < ms_iou)
    semantic_risk = ms_cat != "semantic_miss" and cand_cat == "semantic_miss"
    temporal_risk = cand_key < ms_key and ms_cat in POSITIVE_TEMPORAL_CATEGORIES and not semantic_risk
    intervene = semantic_recovery or temporal_positive or (cand_key > ms_key)

    utility = cand_iou - ms_iou
    utility += 0.55 * (float(cand_iou >= 0.7) - float(ms_iou >= 0.7))
    utility += 0.25 * (float(cand_iou >= 0.5) - float(ms_iou >= 0.5))
    if semantic_recovery:
        utility += 0.25
    if temporal_positive:
        utility += 0.20
    if anchor_risk:
        utility -= 0.50
    if semantic_risk:
        utility -= 0.25
    if temporal_risk:
        utility -= 0.20
    utility = max(-1.5, min(1.5, float(utility)))

    return {
        "target_semantic_recovery": float(semantic_recovery),
        "target_temporal_positive": float(temporal_positive),
        "target_anchor_risk": float(anchor_risk),
        "target_semantic_risk": float(semantic_risk),
        "target_temporal_risk": float(temporal_risk),
        "target_intervene": float(intervene and utility > 0.0),
        "target_utility": utility,
    }


def build_selector_rows(items, candidate_records, ms_records):
    rows = []
    for item in items:
        qid = item["qid"]
        if qid not in candidate_records or qid not in ms_records:
            continue
        candidate = candidate_records[qid]
        ms_record = ms_records[qid]
        labels = role_labels(candidate, ms_record)
        rows.append(
            {
                "qid": qid,
                "features": top_candidate_features(candidate, ms_record),
                **labels,
            }
        )
    return rows


class SelectorDataset(Dataset):
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
        self.semantic_recovery = torch.tensor([row["target_semantic_recovery"] for row in rows], dtype=torch.float32)
        self.temporal_positive = torch.tensor([row["target_temporal_positive"] for row in rows], dtype=torch.float32)
        self.anchor_risk = torch.tensor([row["target_anchor_risk"] for row in rows], dtype=torch.float32)
        self.semantic_risk = torch.tensor([row["target_semantic_risk"] for row in rows], dtype=torch.float32)
        self.temporal_risk = torch.tensor([row["target_temporal_risk"] for row in rows], dtype=torch.float32)
        self.intervene = torch.tensor([row["target_intervene"] for row in rows], dtype=torch.float32)
        self.utility = torch.tensor([row["target_utility"] for row in rows], dtype=torch.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.semantic_recovery[idx],
            self.temporal_positive[idx],
            self.anchor_risk[idx],
            self.semantic_risk[idx],
            self.temporal_risk[idx],
            self.intervene[idx],
            self.utility[idx],
        )


class SemanticTemporalSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.12):
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
        self.semantic_recovery_head = nn.Linear(hidden_dim, 1)
        self.temporal_positive_head = nn.Linear(hidden_dim, 1)
        self.anchor_risk_head = nn.Linear(hidden_dim, 1)
        self.semantic_risk_head = nn.Linear(hidden_dim, 1)
        self.temporal_risk_head = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        h = self.shared(features)
        return {
            "intervene_logit": self.intervene_head(h).squeeze(-1),
            "semantic_recovery_logit": self.semantic_recovery_head(h).squeeze(-1),
            "temporal_positive_logit": self.temporal_positive_head(h).squeeze(-1),
            "anchor_risk_logit": self.anchor_risk_head(h).squeeze(-1),
            "semantic_risk_logit": self.semantic_risk_head(h).squeeze(-1),
            "temporal_risk_logit": self.temporal_risk_head(h).squeeze(-1),
            "utility": 1.5 * torch.tanh(self.utility_head(h).squeeze(-1)),
        }


def train_epoch(model, loader, optimizer, losses, device, args):
    model.train()
    totals = defaultdict(float)
    steps = 0
    for batch in loader:
        (
            features,
            semantic_y,
            temporal_y,
            anchor_risk_y,
            semantic_risk_y,
            temporal_risk_y,
            intervene_y,
            utility_y,
        ) = batch
        features = features.to(device)
        semantic_y = semantic_y.to(device)
        temporal_y = temporal_y.to(device)
        anchor_risk_y = anchor_risk_y.to(device)
        semantic_risk_y = semantic_risk_y.to(device)
        temporal_risk_y = temporal_risk_y.to(device)
        intervene_y = intervene_y.to(device)
        utility_y = utility_y.to(device)

        out = model(features)
        losses_by_name = {
            "intervene": losses["intervene"](out["intervene_logit"], intervene_y),
            "semantic": losses["semantic"](out["semantic_recovery_logit"], semantic_y),
            "temporal": losses["temporal"](out["temporal_positive_logit"], temporal_y),
            "anchor_risk": losses["anchor_risk"](out["anchor_risk_logit"], anchor_risk_y),
            "semantic_risk": losses["semantic_risk"](out["semantic_risk_logit"], semantic_risk_y),
            "temporal_risk": losses["temporal_risk"](out["temporal_risk_logit"], temporal_risk_y),
            "utility": F.mse_loss(out["utility"], utility_y),
        }
        loss = (
            args.intervene_loss_weight * losses_by_name["intervene"]
            + args.semantic_loss_weight * losses_by_name["semantic"]
            + args.temporal_loss_weight * losses_by_name["temporal"]
            + args.anchor_risk_loss_weight * losses_by_name["anchor_risk"]
            + args.semantic_risk_loss_weight * losses_by_name["semantic_risk"]
            + args.temporal_risk_loss_weight * losses_by_name["temporal_risk"]
            + args.utility_loss_weight * losses_by_name["utility"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.item())
        for name, value in losses_by_name.items():
            totals[name] += float(value.item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def predict(model, dataset, rows, device):
    model.eval()
    features = dataset.features.to(device)
    predictions = []
    for st in range(0, len(features), 4096):
        out = model(features[st : st + 4096])
        batch = {
            "pred_intervene": torch.sigmoid(out["intervene_logit"]).cpu().tolist(),
            "pred_semantic_recovery": torch.sigmoid(out["semantic_recovery_logit"]).cpu().tolist(),
            "pred_temporal_positive": torch.sigmoid(out["temporal_positive_logit"]).cpu().tolist(),
            "pred_anchor_risk": torch.sigmoid(out["anchor_risk_logit"]).cpu().tolist(),
            "pred_semantic_risk": torch.sigmoid(out["semantic_risk_logit"]).cpu().tolist(),
            "pred_temporal_risk": torch.sigmoid(out["temporal_risk_logit"]).cpu().tolist(),
            "pred_utility": out["utility"].cpu().tolist(),
        }
        for idx in range(len(batch["pred_intervene"])):
            row = dict(rows[st + idx])
            for name, values in batch.items():
                row[name] = float(values[idx])
            predictions.append(row)
    return predictions


def selector_score(pred, params):
    return (
        float(pred.get("pred_intervene", 0.0))
        + params["semantic_alpha"] * float(pred.get("pred_semantic_recovery", 0.0))
        + params["temporal_alpha"] * float(pred.get("pred_temporal_positive", 0.0))
        - params["anchor_risk_alpha"] * float(pred.get("pred_anchor_risk", 0.0))
        - params["semantic_risk_alpha"] * float(pred.get("pred_semantic_risk", 0.0))
        - params["temporal_risk_alpha"] * float(pred.get("pred_temporal_risk", 0.0))
        + params["utility_alpha"] * float(pred.get("pred_utility", 0.0))
    )


def apply_selector(items, predictions, candidate_records, ms_records, params):
    pred_by_qid = {row["qid"]: row for row in predictions}
    candidates = []
    for item in items:
        qid = item["qid"]
        pred = pred_by_qid.get(qid)
        candidate = candidate_records.get(qid)
        if not pred or not candidate or not candidate.get("changed_from_ms"):
            continue
        positive_signal = max(float(pred["pred_semantic_recovery"]), float(pred["pred_temporal_positive"]))
        if (
            float(pred["pred_intervene"]) >= params["min_intervene"]
            and positive_signal >= params["min_positive"]
            and float(pred["pred_anchor_risk"]) <= params["max_anchor_risk"]
            and float(pred["pred_semantic_risk"]) <= params["max_semantic_risk"]
            and float(pred["pred_temporal_risk"]) <= params["max_temporal_risk"]
            and float(pred["pred_utility"]) >= params["min_utility"]
        ):
            candidates.append((qid, selector_score(pred, params)))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    keep = set(qid for qid, _ in candidates[: int(math.ceil(float(params["budget"]) * len(items)))])

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
                "ms_top1_iou": float(ms_records[qid].get("top1_iou", 0.0)),
                "ms_category": ms_records[qid].get("category", ""),
                "adapter_category": candidate_records[qid].get("adapter_category", ""),
                "pred_intervene": float(pred.get("pred_intervene", 0.0)),
                "pred_semantic_recovery": float(pred.get("pred_semantic_recovery", 0.0)),
                "pred_temporal_positive": float(pred.get("pred_temporal_positive", 0.0)),
                "pred_anchor_risk": float(pred.get("pred_anchor_risk", 0.0)),
                "pred_semantic_risk": float(pred.get("pred_semantic_risk", 0.0)),
                "pred_temporal_risk": float(pred.get("pred_temporal_risk", 0.0)),
                "pred_utility": float(pred.get("pred_utility", 0.0)),
                "windows": chosen.get("windows", []),
            }
        )
    metrics, case_rows = evaluate_predictions(items, windows_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, rows


def tune_selector(items, predictions, candidate_records, ms_records, objective):
    if objective == "balanced":
        grid = {
            "min_intervene": [0.45, 0.60],
            "min_positive": [0.40, 0.55],
            "max_anchor_risk": [0.25, 0.50],
            "max_semantic_risk": [0.35, 0.55],
            "max_temporal_risk": [0.35, 0.55],
            "min_utility": [-0.10, 0.05],
            "budget": [0.20, 0.50, 0.80],
        }
    elif objective == "precision":
        grid = {
            "min_intervene": [0.60, 0.75],
            "min_positive": [0.55, 0.70],
            "max_anchor_risk": [0.12, 0.25],
            "max_semantic_risk": [0.25, 0.40],
            "max_temporal_risk": [0.25, 0.40],
            "min_utility": [0.0, 0.15],
            "budget": [0.05, 0.10],
        }
    else:
        raise ValueError(f"unknown objective: {objective}")

    best = None
    for min_intervene in grid["min_intervene"]:
        for min_positive in grid["min_positive"]:
            for max_anchor_risk in grid["max_anchor_risk"]:
                for max_semantic_risk in grid["max_semantic_risk"]:
                    for max_temporal_risk in grid["max_temporal_risk"]:
                        for min_utility in grid["min_utility"]:
                            for budget in grid["budget"]:
                                params = {
                                    "min_intervene": min_intervene,
                                    "min_positive": min_positive,
                                    "max_anchor_risk": max_anchor_risk,
                                    "max_semantic_risk": max_semantic_risk,
                                    "max_temporal_risk": max_temporal_risk,
                                    "min_utility": min_utility,
                                    "semantic_alpha": 0.8,
                                    "temporal_alpha": 0.8,
                                    "anchor_risk_alpha": 1.0,
                                    "semantic_risk_alpha": 0.7,
                                    "temporal_risk_alpha": 0.7,
                                    "utility_alpha": 0.4,
                                    "budget": budget,
                                }
                                metrics, rows = apply_selector(items, predictions, candidate_records, ms_records, params)
                                stats = summarize_interventions(rows)
                                if objective == "balanced":
                                    key = (
                                        metrics["R1@0.7"],
                                        metrics["R1@0.5"],
                                        metrics["top1_iou"],
                                        stats["recovered_semantic_miss"],
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
                                        stats["recovered_semantic_miss"],
                                        -stats["changed"],
                                    )
                                if best is None or key > best["key"]:
                                    best = {"params": params, "metrics": metrics, "rows": rows, "summary": stats, "key": key}
    return best


def metric_average(seed_results, name):
    keys = ["R1@0.5", "R1@0.7", "top1_iou", "top5_iou"]
    out = {}
    for key in keys:
        values = [result["metrics"][name][key] for result in seed_results]
        out[f"mean_{key}"] = mean(values)
        out[f"std_{key}"] = std(values)
        out[f"values_{key}"] = values
    return out


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
    seed_rows = []
    for result in summary["seed_results"]:
        seed_rows.append(
            [
                str(result["seed"]),
                fmt_pct(result["metrics"]["semantic_temporal_balanced_selector"]["R1@0.7"]),
                fmt_pct(result["metrics"]["semantic_temporal_precision_selector"]["R1@0.7"]),
                str(result["interventions"]["semantic_temporal_balanced_selector"]["changed"]),
                str(result["interventions"]["semantic_temporal_balanced_selector"]["regressed"]),
                str(result["interventions"]["semantic_temporal_balanced_selector"]["good_regressed"]),
            ]
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Semantic-Temporal Candidate Selector V1</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #0f766e; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #e8f3f1; }}
  </style>
</head>
<body>
  <h1>Semantic-Temporal Candidate Selector V1</h1>
  <div class="note">
    This selector decomposes interventions into semantic recovery, temporal correction, anchor risk, semantic risk, and temporal risk.
  </div>
  <h2>Validation Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Intervention Quality</h2>
  {table(["System", "Changed", "Improved", "Regressed", "Recovered Semantic Miss", "Good Regressed", "Precision"], intervention_rows)}
  <h2>Seed Results</h2>
  {table(["Seed", "Balanced R1@0.7", "Precision R1@0.7", "Changed", "Regressed", "Good Regressed"], seed_rows)}
</body>
</html>"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def run_seed(seed, args, train_dataset, val_dataset, train_rows, val_rows, train_candidate_records, val_candidate_records, train_items, val_items, train_ms_records, val_ms_records, device):
    random.seed(seed)
    torch.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SemanticTemporalSelector(train_dataset.features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    losses = {
        "intervene": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.intervene).to(device)),
        "semantic": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.semantic_recovery).to(device)),
        "temporal": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.temporal_positive).to(device)),
        "anchor_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.anchor_risk).to(device)),
        "semantic_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.semantic_risk).to(device)),
        "temporal_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.temporal_risk).to(device)),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        train_epoch(model, train_loader, optimizer, losses, device, args)
    train_preds = predict(model, train_dataset, train_rows, device)
    val_preds = predict(model, val_dataset, val_rows, device)
    balanced = tune_selector(train_items, train_preds, train_candidate_records, train_ms_records, objective="balanced")
    precision = tune_selector(train_items, train_preds, train_candidate_records, train_ms_records, objective="precision") or balanced
    balanced_metrics, balanced_rows = apply_selector(val_items, val_preds, val_candidate_records, val_ms_records, balanced["params"])
    precision_metrics, precision_rows = apply_selector(val_items, val_preds, val_candidate_records, val_ms_records, precision["params"])

    ckpt_path = os.path.join(args.output_dir, f"selector_seed{seed}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": train_dataset.feature_mean,
            "feature_std": train_dataset.feature_std,
            "balanced_params": balanced["params"],
            "precision_params": precision["params"],
            "args": vars(args),
            "seed": seed,
            "metrics": {
                "balanced": balanced_metrics,
                "precision": precision_metrics,
            },
        },
        ckpt_path,
    )
    return {
        "seed": seed,
        "checkpoint": ckpt_path,
        "balanced_params": balanced["params"],
        "precision_params": precision["params"],
        "metrics": {
            "semantic_temporal_balanced_selector": balanced_metrics,
            "semantic_temporal_precision_selector": precision_metrics,
        },
        "interventions": {
            "semantic_temporal_balanced_selector": summarize_interventions(balanced_rows),
            "semantic_temporal_precision_selector": summarize_interventions(precision_rows),
        },
        "mode_counts": {
            "semantic_temporal_balanced_selector": mode_counts(balanced_rows),
            "semantic_temporal_precision_selector": mode_counts(precision_rows),
        },
        "candidate_source_counts": {
            "semantic_temporal_balanced_selector": candidate_source_counts(balanced_rows),
            "semantic_temporal_precision_selector": candidate_source_counts(precision_rows),
        },
        "rows": {
            "balanced": balanced_rows,
            "precision": precision_rows,
        },
    }


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
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--intervene_loss_weight", type=float, default=1.0)
    parser.add_argument("--semantic_loss_weight", type=float, default=1.1)
    parser.add_argument("--temporal_loss_weight", type=float, default=1.0)
    parser.add_argument("--anchor_risk_loss_weight", type=float, default=1.6)
    parser.add_argument("--semantic_risk_loss_weight", type=float, default=1.1)
    parser.add_argument("--temporal_risk_loss_weight", type=float, default=1.1)
    parser.add_argument("--utility_loss_weight", type=float, default=0.5)
    parser.add_argument("--topn_per_source", type=int, default=2)
    parser.add_argument("--source_quotas", default="")
    parser.add_argument("--feature_version", choices=["basic", "shape_v2"], default="shape_v2")
    return parser.parse_args()


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

    print("applying MS decoder...")
    train_ms_records, train_ms_metrics = apply_decoder(train_ms_items, ms_decoder, device, "ms_clap")
    val_ms_records, val_ms_metrics = apply_decoder(val_ms_items, ms_decoder, device, "ms_clap")
    print("applying adapter decoder...")
    train_adapter_records, train_adapter_metrics = apply_decoder(train_adapter_items, adapter_decoder, device, "adapter")
    val_adapter_records, val_adapter_metrics = apply_decoder(val_adapter_items, adapter_decoder, device, "adapter")

    print("building candidate rows...")
    train_candidate_rows = build_fusion_rows(
        train_ms_items,
        train_adapter_items,
        train_ms_records,
        train_adapter_records,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    val_candidate_rows = build_fusion_rows(
        val_ms_items,
        val_adapter_items,
        val_ms_records,
        val_adapter_records,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    print("scoring with frozen fusion scorer...")
    train_fusion_metrics, train_fusion_rows = apply_fusion_scorer(
        train_ms_items,
        train_candidate_rows,
        train_ms_records,
        train_adapter_records,
        fusion_scorer,
        device,
    )
    val_fusion_metrics, val_fusion_rows = apply_fusion_scorer(
        val_ms_items,
        val_candidate_rows,
        val_ms_records,
        val_adapter_records,
        fusion_scorer,
        device,
    )
    train_candidate_records = {row["qid"]: row for row in train_fusion_rows}
    val_candidate_records = {row["qid"]: row for row in val_fusion_rows}
    train_rows = build_selector_rows(train_ms_items, train_candidate_records, train_ms_records)
    val_rows = build_selector_rows(val_ms_items, val_candidate_records, val_ms_records)
    train_dataset = SelectorDataset(train_rows)
    val_dataset = SelectorDataset(val_rows, feature_mean=train_dataset.feature_mean, feature_std=train_dataset.feature_std)
    print("selector feature dim:", int(train_dataset.features.shape[1]))
    label_counts = {
        "train_semantic_recovery": int(train_dataset.semantic_recovery.sum().item()),
        "train_temporal_positive": int(train_dataset.temporal_positive.sum().item()),
        "train_anchor_risk": int(train_dataset.anchor_risk.sum().item()),
        "train_semantic_risk": int(train_dataset.semantic_risk.sum().item()),
        "train_temporal_risk": int(train_dataset.temporal_risk.sum().item()),
        "train_intervene": int(train_dataset.intervene.sum().item()),
        "val_semantic_recovery": int(val_dataset.semantic_recovery.sum().item()),
        "val_temporal_positive": int(val_dataset.temporal_positive.sum().item()),
        "val_anchor_risk": int(val_dataset.anchor_risk.sum().item()),
        "val_semantic_risk": int(val_dataset.semantic_risk.sum().item()),
        "val_temporal_risk": int(val_dataset.temporal_risk.sum().item()),
        "val_intervene": int(val_dataset.intervene.sum().item()),
    }
    print("label counts:", json.dumps(label_counts, indent=2))

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
            train_rows,
            val_rows,
            train_candidate_records,
            val_candidate_records,
            train_ms_items,
            val_ms_items,
            train_ms_records,
            val_ms_records,
            device,
        )
        seed_results.append(result)
        print(seed, "balanced R1@0.7", f"{100 * result['metrics']['semantic_temporal_balanced_selector']['R1@0.7']:.2f}")

    best_seed_result = max(seed_results, key=lambda item: item["metrics"]["semantic_temporal_balanced_selector"]["R1@0.7"])
    metrics = {
        "ms_clap_shape_v2_top2": val_ms_metrics,
        "adapter_shape_v2_top2": val_adapter_metrics,
        "frozen_candidate_fusion": val_fusion_metrics,
        "semantic_temporal_balanced_selector_best_seed": best_seed_result["metrics"]["semantic_temporal_balanced_selector"],
        "semantic_temporal_precision_selector_best_seed": best_seed_result["metrics"]["semantic_temporal_precision_selector"],
        "ms_candidate_oracle": ms_oracle_metrics,
        "adapter_candidate_oracle": adapter_oracle_metrics,
        "merged_candidate_oracle": merged_oracle_metrics,
    }
    interventions = {
        "frozen_candidate_fusion": summarize_interventions(val_fusion_rows),
        "semantic_temporal_balanced_selector_best_seed": best_seed_result["interventions"]["semantic_temporal_balanced_selector"],
        "semantic_temporal_precision_selector_best_seed": best_seed_result["interventions"]["semantic_temporal_precision_selector"],
        "merged_candidate_oracle": summarize_interventions(merged_oracle_rows),
    }
    summary = {
        "best_seed": best_seed_result["seed"],
        "best_seed_checkpoint": best_seed_result["checkpoint"],
        "seeds": parse_seeds(args.seeds),
        "train_candidates": len(train_candidate_rows),
        "val_candidates": len(val_candidate_rows),
        "selector_feature_dim": int(train_dataset.features.shape[1]),
        "label_counts": label_counts,
        "metrics": metrics,
        "interventions": interventions,
        "seed_averages": {
            "semantic_temporal_balanced_selector": metric_average(seed_results, "semantic_temporal_balanced_selector"),
            "semantic_temporal_precision_selector": metric_average(seed_results, "semantic_temporal_precision_selector"),
        },
        "seed_results": [
            {
                "seed": result["seed"],
                "checkpoint": result["checkpoint"],
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
    save_json(val_fusion_rows, os.path.join(args.output_dir, "frozen_candidate_fusion_case_rows.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in best_seed_result["rows"]["balanced"]],
        os.path.join(args.output_dir, "best_seed_balanced_predictions.jsonl"),
    )
    write_html(args.output_dir, summary)
    print(json.dumps(summary["seed_averages"], indent=2))
    print("saved", os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
