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
    apply_oracle_budget,
    build_fusion_rows,
    candidate_source_counts,
    mode_counts,
    oracle_from_rows,
    scored_rows_to_records,
    summarize_interventions,
    write_case_svgs,
)
from train_learned_evidence_decoder import load_items, model_selection_key, parse_source_quotas
from train_representation_fusion_selector import quality_key
from train_two_mode_decoder_selector import apply_decoder, evaluate_predictions, fmt_float, fmt_pct, load_decoder_checkpoint, safe_div, table, window_iou


def clamp(value, lo=-1.5, hi=1.5):
    return max(lo, min(hi, float(value)))


def candidate_key(row):
    iou = float(row.get("target_raw_iou", 0.0))
    return (iou >= 0.7, iou >= 0.5, iou)


def candidate_utility(row, ms_record):
    raw_iou = float(row.get("target_raw_iou", 0.0))
    ms_iou = float(ms_record.get("top1_iou", 0.0))
    utility = raw_iou - ms_iou
    utility += 0.50 * (float(raw_iou >= 0.7) - float(ms_iou >= 0.7))
    utility += 0.25 * (float(raw_iou >= 0.5) - float(ms_iou >= 0.5))
    if ms_record.get("category") == "semantic_miss" and raw_iou >= 0.5:
        utility += 0.20
    if ms_record.get("category") == "good" and raw_iou < 0.7:
        utility -= 0.40
    if ms_record.get("category") == "boundary_error" and raw_iou < 0.1:
        utility -= 0.15
    return clamp(utility)


def add_risk_labels(rows, ms_records):
    labeled = []
    for row in rows:
        item = dict(row)
        ms_record = ms_records[item["qid"]]
        ms_key = quality_key(ms_record)
        cand_key = candidate_key(item)
        raw_iou = float(item.get("target_raw_iou", 0.0))
        item["target_quality"] = float(item.get("target_iou", raw_iou))
        item["target_gain"] = 1.0 if cand_key > ms_key else 0.0
        item["target_risk"] = 1.0 if cand_key < ms_key else 0.0
        item["target_good_risk"] = 1.0 if ms_record.get("category") == "good" and raw_iou < 0.7 else 0.0
        item["target_semantic_recovery"] = 1.0 if ms_record.get("category") == "semantic_miss" and raw_iou >= 0.5 else 0.0
        item["target_utility"] = candidate_utility(item, ms_record)
        labeled.append(item)
    return labeled


class RiskCandidateDataset(Dataset):
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
        self.quality = torch.tensor([row["target_quality"] for row in rows], dtype=torch.float32)
        self.gain = torch.tensor([row["target_gain"] for row in rows], dtype=torch.float32)
        self.risk = torch.tensor([row["target_risk"] for row in rows], dtype=torch.float32)
        self.good_risk = torch.tensor([row["target_good_risk"] for row in rows], dtype=torch.float32)
        self.utility = torch.tensor([row["target_utility"] for row in rows], dtype=torch.float32)
        self.qids = [row["qid"] for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.quality[idx],
            self.gain[idx],
            self.risk[idx],
            self.good_risk[idx],
            self.utility[idx],
            self.qids[idx],
        )


class CandidateRiskCalibrator(nn.Module):
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
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.good_risk_head = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return {
            "quality_logit": self.quality_head(h).squeeze(-1),
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


def action_score_from_outputs(outputs, args):
    quality = torch.sigmoid(outputs["quality_logit"])
    gain = torch.sigmoid(outputs["gain_logit"])
    risk = torch.sigmoid(outputs["risk_logit"])
    good_risk = torch.sigmoid(outputs["good_risk_logit"])
    utility = outputs["utility"]
    return (
        args.quality_alpha * quality
        + args.gain_alpha * gain
        - args.risk_alpha * risk
        - args.good_risk_alpha * good_risk
        + args.utility_alpha * utility
    )


def pairwise_utility_loss(scores, targets, qids, margin=0.10, max_pairs=512):
    by_qid = defaultdict(list)
    for idx, qid in enumerate(qids):
        by_qid[qid].append(idx)
    losses = []
    device = scores.device
    for indices in by_qid.values():
        if len(indices) < 2:
            continue
        indices = indices[:80]
        pairs = []
        for i in indices:
            for j in indices:
                if targets[i] - targets[j] >= margin:
                    pairs.append((i, j))
        if len(pairs) > max_pairs:
            pairs = random.sample(pairs, max_pairs)
        for i, j in pairs:
            losses.append(F.relu(0.10 - (scores[i] - scores[j])))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def train_epoch(model, loader, optimizer, device, losses, args):
    model.train()
    totals = defaultdict(float)
    steps = 0
    for features, quality_y, gain_y, risk_y, good_risk_y, utility_y, qids in loader:
        features = features.to(device)
        quality_y = quality_y.to(device)
        gain_y = gain_y.to(device)
        risk_y = risk_y.to(device)
        good_risk_y = good_risk_y.to(device)
        utility_y = utility_y.to(device)
        outputs = model(features)
        quality_loss = F.mse_loss(torch.sigmoid(outputs["quality_logit"]), quality_y)
        gain_loss = losses["gain"](outputs["gain_logit"], gain_y)
        risk_loss = losses["risk"](outputs["risk_logit"], risk_y)
        good_risk_loss = losses["good_risk"](outputs["good_risk_logit"], good_risk_y)
        utility_loss = F.mse_loss(outputs["utility"], utility_y)
        action = action_score_from_outputs(outputs, args)
        rank_loss = pairwise_utility_loss(action, utility_y, qids) if args.lambda_pairwise > 0 else torch.zeros((), device=device)
        loss = (
            args.quality_loss_weight * quality_loss
            + args.gain_loss_weight * gain_loss
            + args.risk_loss_weight * risk_loss
            + args.good_risk_loss_weight * good_risk_loss
            + args.utility_loss_weight * utility_loss
            + args.lambda_pairwise * rank_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.item())
        totals["quality"] += float(quality_loss.item())
        totals["gain"] += float(gain_loss.item())
        totals["risk"] += float(risk_loss.item())
        totals["good_risk"] += float(good_risk_loss.item())
        totals["utility"] += float(utility_loss.item())
        totals["rank"] += float(rank_loss.item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def predict_rows(model, dataset, rows, device, args):
    model.eval()
    features = dataset.features.to(device)
    out = []
    batch_size = 4096
    for st in range(0, len(features), batch_size):
        outputs = model(features[st : st + batch_size])
        action = action_score_from_outputs(outputs, args)
        batch = {
            "quality": torch.sigmoid(outputs["quality_logit"]).cpu().tolist(),
            "gain": torch.sigmoid(outputs["gain_logit"]).cpu().tolist(),
            "risk": torch.sigmoid(outputs["risk_logit"]).cpu().tolist(),
            "good_risk": torch.sigmoid(outputs["good_risk_logit"]).cpu().tolist(),
            "utility": outputs["utility"].cpu().tolist(),
            "action": action.cpu().tolist(),
        }
        size = len(batch["action"])
        for idx in range(size):
            row = dict(rows[st + idx])
            row["pred_quality_head"] = float(batch["quality"][idx])
            row["pred_gain"] = float(batch["gain"][idx])
            row["pred_risk"] = float(batch["risk"][idx])
            row["pred_good_risk"] = float(batch["good_risk"][idx])
            row["pred_utility"] = float(batch["utility"][idx])
            row["pred_action"] = float(batch["action"][idx])
            row["pred_quality"] = float(batch["action"][idx])
            out.append(row)
    return out


def top_calibrated_row(record):
    rows = record.get("top_rows", [])
    return rows[0] if rows else {}


def apply_risk_gate(items, risk_rows, ms_records, params):
    risk_by_qid = {row["qid"]: row for row in risk_rows}
    candidates = []
    for item in items:
        qid = item["qid"]
        record = risk_by_qid.get(qid)
        if not record or not record.get("changed_from_ms"):
            continue
        top = top_calibrated_row(record)
        if (
            float(top.get("pred_gain", 0.0)) >= params["min_gain"]
            and float(top.get("pred_risk", 1.0)) <= params["max_risk"]
            and float(top.get("pred_good_risk", 1.0)) <= params["max_good_risk"]
            and float(top.get("pred_utility", -9.0)) >= params["min_utility"]
        ):
            score = (
                float(top.get("pred_gain", 0.0))
                - params["risk_alpha"] * float(top.get("pred_risk", 0.0))
                - params["good_risk_alpha"] * float(top.get("pred_good_risk", 0.0))
                + params["utility_alpha"] * float(top.get("pred_utility", 0.0))
                + params["quality_alpha"] * float(top.get("pred_quality_head", 0.0))
            )
            candidates.append((qid, score))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    budget = int(math.ceil(float(params["budget"]) * len(items)))
    keep = set(qid for qid, _ in candidates[:budget])

    pred_by_qid = {}
    rows = []
    for item in items:
        qid = item["qid"]
        use_risk = qid in keep
        chosen = risk_by_qid[qid] if use_risk else ms_records[qid]
        top = top_calibrated_row(chosen) if use_risk else {}
        pred_by_qid[qid] = chosen.get("windows", [])
        rows.append(
            {
                "qid": qid,
                "query": item.get("query", ""),
                "vid": item.get("vid", ""),
                "chosen_representation": chosen.get("chosen_representation", "ms_clap") if use_risk else "ms_clap",
                "candidate_source": chosen.get("candidate_source", "") if use_risk else "",
                "candidate_source_rank": chosen.get("candidate_source_rank", -1) if use_risk else -1,
                "changed_from_ms": use_risk,
                "pred_gain": float(top.get("pred_gain", 0.0)),
                "pred_risk": float(top.get("pred_risk", 0.0)),
                "pred_good_risk": float(top.get("pred_good_risk", 0.0)),
                "pred_utility": float(top.get("pred_utility", 0.0)),
                "ms_top1_iou": float(ms_records[qid].get("top1_iou", 0.0)),
                "adapter_top1_iou": float(risk_by_qid[qid].get("adapter_top1_iou", 0.0)),
                "ms_category": ms_records[qid].get("category", ""),
                "adapter_category": risk_by_qid[qid].get("adapter_category", ""),
                "windows": chosen.get("windows", []),
                "top_rows": chosen.get("top_rows", []) if use_risk else [],
            }
        )
    metrics, case_rows = evaluate_predictions(items, pred_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, rows


def tune_risk_gate(items, risk_rows, ms_records, objective):
    best = None
    if objective == "balanced":
        grid = {
            "min_gain": [0.45, 0.55, 0.65, 0.75],
            "max_risk": [0.25, 0.35, 0.45, 0.55],
            "max_good_risk": [0.08, 0.15, 0.25],
            "min_utility": [-0.10, 0.0, 0.10],
            "budget": [0.05, 0.10, 0.20, 0.30, 0.50],
        }
    elif objective == "precision":
        grid = {
            "min_gain": [0.55, 0.65, 0.75, 0.85],
            "max_risk": [0.15, 0.25, 0.35],
            "max_good_risk": [0.04, 0.08, 0.12],
            "min_utility": [-0.10, 0.0, 0.10, 0.20],
            "budget": [0.03, 0.05, 0.08, 0.10, 0.15],
        }
    else:
        raise ValueError(f"unknown objective: {objective}")
    for min_gain in grid["min_gain"]:
        for max_risk in grid["max_risk"]:
            for max_good_risk in grid["max_good_risk"]:
                for min_utility in grid["min_utility"]:
                    for budget in grid["budget"]:
                        params = {
                            "min_gain": min_gain,
                            "max_risk": max_risk,
                            "max_good_risk": max_good_risk,
                            "min_utility": min_utility,
                            "risk_alpha": 1.0,
                            "good_risk_alpha": 0.8,
                            "utility_alpha": 0.4,
                            "quality_alpha": 0.2,
                            "budget": budget,
                        }
                        metrics, rows = apply_risk_gate(items, risk_rows, ms_records, params)
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
                        else:
                            if summary["changed"] < 20:
                                continue
                            key = (
                                -summary["good_regressed"],
                                -summary["regressed"],
                                summary["intervention_precision"],
                                metrics["R1@0.7"],
                                metrics["top1_iou"],
                                -summary["changed"],
                            )
                        if best is None or key > best["key"]:
                            best = {"params": params, "metrics": metrics, "rows": rows, "summary": summary, "key": key}
    return best


def prediction_summary(rows):
    def avg(values):
        values = list(values)
        return sum(values) / max(len(values), 1)

    return {
        "avg_pred_gain": avg(row.get("pred_gain", 0.0) for row in rows),
        "avg_pred_risk": avg(row.get("pred_risk", 0.0) for row in rows),
        "avg_pred_good_risk": avg(row.get("pred_good_risk", 0.0) for row in rows),
        "avg_pred_utility": avg(row.get("pred_utility", 0.0) for row in rows),
        "gain_positive_avg_gain": avg(row.get("pred_gain", 0.0) for row in rows if row.get("target_gain", 0.0) > 0.5),
        "risk_positive_avg_risk": avg(row.get("pred_risk", 0.0) for row in rows if row.get("target_risk", 0.0) > 0.5),
        "good_risk_positive_avg_good_risk": avg(
            row.get("pred_good_risk", 0.0) for row in rows if row.get("target_good_risk", 0.0) > 0.5
        ),
        "avg_true_utility": avg(row.get("target_utility", 0.0) for row in rows),
    }


def load_reference_metrics(v1_path, gate_path):
    metrics = {}
    if v1_path and os.path.exists(v1_path):
        with open(v1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        src = data.get("metrics", {})
        metrics["candidate_level_fusion_v1_ref"] = src.get("candidate_level_fusion_v1")
        metrics["candidate_level_balanced_gate_v1_ref"] = src.get("candidate_level_balanced_gate_v1")
        metrics["merged_candidate_oracle_v1_ref"] = src.get("merged_candidate_oracle")
        metrics["merged_candidate_oracle_20pct_budget_v1_ref"] = src.get("merged_candidate_oracle_20pct_budget")
    if gate_path and os.path.exists(gate_path):
        with open(gate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        src = data.get("metrics", {})
        metrics["representation_gate_v2_ref"] = src.get("risk_aware_balanced_gate")
        metrics["whole_prediction_oracle_ref"] = src.get("oracle_ms_or_adapter")
    return {key: value for key, value in metrics.items() if value}


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
                str(stats["adapter_top1_changed"]),
                fmt_pct(stats["intervention_precision"]),
            ]
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Candidate-Level Risk Calibration V2</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #e8eef8; }}
    img {{ width: 100%; max-width: 1180px; display: block; margin: 14px 0; border: 1px solid #ddd; background: white; }}
  </style>
</head>
<body>
  <h1>Candidate-Level Risk Calibration V2</h1>
  <div class="note">
    Default is MS-CLAP. A merged candidate may intervene only when predicted gain is high and predicted regression risk is low.
  </div>
  <h2>Validation Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Intervention Quality</h2>
  {table(["System", "Changed", "Improved", "Regressed", "Recovered Semantic Miss", "Good Regressed", "Adapter Top1", "Precision"], intervention_rows)}
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
    parser.add_argument("--reference_v1_summary_path", default="")
    parser.add_argument("--reference_gate_summary_path", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--quality_loss_weight", type=float, default=0.8)
    parser.add_argument("--gain_loss_weight", type=float, default=1.0)
    parser.add_argument("--risk_loss_weight", type=float, default=1.2)
    parser.add_argument("--good_risk_loss_weight", type=float, default=1.8)
    parser.add_argument("--utility_loss_weight", type=float, default=0.6)
    parser.add_argument("--lambda_pairwise", type=float, default=0.25)
    parser.add_argument("--quality_alpha", type=float, default=0.7)
    parser.add_argument("--gain_alpha", type=float, default=0.8)
    parser.add_argument("--risk_alpha", type=float, default=0.7)
    parser.add_argument("--good_risk_alpha", type=float, default=0.8)
    parser.add_argument("--utility_alpha", type=float, default=0.35)
    parser.add_argument("--topn_per_source", type=int, default=2)
    parser.add_argument("--source_quotas", default="")
    parser.add_argument("--feature_version", choices=["basic", "shape_v2"], default="shape_v2")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_quotas = parse_source_quotas(args.source_quotas)

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

    print("building merged train candidates...")
    train_rows = build_fusion_rows(
        train_ms_items,
        train_adapter_items,
        train_ms_records,
        train_adapter_records,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    print("building merged val candidates...")
    val_rows = build_fusion_rows(
        val_ms_items,
        val_adapter_items,
        val_ms_records,
        val_adapter_records,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    train_rows = add_risk_labels(train_rows, train_ms_records)
    val_rows = add_risk_labels(val_rows, val_ms_records)
    print("train candidates:", len(train_rows))
    print("val candidates:", len(val_rows))
    print("feature dim:", len(train_rows[0]["features"]) if train_rows else 0)

    train_dataset = RiskCandidateDataset(train_rows)
    val_dataset = RiskCandidateDataset(val_rows, feature_mean=train_dataset.feature_mean, feature_std=train_dataset.feature_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = CandidateRiskCalibrator(train_dataset.features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    losses = {
        "gain": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.gain).to(device)),
        "risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.risk).to(device)),
        "good_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.good_risk).to(device)),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metrics = None
    best_epoch = 0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, losses, args)
        scored_val = predict_rows(model, val_dataset, val_rows, device, args)
        val_metrics, _ = scored_rows_to_records(val_ms_items, scored_val, val_ms_records, val_adapter_records)
        print(
            f"[epoch {epoch + 1}] loss={train_metrics['loss']:.4f} "
            f"R1@0.7={100 * val_metrics['R1@0.7']:.2f} "
            f"R1@0.5={100 * val_metrics['R1@0.5']:.2f}"
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
                    "epoch": best_epoch,
                    "metrics": best_metrics,
                },
                os.path.join(args.output_dir, "best.pt"),
            )

    ckpt = torch.load(os.path.join(args.output_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    scored_train = predict_rows(model, train_dataset, train_rows, device, args)
    scored_val = predict_rows(model, val_dataset, val_rows, device, args)
    risk_rerank_metrics, risk_rerank_rows = scored_rows_to_records(val_ms_items, scored_val, val_ms_records, val_adapter_records)
    train_rerank_metrics, train_rerank_rows = scored_rows_to_records(train_ms_items, scored_train, train_ms_records, train_adapter_records)

    balanced_train = tune_risk_gate(train_ms_items, train_rerank_rows, train_ms_records, objective="balanced")
    precision_train = tune_risk_gate(train_ms_items, train_rerank_rows, train_ms_records, objective="precision") or balanced_train
    balanced_metrics, balanced_rows = apply_risk_gate(val_ms_items, risk_rerank_rows, val_ms_records, balanced_train["params"])
    precision_metrics, precision_rows = apply_risk_gate(val_ms_items, risk_rerank_rows, val_ms_records, precision_train["params"])

    ms_oracle_metrics, ms_oracle_rows = oracle_from_rows(val_ms_items, val_rows, val_ms_records, val_adapter_records, representation="ms_clap")
    adapter_oracle_metrics, adapter_oracle_rows = oracle_from_rows(
        val_ms_items,
        val_rows,
        val_ms_records,
        val_adapter_records,
        representation="adapter",
    )
    merged_oracle_metrics, merged_oracle_rows = oracle_from_rows(val_ms_items, val_rows, val_ms_records, val_adapter_records)
    budget_oracle_metrics, budget_oracle_rows = apply_oracle_budget(val_ms_items, merged_oracle_rows, val_ms_records, budget=0.2)

    metrics = {
        "ms_clap_shape_v2_top2": val_ms_metrics,
        "adapter_shape_v2_top2": val_adapter_metrics,
        "candidate_risk_reranker_v2": risk_rerank_metrics,
        "candidate_risk_balanced_gate_v2": balanced_metrics,
        "candidate_risk_precision_gate_v2": precision_metrics,
        "ms_candidate_oracle": ms_oracle_metrics,
        "adapter_candidate_oracle": adapter_oracle_metrics,
        "merged_candidate_oracle": merged_oracle_metrics,
        "merged_candidate_oracle_20pct_budget": budget_oracle_metrics,
    }
    metrics.update(load_reference_metrics(args.reference_v1_summary_path, args.reference_gate_summary_path))

    interventions = {
        "candidate_risk_reranker_v2": summarize_interventions(risk_rerank_rows),
        "candidate_risk_balanced_gate_v2": summarize_interventions(balanced_rows),
        "candidate_risk_precision_gate_v2": summarize_interventions(precision_rows),
        "merged_candidate_oracle": summarize_interventions(merged_oracle_rows),
        "merged_candidate_oracle_20pct_budget": summarize_interventions(budget_oracle_rows),
    }
    summary = {
        "best_epoch": best_epoch,
        "balanced_params": balanced_train["params"],
        "precision_params": precision_train["params"],
        "train_candidates": len(train_rows),
        "val_candidates": len(val_rows),
        "feature_dim": int(train_dataset.features.shape[1]),
        "label_counts": {
            "train_gain": int(sum(row["target_gain"] for row in train_rows)),
            "train_risk": int(sum(row["target_risk"] for row in train_rows)),
            "train_good_risk": int(sum(row["target_good_risk"] for row in train_rows)),
            "train_semantic_recovery": int(sum(row["target_semantic_recovery"] for row in train_rows)),
            "val_gain": int(sum(row["target_gain"] for row in val_rows)),
            "val_risk": int(sum(row["target_risk"] for row in val_rows)),
            "val_good_risk": int(sum(row["target_good_risk"] for row in val_rows)),
            "val_semantic_recovery": int(sum(row["target_semantic_recovery"] for row in val_rows)),
        },
        "prediction_summary": {
            "train": prediction_summary(scored_train),
            "val": prediction_summary(scored_val),
        },
        "metrics": metrics,
        "interventions": interventions,
        "mode_counts": {
            "candidate_risk_reranker_v2": mode_counts(risk_rerank_rows),
            "candidate_risk_balanced_gate_v2": mode_counts(balanced_rows),
            "candidate_risk_precision_gate_v2": mode_counts(precision_rows),
            "merged_candidate_oracle": mode_counts(merged_oracle_rows),
            "merged_candidate_oracle_20pct_budget": mode_counts(budget_oracle_rows),
        },
        "candidate_source_counts": {
            "candidate_risk_reranker_v2": candidate_source_counts(risk_rerank_rows),
            "candidate_risk_balanced_gate_v2": candidate_source_counts(balanced_rows),
            "candidate_risk_precision_gate_v2": candidate_source_counts(precision_rows),
            "merged_candidate_oracle": candidate_source_counts(merged_oracle_rows),
        },
        "args": vars(args),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for name, rows in [
        ("risk_reranker_case_rows.json", risk_rerank_rows),
        ("balanced_gate_case_rows.json", balanced_rows),
        ("precision_gate_case_rows.json", precision_rows),
        ("merged_oracle_case_rows.json", merged_oracle_rows),
        ("merged_oracle_budget_case_rows.json", budget_oracle_rows),
    ]:
        with open(os.path.join(args.output_dir, name), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    write_case_svgs(args.output_dir, {item["qid"]: item for item in val_ms_items}, val_ms_records, risk_rerank_rows)
    write_html(args.output_dir, summary)
    print(json.dumps({"best_epoch": best_epoch, "risk_reranker_metrics": risk_rerank_metrics}, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
