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

from evidence_decoder_experiment import save_json, save_jsonl, svg_for_comparison
from train_candidate_level_representation_fusion import (
    build_fusion_rows,
    candidate_source_counts,
    mode_counts,
    oracle_from_rows,
    scored_rows_to_records,
    summarize_interventions,
)
from train_learned_evidence_decoder import load_items, model_selection_key, parse_source_quotas
from train_representation_fusion_selector import quality_key
from train_two_mode_decoder_selector import apply_decoder, fmt_float, fmt_pct, load_decoder_checkpoint, table, window_iou


POSITIVE_TEMPORAL_CATEGORIES = {"boundary_error", "candidate_exists", "evidence_good_decode_bad"}


def clamp(value, lo=-1.5, hi=1.5):
    return max(lo, min(hi, float(value)))


def candidate_key_from_iou(value):
    value = float(value)
    return (value >= 0.7, value >= 0.5, value)


def add_semantic_temporal_labels(rows, ms_records, quality_target="evidence"):
    labeled = []
    for row in rows:
        item = dict(row)
        ms_record = ms_records[item["qid"]]
        ms_iou = float(ms_record.get("top1_iou", 0.0))
        raw_iou = float(item.get("target_raw_iou", 0.0))
        ms_cat = ms_record.get("category", "")
        ms_key = quality_key(ms_record)
        cand_key = candidate_key_from_iou(raw_iou)
        ms_top_window = ms_record.get("windows", [[]])[0] if ms_record.get("windows") else []
        anchor_overlap = window_iou(item.get("window", []), ms_top_window)
        anchor_disagreement = 1.0 - anchor_overlap

        anchor_reliable = ms_cat == "good" or ms_iou >= 0.7
        semantic_gain = ms_cat == "semantic_miss" and raw_iou >= 0.5
        temporal_gain = cand_key > ms_key and (
            ms_cat in POSITIVE_TEMPORAL_CATEGORIES or (ms_cat != "semantic_miss" and raw_iou >= 0.7)
        )
        anchor_risk = ms_cat == "good" and (raw_iou < 0.7 or raw_iou < ms_iou)
        semantic_risk = ms_cat != "semantic_miss" and raw_iou < 0.1
        temporal_risk = cand_key < ms_key and not semantic_risk
        strict_gain = cand_key > ms_key

        utility = raw_iou - ms_iou
        utility += 0.55 * (float(raw_iou >= 0.7) - float(ms_iou >= 0.7))
        utility += 0.25 * (float(raw_iou >= 0.5) - float(ms_iou >= 0.5))
        if semantic_gain:
            utility += 0.25
        if temporal_gain:
            utility += 0.20
        if anchor_risk:
            utility -= 0.50
        if semantic_risk:
            utility -= 0.25
        if temporal_risk:
            utility -= 0.20
        utility = clamp(utility)

        if quality_target == "fusion":
            target_quality = max(0.0, min(1.0, float(item.get("target_iou", raw_iou))))
        else:
            target_quality = raw_iou
            if raw_iou >= 0.7:
                target_quality += 0.20
            elif raw_iou >= 0.5:
                target_quality += 0.08
            if semantic_gain:
                target_quality += 0.08
            if temporal_gain:
                target_quality += 0.08
            if anchor_risk:
                target_quality -= 0.20
            if semantic_risk:
                target_quality -= 0.10
            if temporal_risk:
                target_quality -= 0.10
            target_quality = max(0.0, min(1.0, float(target_quality)))

        item.update(
            {
                "target_quality": target_quality,
                "target_anchor_reliable": float(anchor_reliable),
                "target_semantic_gain": float(semantic_gain),
                "target_temporal_gain": float(temporal_gain),
                "target_anchor_risk": float(anchor_risk),
                "target_semantic_risk": float(semantic_risk),
                "target_temporal_risk": float(temporal_risk),
                "target_strict_gain": float(strict_gain),
                "target_utility": utility,
                "anchor_overlap": float(anchor_overlap),
                "anchor_disagreement": float(anchor_disagreement),
            }
        )
        labeled.append(item)
    return labeled


class SemanticTemporalCandidateDataset(Dataset):
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
        self.anchor_reliable = torch.tensor([row["target_anchor_reliable"] for row in rows], dtype=torch.float32)
        self.semantic_gain = torch.tensor([row["target_semantic_gain"] for row in rows], dtype=torch.float32)
        self.temporal_gain = torch.tensor([row["target_temporal_gain"] for row in rows], dtype=torch.float32)
        self.anchor_risk = torch.tensor([row["target_anchor_risk"] for row in rows], dtype=torch.float32)
        self.semantic_risk = torch.tensor([row["target_semantic_risk"] for row in rows], dtype=torch.float32)
        self.temporal_risk = torch.tensor([row["target_temporal_risk"] for row in rows], dtype=torch.float32)
        self.strict_gain = torch.tensor([row["target_strict_gain"] for row in rows], dtype=torch.float32)
        self.utility = torch.tensor([row["target_utility"] for row in rows], dtype=torch.float32)
        self.anchor_disagreement = torch.tensor([row["anchor_disagreement"] for row in rows], dtype=torch.float32)
        self.qids = [row["qid"] for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.quality[idx],
            self.anchor_reliable[idx],
            self.semantic_gain[idx],
            self.temporal_gain[idx],
            self.anchor_risk[idx],
            self.semantic_risk[idx],
            self.temporal_risk[idx],
            self.strict_gain[idx],
            self.utility[idx],
            self.anchor_disagreement[idx],
            self.qids[idx],
        )


class SemanticTemporalCandidateScorer(nn.Module):
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
        self.anchor_reliability_head = nn.Linear(hidden_dim, 1)
        self.semantic_gain_head = nn.Linear(hidden_dim, 1)
        self.temporal_gain_head = nn.Linear(hidden_dim, 1)
        self.anchor_risk_head = nn.Linear(hidden_dim, 1)
        self.semantic_risk_head = nn.Linear(hidden_dim, 1)
        self.temporal_risk_head = nn.Linear(hidden_dim, 1)
        self.strict_gain_head = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        h = self.shared(features)
        return {
            "quality_logit": self.quality_head(h).squeeze(-1),
            "anchor_reliability_logit": self.anchor_reliability_head(h).squeeze(-1),
            "semantic_gain_logit": self.semantic_gain_head(h).squeeze(-1),
            "temporal_gain_logit": self.temporal_gain_head(h).squeeze(-1),
            "anchor_risk_logit": self.anchor_risk_head(h).squeeze(-1),
            "semantic_risk_logit": self.semantic_risk_head(h).squeeze(-1),
            "temporal_risk_logit": self.temporal_risk_head(h).squeeze(-1),
            "strict_gain_logit": self.strict_gain_head(h).squeeze(-1),
            "utility": 1.5 * torch.tanh(self.utility_head(h).squeeze(-1)),
        }


def pos_weight(labels, max_value=20.0):
    pos = float(labels.sum().item())
    neg = float(labels.numel() - pos)
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(min(max_value, neg / pos))


def action_score(outputs, args, anchor_disagreement=None):
    quality = torch.sigmoid(outputs["quality_logit"])
    if args.score_mode == "quality":
        return quality

    anchor_reliability = torch.sigmoid(outputs["anchor_reliability_logit"])
    if args.score_mode == "quality_guard":
        score = quality
        if anchor_disagreement is not None and args.anchor_guard_alpha > 0:
            score = score - args.anchor_guard_alpha * anchor_reliability * anchor_disagreement
        return score

    semantic_gain = torch.sigmoid(outputs["semantic_gain_logit"])
    temporal_gain = torch.sigmoid(outputs["temporal_gain_logit"])
    anchor_risk = torch.sigmoid(outputs["anchor_risk_logit"])
    semantic_risk = torch.sigmoid(outputs["semantic_risk_logit"])
    temporal_risk = torch.sigmoid(outputs["temporal_risk_logit"])
    strict_gain = torch.sigmoid(outputs["strict_gain_logit"])
    utility = outputs["utility"]
    score = (
        args.quality_alpha * quality
        + args.semantic_alpha * semantic_gain
        + args.temporal_alpha * temporal_gain
        + args.strict_gain_alpha * strict_gain
        + args.utility_alpha * utility
        - args.anchor_risk_alpha * anchor_risk
        - args.semantic_risk_alpha * semantic_risk
        - args.temporal_risk_alpha * temporal_risk
    )
    if anchor_disagreement is not None and args.anchor_guard_alpha > 0:
        score = score - args.anchor_guard_alpha * anchor_reliability * anchor_disagreement
    return score


def pairwise_utility_loss(scores, targets, qids, margin=0.10, max_pairs=256):
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


def train_epoch(model, loader, optimizer, losses, device, args):
    model.train()
    totals = defaultdict(float)
    steps = 0
    for batch in loader:
        (
            features,
            quality_y,
            anchor_reliable_y,
            semantic_y,
            temporal_y,
            anchor_risk_y,
            semantic_risk_y,
            temporal_risk_y,
            strict_gain_y,
            utility_y,
            anchor_disagreement,
            qids,
        ) = batch
        features = features.to(device)
        quality_y = quality_y.to(device)
        anchor_reliable_y = anchor_reliable_y.to(device)
        semantic_y = semantic_y.to(device)
        temporal_y = temporal_y.to(device)
        anchor_risk_y = anchor_risk_y.to(device)
        semantic_risk_y = semantic_risk_y.to(device)
        temporal_risk_y = temporal_risk_y.to(device)
        strict_gain_y = strict_gain_y.to(device)
        utility_y = utility_y.to(device)
        anchor_disagreement = anchor_disagreement.to(device)

        out = model(features)
        score = action_score(out, args, anchor_disagreement)
        losses_by_name = {
            "quality": F.mse_loss(torch.sigmoid(out["quality_logit"]), quality_y),
            "anchor_reliable": losses["anchor_reliable"](out["anchor_reliability_logit"], anchor_reliable_y),
            "semantic": losses["semantic"](out["semantic_gain_logit"], semantic_y),
            "temporal": losses["temporal"](out["temporal_gain_logit"], temporal_y),
            "anchor_risk": losses["anchor_risk"](out["anchor_risk_logit"], anchor_risk_y),
            "semantic_risk": losses["semantic_risk"](out["semantic_risk_logit"], semantic_risk_y),
            "temporal_risk": losses["temporal_risk"](out["temporal_risk_logit"], temporal_risk_y),
            "strict_gain": losses["strict_gain"](out["strict_gain_logit"], strict_gain_y),
            "utility": F.mse_loss(out["utility"], utility_y),
        }
        rank_targets = quality_y if args.score_mode in {"quality", "quality_guard"} else utility_y
        losses_by_name["rank"] = (
            pairwise_utility_loss(score, rank_targets, qids) if args.lambda_pairwise > 0 else torch.zeros((), device=device)
        )
        loss = (
            args.quality_loss_weight * losses_by_name["quality"]
            + args.anchor_reliable_loss_weight * losses_by_name["anchor_reliable"]
            + args.semantic_loss_weight * losses_by_name["semantic"]
            + args.temporal_loss_weight * losses_by_name["temporal"]
            + args.anchor_risk_loss_weight * losses_by_name["anchor_risk"]
            + args.semantic_risk_loss_weight * losses_by_name["semantic_risk"]
            + args.temporal_risk_loss_weight * losses_by_name["temporal_risk"]
            + args.strict_gain_loss_weight * losses_by_name["strict_gain"]
            + args.utility_loss_weight * losses_by_name["utility"]
            + args.lambda_pairwise * losses_by_name["rank"]
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
def score_rows(model, dataset, rows, device, args):
    model.eval()
    features = dataset.features.to(device)
    output_rows = []
    for st in range(0, len(features), 4096):
        out = model(features[st : st + 4096])
        anchor_disagreement = dataset.anchor_disagreement[st : st + 4096].to(device)
        score = action_score(out, args, anchor_disagreement)
        anchor_reliability = torch.sigmoid(out["anchor_reliability_logit"])
        batch = {
            "pred_quality_head": torch.sigmoid(out["quality_logit"]).cpu().tolist(),
            "pred_anchor_reliability": anchor_reliability.cpu().tolist(),
            "pred_semantic_gain": torch.sigmoid(out["semantic_gain_logit"]).cpu().tolist(),
            "pred_temporal_gain": torch.sigmoid(out["temporal_gain_logit"]).cpu().tolist(),
            "pred_anchor_risk": torch.sigmoid(out["anchor_risk_logit"]).cpu().tolist(),
            "pred_semantic_risk": torch.sigmoid(out["semantic_risk_logit"]).cpu().tolist(),
            "pred_temporal_risk": torch.sigmoid(out["temporal_risk_logit"]).cpu().tolist(),
            "pred_strict_gain": torch.sigmoid(out["strict_gain_logit"]).cpu().tolist(),
            "pred_utility": out["utility"].cpu().tolist(),
            "pred_anchor_guard": (anchor_reliability * anchor_disagreement).cpu().tolist(),
            "anchor_disagreement": anchor_disagreement.cpu().tolist(),
            "pred_quality": score.cpu().tolist(),
        }
        size = len(batch["pred_quality"])
        for idx in range(size):
            row = dict(rows[st + idx])
            for key, values in batch.items():
                row[key] = float(values[idx])
            output_rows.append(row)
    return output_rows


def role_counts(rows):
    top_rows = [row.get("top_rows", [{}])[0] if row.get("top_rows") else {} for row in rows]
    return {
        "anchor_reliable_top1": sum(float(row.get("pred_anchor_reliability", 0.0)) >= 0.5 for row in top_rows),
        "anchor_guard_top1": sum(float(row.get("pred_anchor_guard", 0.0)) >= 0.5 for row in top_rows),
        "semantic_gain_top1": sum(float(row.get("pred_semantic_gain", 0.0)) >= 0.5 for row in top_rows),
        "temporal_gain_top1": sum(float(row.get("pred_temporal_gain", 0.0)) >= 0.5 for row in top_rows),
        "anchor_risk_top1": sum(float(row.get("pred_anchor_risk", 0.0)) >= 0.5 for row in top_rows),
        "semantic_risk_top1": sum(float(row.get("pred_semantic_risk", 0.0)) >= 0.5 for row in top_rows),
        "temporal_risk_top1": sum(float(row.get("pred_temporal_risk", 0.0)) >= 0.5 for row in top_rows),
    }


def label_counts(dataset):
    return {
        "anchor_reliable": int(dataset.anchor_reliable.sum().item()),
        "semantic_gain": int(dataset.semantic_gain.sum().item()),
        "temporal_gain": int(dataset.temporal_gain.sum().item()),
        "anchor_risk": int(dataset.anchor_risk.sum().item()),
        "semantic_risk": int(dataset.semantic_risk.sum().item()),
        "temporal_risk": int(dataset.temporal_risk.sum().item()),
        "strict_gain": int(dataset.strict_gain.sum().item()),
    }


def write_case_svgs(output_dir, items_by_qid, ms_records, rows, max_items=10):
    improvements = []
    regressions = []
    for row in rows:
        delta = float(row.get("top1_iou", 0.0)) - float(row.get("ms_top1_iou", 0.0))
        record = (delta, row["qid"], row)
        if delta > 0.25:
            improvements.append(record)
        elif delta < -0.25:
            regressions.append(record)
    for folder, records, reverse in [("improvements", improvements, True), ("regressions", regressions, False)]:
        path = os.path.join(output_dir, "comparison_svgs", folder)
        os.makedirs(path, exist_ok=True)
        for idx, (_, qid, row) in enumerate(sorted(records, key=lambda item: item[0], reverse=reverse)[:max_items]):
            with open(os.path.join(path, f"{idx:03d}_{qid}.svg"), "w", encoding="utf-8") as f:
                f.write(svg_for_comparison(items_by_qid[qid], ms_records[qid]["windows"], row["windows"]))


def image_tags(output_dir, folder):
    folder_path = os.path.join(output_dir, "comparison_svgs", folder)
    if not os.path.isdir(folder_path):
        return "<p>No cases.</p>"
    tags = []
    for name in sorted(os.listdir(folder_path)):
        tags.append(f'<img src="comparison_svgs/{folder}/{escape(name)}" alt="{escape(name)}"/>')
    return "".join(tags) if tags else "<p>No cases.</p>"


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
    role_rows = [[escape(name), str(value)] for name, value in summary["role_counts"].items()]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Semantic-Temporal Candidate Scorer V2</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #0f766e; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #e8f3f1; }}
    img {{ width: 100%; max-width: 1180px; display: block; margin: 14px 0; border: 1px solid #ddd; background: white; }}
  </style>
</head>
<body>
  <h1>Semantic-Temporal Candidate Scorer V2</h1>
  <div class="note">
    Candidate score is decomposed into semantic gain, temporal gain, strict gain, utility, and risk heads.
  </div>
  <h2>Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Intervention Quality</h2>
  {table(["System", "Changed", "Improved", "Regressed", "Recovered Semantic", "Good Regressed", "Precision"], intervention_rows)}
  <h2>Predicted Role Counts For Top1</h2>
  {table(["Role", "Count"], role_rows)}
  <h2>Improvements</h2>
  {image_tags(output_dir, "improvements")}
  <h2>Regressions</h2>
  {image_tags(output_dir, "regressions")}
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
    parser.add_argument("--epochs", type=int, default=30)
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
    parser.add_argument("--quality_alpha", type=float, default=0.7)
    parser.add_argument("--semantic_alpha", type=float, default=0.8)
    parser.add_argument("--temporal_alpha", type=float, default=0.8)
    parser.add_argument("--strict_gain_alpha", type=float, default=0.5)
    parser.add_argument("--utility_alpha", type=float, default=0.35)
    parser.add_argument("--anchor_risk_alpha", type=float, default=0.8)
    parser.add_argument("--semantic_risk_alpha", type=float, default=0.5)
    parser.add_argument("--temporal_risk_alpha", type=float, default=0.6)
    parser.add_argument("--anchor_guard_alpha", type=float, default=0.7)
    parser.add_argument("--score_mode", choices=["hybrid", "quality", "quality_guard"], default="hybrid")
    parser.add_argument("--quality_target", choices=["evidence", "fusion"], default="evidence")
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

    print("applying decoders...")
    train_ms_records, train_ms_metrics = apply_decoder(train_ms_items, ms_decoder, device, "ms_clap")
    val_ms_records, val_ms_metrics = apply_decoder(val_ms_items, ms_decoder, device, "ms_clap")
    train_adapter_records, train_adapter_metrics = apply_decoder(train_adapter_items, adapter_decoder, device, "adapter")
    val_adapter_records, val_adapter_metrics = apply_decoder(val_adapter_items, adapter_decoder, device, "adapter")

    print("building candidates...")
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print("feature dim:", int(train_dataset.features.shape[1]))
    print("label counts:", json.dumps({"train": label_counts(train_dataset), "val": label_counts(val_dataset)}, indent=2))

    model = SemanticTemporalCandidateScorer(
        input_dim=int(train_dataset.features.shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    losses = {
        "anchor_reliable": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.anchor_reliable).to(device)),
        "semantic": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.semantic_gain).to(device)),
        "temporal": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.temporal_gain).to(device)),
        "anchor_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.anchor_risk).to(device)),
        "semantic_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.semantic_risk).to(device)),
        "temporal_risk": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.temporal_risk).to(device)),
        "strict_gain": nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_dataset.strict_gain).to(device)),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metrics = None
    best_epoch = 0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, losses, device, args)
        scored_val = score_rows(model, val_dataset, val_rows, device, args)
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
    scored_val = score_rows(model, val_dataset, val_rows, device, args)
    scorer_metrics, scorer_rows = scored_rows_to_records(val_ms_items, scored_val, val_ms_records, val_adapter_records)
    ms_oracle_metrics, ms_oracle_rows = oracle_from_rows(val_ms_items, val_rows, val_ms_records, val_adapter_records, representation="ms_clap")
    adapter_oracle_metrics, adapter_oracle_rows = oracle_from_rows(
        val_ms_items,
        val_rows,
        val_ms_records,
        val_adapter_records,
        representation="adapter",
    )
    merged_oracle_metrics, merged_oracle_rows = oracle_from_rows(val_ms_items, val_rows, val_ms_records, val_adapter_records)

    metrics = {
        "ms_clap_shape_v2_top2": val_ms_metrics,
        "adapter_shape_v2_top2": val_adapter_metrics,
        "semantic_temporal_candidate_scorer_v2": scorer_metrics,
        "ms_candidate_oracle": ms_oracle_metrics,
        "adapter_candidate_oracle": adapter_oracle_metrics,
        "merged_candidate_oracle": merged_oracle_metrics,
    }
    interventions = {
        "semantic_temporal_candidate_scorer_v2": summarize_interventions(scorer_rows),
        "merged_candidate_oracle": summarize_interventions(merged_oracle_rows),
    }
    summary = {
        "best_epoch": best_epoch,
        "train_candidates": len(train_rows),
        "val_candidates": len(val_rows),
        "feature_dim": int(train_dataset.features.shape[1]),
        "label_counts": {"train": label_counts(train_dataset), "val": label_counts(val_dataset)},
        "metrics": metrics,
        "interventions": interventions,
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
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(scorer_rows, os.path.join(args.output_dir, "case_rows.json"))
    save_json(merged_oracle_rows, os.path.join(args.output_dir, "merged_oracle_case_rows.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in scorer_rows],
        os.path.join(args.output_dir, "predictions.jsonl"),
    )
    write_case_svgs(args.output_dir, {item["qid"]: item for item in val_ms_items}, val_ms_records, scorer_rows)
    write_html(args.output_dir, summary)
    print(json.dumps({"best_epoch": best_epoch, "metrics": scorer_metrics, "interventions": interventions["semantic_temporal_candidate_scorer_v2"]}, indent=2))
    print("saved", os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
