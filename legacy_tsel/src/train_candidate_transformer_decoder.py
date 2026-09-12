import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from evidence_decoder_experiment import (
    categorize,
    evidence_gap,
    fmt_pct,
    html_table,
    nms_windows,
    save_json,
    save_jsonl,
)
from metrics import best_iou_among_topk, best_iou_for_window, recall_at_1_iou, recall_at_k_iou
from train_learned_evidence_decoder import (
    SOURCE_NAMES,
    build_rows,
    load_items,
    model_selection_key,
    parse_source_quotas,
)


def feature_stats(rows):
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    return features.mean(dim=0), features.std(dim=0, unbiased=False).clamp(min=1e-6)


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row)
    return grouped


class CandidateSetDataset(Dataset):
    def __init__(self, items, rows, feature_mean=None, feature_std=None):
        self.items = items
        self.rows_by_qid = group_rows(rows)
        if feature_mean is None or feature_std is None:
            feature_mean, feature_std = feature_stats(rows)
        self.feature_mean = feature_mean.cpu()
        self.feature_std = feature_std.cpu()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        rows = self.rows_by_qid.get(item["qid"], [])
        if rows:
            features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
            features = (features - self.feature_mean) / self.feature_std
            targets = torch.tensor([row["target_iou"] for row in rows], dtype=torch.float32)
        else:
            features = torch.zeros(1, len(self.feature_mean), dtype=torch.float32)
            targets = torch.zeros(1, dtype=torch.float32)
        return {
            "qid": item["qid"],
            "features": features,
            "targets": targets,
            "rows": rows,
        }


def collate_candidate_sets(batch):
    max_len = max(record["features"].shape[0] for record in batch)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    targets = torch.zeros(len(batch), max_len, dtype=torch.float32)
    valid = torch.zeros(len(batch), max_len, dtype=torch.bool)
    qids = []
    rows = []
    for idx, record in enumerate(batch):
        length = record["features"].shape[0]
        features[idx, :length] = record["features"]
        targets[idx, :length] = record["targets"]
        valid[idx, :length] = True
        qids.append(record["qid"])
        rows.append(record["rows"])
    return features, targets, valid, qids, rows


class CandidateContextTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, valid_mask):
        hidden = self.input_proj(features)
        encoded = self.encoder(hidden, src_key_padding_mask=~valid_mask)
        return self.scorer(encoded).squeeze(-1)


def pairwise_set_loss(logits, targets, valid_mask, margin=0.08, min_delta=0.15):
    losses = []
    for sample_logits, sample_targets, sample_valid in zip(logits, targets, valid_mask):
        indices = torch.where(sample_valid)[0]
        if len(indices) < 2:
            continue
        sample_logits = sample_logits[indices]
        sample_targets = sample_targets[indices]
        diff = sample_targets[:, None] - sample_targets[None, :]
        pair_mask = diff > min_delta
        if not pair_mask.any():
            continue
        score_diff = sample_logits[:, None] - sample_logits[None, :]
        losses.append(F.relu(margin - score_diff[pair_mask]).mean())
    if not losses:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return torch.stack(losses).mean()


def train_epoch(model, loader, optimizer, device, lambda_pairwise):
    model.train()
    totals = defaultdict(float)
    steps = 0
    for features, targets, valid, _, _ in loader:
        features = features.to(device)
        targets = targets.to(device)
        valid = valid.to(device)
        logits = model(features, valid)
        pointwise = F.mse_loss(torch.sigmoid(logits)[valid], targets[valid])
        pairwise = pairwise_set_loss(logits, targets, valid) if lambda_pairwise > 0 else torch.zeros((), device=device)
        loss = pointwise + lambda_pairwise * pairwise
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        totals["loss"] += float(loss.item())
        totals["pointwise"] += float(pointwise.item())
        totals["pairwise"] += float(pairwise.item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def score_dataset(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_candidate_sets)
    model.eval()
    scored = {}
    for features, _, valid, qids, batch_rows in loader:
        features = features.to(device)
        valid = valid.to(device)
        logits = model(features, valid)
        probs = torch.sigmoid(logits).cpu()
        valid_cpu = valid.cpu()
        for batch_idx, qid in enumerate(qids):
            rows = []
            row_count = int(valid_cpu[batch_idx].sum().item())
            for row, score in zip(batch_rows[batch_idx][:row_count], probs[batch_idx, :row_count].tolist()):
                item = dict(row)
                item["pred_quality"] = float(score)
                rows.append(item)
            scored[qid] = rows
    return scored


def rows_to_predictions(scored_by_qid, topn=10):
    predictions = {}
    for qid, rows in scored_by_qid.items():
        windows = [[*row["window"][:2], row["pred_quality"]] for row in sorted(rows, key=lambda x: x["pred_quality"], reverse=True)]
        predictions[qid] = nms_windows(windows, threshold=0.7, topn=topn)
    return predictions


def evaluate_predictions(items, pred_by_qid, topn=10):
    totals = defaultdict(float)
    categories = Counter()
    rows = []
    for item in items:
        qid = item["qid"]
        windows = pred_by_qid.get(qid, [])[:topn]
        gt_windows = item["gt_windows"]
        gap = evidence_gap(item)
        category = categorize(windows, gt_windows, gap)
        categories[category] += 1
        top1_iou = best_iou_for_window(windows[0], gt_windows) if windows else 0.0
        top5_iou = max((best_iou_for_window(pred, gt_windows) for pred in windows[:5]), default=0.0)
        totals["R1@0.5"] += recall_at_1_iou(windows, gt_windows, threshold=0.5)
        totals["R1@0.7"] += recall_at_1_iou(windows, gt_windows, threshold=0.7)
        totals["R3@0.7"] += recall_at_k_iou(windows, gt_windows, threshold=0.7, k=3)
        totals["R5@0.7"] += recall_at_k_iou(windows, gt_windows, threshold=0.7, k=5)
        totals["best_iou_top5"] += best_iou_among_topk(windows, gt_windows, k=5)
        totals["top1_iou"] += top1_iou
        totals["top5_iou"] += top5_iou
        rows.append(
            {
                "qid": qid,
                "query": item.get("query", ""),
                "vid": item.get("vid", ""),
                "gt_windows": gt_windows,
                "windows": windows,
                "top1_iou": top1_iou,
                "top5_iou": top5_iou,
                "category": category,
                "evidence_gap": gap,
            }
        )
    n = max(len(items), 1)
    metrics = {key: value / n for key, value in totals.items()}
    metrics["num_samples"] = len(items)
    metrics["categories"] = dict(categories)
    return metrics, rows


def write_report(output_dir, metrics, args):
    category_names = ["good", "boundary_error", "candidate_exists", "evidence_good_decode_bad", "semantic_miss"]
    metric_rows = [
        [
            "candidate_transformer",
            fmt_pct(metrics["R1@0.5"]),
            fmt_pct(metrics["R1@0.7"]),
            fmt_pct(metrics["R3@0.7"]),
            fmt_pct(metrics["R5@0.7"]),
            f"{metrics['top1_iou']:.4f}",
            f"{metrics['best_iou_top5']:.4f}",
        ]
    ]
    n = max(metrics["num_samples"], 1)
    cat_rows = [
        ["candidate_transformer"]
        + [f"{metrics['categories'].get(cat, 0)} ({100 * metrics['categories'].get(cat, 0) / n:.1f}%)" for cat in category_names]
    ]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Candidate Context Transformer Decoder</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eeeeea; }}
  </style>
</head>
<body>
  <h1>Candidate Context Transformer Decoder</h1>
  <div class="note">
    Candidate setting: topn_per_source={args.topn_per_source}, source_quotas={args.source_quotas or "uniform"}.
    The model scores candidates jointly within each query using self-attention.
  </div>
  <h2>Metrics</h2>
  {html_table(["Decoder", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5"], metric_rows)}
  <h2>Failure Categories</h2>
  {html_table(["Decoder"] + category_names, cat_rows)}
</body>
</html>"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_evidence_path", required=True)
    parser.add_argument("--val_evidence_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_pairwise", type=float, default=0.2)
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

    train_items = load_items(args.train_evidence_path)
    val_items = load_items(args.val_evidence_path)
    source_quotas = parse_source_quotas(args.source_quotas)
    print("building train rows...")
    train_rows = build_rows(
        train_items,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    print("building val rows...")
    val_rows = build_rows(
        val_items,
        topn_per_source=args.topn_per_source,
        feature_version=args.feature_version,
        source_quotas=source_quotas,
    )
    print("train candidates:", len(train_rows))
    print("val candidates:", len(val_rows))
    print("source quotas:", source_quotas if source_quotas else "uniform")

    feature_mean, feature_std = feature_stats(train_rows)
    train_dataset = CandidateSetDataset(train_items, train_rows, feature_mean=feature_mean, feature_std=feature_std)
    val_dataset = CandidateSetDataset(val_items, val_rows, feature_mean=feature_mean, feature_std=feature_std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_candidate_sets,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CandidateContextTransformer(
        input_dim=len(feature_mean),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_metrics = None
    best_rows = None
    best_epoch = 0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.lambda_pairwise)
        scored_val = score_dataset(model, val_dataset, device, batch_size=args.batch_size)
        pred_by_qid = rows_to_predictions(scored_val, topn=10)
        val_metrics, val_case_rows = evaluate_predictions(val_items, pred_by_qid, topn=10)
        print(
            f"[epoch {epoch + 1}] loss={train_metrics['loss']:.4f} "
            f"point={train_metrics['pointwise']:.4f} pair={train_metrics['pairwise']:.4f} "
            f"R1@0.7={100 * val_metrics['R1@0.7']:.2f} R1@0.5={100 * val_metrics['R1@0.5']:.2f}"
        )
        if best_metrics is None or model_selection_key(val_metrics) > model_selection_key(best_metrics):
            best_metrics = val_metrics
            best_rows = val_case_rows
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "args": vars(args),
                    "epoch": best_epoch,
                    "metrics": best_metrics,
                },
                os.path.join(args.output_dir, "best.pt"),
            )

    ckpt = torch.load(os.path.join(args.output_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    scored_val = score_dataset(model, val_dataset, device, batch_size=args.batch_size)
    pred_by_qid = rows_to_predictions(scored_val, topn=10)
    best_metrics, best_rows = evaluate_predictions(val_items, pred_by_qid, topn=10)
    best_epoch = int(ckpt.get("epoch", best_epoch))

    save_json(
        {
            "best_epoch": best_epoch,
            "train_candidates": len(train_rows),
            "val_candidates": len(val_rows),
            "metrics": best_metrics,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "summary.json"),
    )
    save_json(best_rows, os.path.join(args.output_dir, "transformer_case_rows.json"))
    save_jsonl(
        [{"qid": qid, "pred_relevant_windows": windows} for qid, windows in pred_by_qid.items()],
        os.path.join(args.output_dir, "transformer_predictions.jsonl"),
    )
    write_report(args.output_dir, best_metrics, args)
    print(json.dumps({"best_epoch": best_epoch, "metrics": best_metrics}, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
