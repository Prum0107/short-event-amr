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
    DECODERS,
    categorize,
    evidence_gap,
    fmt_pct,
    html_table,
    nms_windows,
    save_json,
    save_jsonl,
    score_values,
    smooth_scores,
    svg_for_comparison,
    write_html_report,
)
from metrics import best_iou_among_topk, best_iou_for_window, recall_at_1_iou, recall_at_k_iou


SOURCE_NAMES = [
    "current_start_end",
    "threshold_mean",
    "threshold_p60",
    "threshold_p70",
    "peak_drop",
    "peak_expand",
    "multiscale_10_20_40_80_150",
    "dense_contrast",
]


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prefix_sum(scores):
    prefix = [0.0]
    for score in scores:
        prefix.append(prefix[-1] + float(score))
    return prefix


def range_values(scores, st, ed):
    st = max(0, int(st))
    ed = min(len(scores), int(ed))
    return scores[st:ed]


def range_mean(prefix, st, ed):
    st = max(0, int(st))
    ed = min(len(prefix) - 1, int(ed))
    if ed <= st:
        return 0.0
    return (prefix[ed] - prefix[st]) / (ed - st)


def value_at(scores, idx):
    if not scores:
        return 0.0
    idx = max(0, min(len(scores) - 1, int(idx)))
    return float(scores[idx])


def safe_div(num, den):
    return float(num) / float(den) if abs(float(den)) > 1e-8 else 0.0


def mean_value(values):
    return score_values(values, "mean") if values else 0.0


def max_value(values):
    return score_values(values, "max") if values else 0.0


def min_value(values):
    return min(values) if values else 0.0


def percentile_value(sorted_values, q):
    if not sorted_values:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    idx = int(round(q * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


def fraction_at_least(values, threshold):
    if not values:
        return 0.0
    return sum(1 for value in values if value >= threshold) / len(values)


def longest_run_at_least(values, threshold):
    best = 0
    current = 0
    for value in values:
        if value >= threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def make_feature_context(item):
    scores = [float(x) for x in item["evidence_scores"]]
    work3 = smooth_scores(scores, kernel=3)
    work5 = smooth_scores(scores, kernel=5)
    duration = max(float(item.get("duration", len(scores))), float(len(scores)), 1.0)
    sorted_work3 = sorted(work3)
    global_mean = mean_value(work3)
    global_std = float(torch.tensor(work3).std(unbiased=False).item()) if len(work3) > 1 else 0.0
    global_top_indices = sorted(range(len(work3)), key=lambda idx: work3[idx], reverse=True)[:5] if work3 else []
    return {
        "scores": scores,
        "work3": work3,
        "work5": work5,
        "duration": duration,
        "sorted_work3": sorted_work3,
        "global_mean": global_mean,
        "global_std": global_std,
        "global_max": max_value(work3),
        "global_sum": sum(work3),
        "p60": percentile_value(sorted_work3, 0.60),
        "p70": percentile_value(sorted_work3, 0.70),
        "p80": percentile_value(sorted_work3, 0.80),
        "p90": percentile_value(sorted_work3, 0.90),
        "global_top_indices": global_top_indices,
        "peak_idx": global_top_indices[0] if global_top_indices else 0,
    }


def candidate_iou(window, gt_windows):
    return best_iou_for_window(window, gt_windows)


def dedupe_candidates(candidates):
    seen = set()
    out = []
    for cand in candidates:
        st, ed = int(round(cand["window"][0])), int(round(cand["window"][1]))
        if ed <= st:
            continue
        key = (st, ed, cand["source"])
        if key in seen:
            continue
        seen.add(key)
        cand["window"] = [st, ed, float(cand.get("score", 0.0))]
        out.append(cand)
    return out


def generate_candidates(item, topn_per_source=10, source_quotas=None):
    candidates = []
    for source in SOURCE_NAMES:
        source_topn = int(source_quotas.get(source, topn_per_source)) if source_quotas else topn_per_source
        if source_topn <= 0:
            continue
        windows = DECODERS[source](item, source_topn)
        for rank, window in enumerate(windows):
            candidates.append(
                {
                    "source": source,
                    "source_rank": rank,
                    "window": window[:3],
                    "score": float(window[2]) if len(window) > 2 else 0.0,
                }
            )
    candidates = dedupe_candidates(candidates)
    ranked = nms_windows([cand["window"] for cand in candidates], threshold=0.95, topn=500)
    keep = {(int(w[0]), int(w[1])) for w in ranked}
    return [cand for cand in candidates if (int(cand["window"][0]), int(cand["window"][1])) in keep]


def one_hot(index, size):
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def extract_candidate_features(item, candidate, feature_version="basic", context=None):
    if context is None:
        context = make_feature_context(item)
    scores = context["scores"]
    work3 = context["work3"]
    work5 = context["work5"]
    st, ed, raw_score = candidate["window"]
    st = int(st)
    ed = int(ed)
    length = max(1, ed - st)
    duration = context["duration"]

    inside = range_values(work3, st, ed)
    inside5 = range_values(work5, st, ed)
    left = range_values(work3, st - length, st)
    right = range_values(work3, ed, ed + length)
    left_short = range_values(work3, st - 5, st)
    right_short = range_values(work3, ed, ed + 5)

    inside_mean = mean_value(inside)
    inside_max = max_value(inside)
    inside_top25 = score_values(inside, "top25") if inside else 0.0
    inside_sum = score_values(inside, "sum") if inside else 0.0
    inside_std = float(torch.tensor(inside).std(unbiased=False).item()) if len(inside) > 1 else 0.0
    left_mean = mean_value(left)
    right_mean = mean_value(right)
    left_short_mean = mean_value(left_short)
    right_short_mean = mean_value(right_short)
    outside_mean = 0.5 * (left_mean + right_mean)
    contrast = inside_mean - outside_mean
    short_contrast = inside_mean - 0.5 * (left_short_mean + right_short_mean)
    start_score = value_at(work3, st)
    end_score = value_at(work3, ed - 1)
    center_score = value_at(work3, (st + ed) // 2)
    left_drop = start_score - value_at(work3, st - 1)
    right_drop = end_score - value_at(work3, ed)
    peak_idx = context["peak_idx"]
    center = 0.5 * (st + ed)
    source_idx = SOURCE_NAMES.index(candidate["source"]) if candidate["source"] in SOURCE_NAMES else -1

    features = [
        inside_mean,
        inside_max,
        inside_top25,
        inside_sum / max(duration, 1.0),
        inside_std,
        score_values(inside5, "mean") if inside5 else 0.0,
        start_score,
        end_score,
        center_score,
        left_mean,
        right_mean,
        left_short_mean,
        right_short_mean,
        contrast,
        short_contrast,
        left_drop,
        right_drop,
        length / duration,
        math.log1p(length) / math.log1p(duration),
        center / duration,
        abs(center - peak_idx) / duration,
        float(raw_score),
        1.0 / (candidate["source_rank"] + 1.0),
    ]
    if feature_version == "shape_v2":
        near = range_values(work3, st - length, ed + length)
        near_outside = left + right
        local_left_edge = range_values(work3, st, st + min(5, length))
        local_right_edge = range_values(work3, max(st, ed - min(5, length)), ed)
        top_indices = context["global_top_indices"]
        top_inside = sum(1 for idx in top_indices if st <= idx < ed)
        top_dist = min((abs(idx - center) for idx in top_indices), default=duration)
        local_peak_idx = st
        if inside:
            local_peak_offset = max(range(len(inside)), key=lambda idx: inside[idx])
            local_peak_idx = st + local_peak_offset
        inside_min = min_value(inside)
        near_outside_max = max_value(near_outside)
        local_left_mean = mean_value(local_left_edge)
        local_right_mean = mean_value(local_right_edge)
        p60 = context["p60"]
        p70 = context["p70"]
        p80 = context["p80"]
        p90 = context["p90"]
        shape_features = [
            context["global_mean"],
            context["global_std"],
            context["global_max"],
            p60,
            p70,
            p80,
            p90,
            safe_div(inside_sum, context["global_sum"]),
            safe_div(inside_mean - context["global_mean"], context["global_std"]),
            inside_max - context["global_mean"],
            inside_max - near_outside_max,
            inside_mean - mean_value(near),
            fraction_at_least(inside, context["global_mean"]),
            fraction_at_least(inside, p70),
            fraction_at_least(inside, p80),
            fraction_at_least(inside, p90),
            safe_div(longest_run_at_least(inside, p70), length),
            safe_div(longest_run_at_least(inside, p80), length),
            fraction_at_least(inside, inside_max * 0.9),
            inside_max - inside_min,
            safe_div(inside_top25, inside_mean),
            local_left_mean - left_short_mean,
            local_right_mean - right_short_mean,
            start_score - left_short_mean,
            end_score - right_short_mean,
            abs(start_score - end_score),
            abs(left_mean - right_mean),
            max_value(left) - max_value(right),
            float(st <= peak_idx < ed),
            safe_div(top_inside, max(len(top_indices), 1)),
            safe_div(top_dist, duration),
            safe_div(abs(local_peak_idx - center), duration),
            safe_div(local_peak_idx - st, length),
            safe_div(ed - local_peak_idx, length),
        ]
        features.extend(shape_features)
    features.extend(one_hot(source_idx, len(SOURCE_NAMES)))
    return features


def build_rows(items, topn_per_source=10, feature_version="basic", source_quotas=None):
    rows = []
    for item in items:
        context = make_feature_context(item)
        for cand in generate_candidates(item, topn_per_source=topn_per_source, source_quotas=source_quotas):
            iou = candidate_iou(cand["window"], item["gt_windows"])
            rows.append(
                {
                    "qid": item["qid"],
                    "query": item.get("query", ""),
                    "vid": item.get("vid", ""),
                    "window": cand["window"],
                    "source": cand["source"],
                    "source_rank": cand["source_rank"],
                    "features": extract_candidate_features(item, cand, feature_version=feature_version, context=context),
                    "target_iou": iou,
                }
            )
    return rows


class CandidateDataset(Dataset):
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
        self.targets = torch.tensor([row["target_iou"] for row in rows], dtype=torch.float32)
        self.qids = [row["qid"] for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.qids[idx]


class CandidateScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def model_selection_key(metrics):
    return (
        float(metrics.get("R1@0.7", 0.0)),
        float(metrics.get("R1@0.5", 0.0)),
        float(metrics.get("top1_iou", 0.0)),
        float(metrics.get("best_iou_top5", 0.0)),
    )


def pairwise_ranking_loss(scores, targets, qids, margin=0.05, max_pairs=512):
    by_qid = defaultdict(list)
    for idx, qid in enumerate(qids):
        by_qid[qid].append(idx)
    losses = []
    device = scores.device
    for indices in by_qid.values():
        if len(indices) < 2:
            continue
        indices = indices[:64]
        pairs = []
        for i in indices:
            for j in indices:
                if targets[i] - targets[j] >= margin:
                    pairs.append((i, j))
        if len(pairs) > max_pairs:
            pairs = random.sample(pairs, max_pairs)
        for i, j in pairs:
            losses.append(F.relu(0.1 - (scores[i] - scores[j])))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def train_epoch(model, loader, optimizer, device, lambda_pairwise):
    model.train()
    totals = defaultdict(float)
    steps = 0
    for features, targets, qids in loader:
        features = features.to(device)
        targets = targets.to(device)
        scores = model(features)
        mse = F.mse_loss(torch.sigmoid(scores), targets)
        rank = pairwise_ranking_loss(scores, targets, qids) if lambda_pairwise > 0 else torch.zeros((), device=device)
        loss = mse + lambda_pairwise * rank
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.item())
        totals["mse"] += float(mse.item())
        totals["rank"] += float(rank.item())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def score_rows(model, dataset, rows, device):
    model.eval()
    features = dataset.features.to(device)
    scores = []
    batch_size = 4096
    for st in range(0, len(features), batch_size):
        logits = model(features[st : st + batch_size])
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    out = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item["pred_quality"] = float(score)
        out.append(item)
    return out


def rows_to_predictions(scored_rows, source_name="learned_scorer", topn=10):
    grouped = defaultdict(list)
    for row in scored_rows:
        grouped[row["qid"]].append(row)
    predictions = {}
    for qid, rows in grouped.items():
        windows = [[*row["window"][:2], row["pred_quality"]] for row in rows]
        predictions[qid] = nms_windows(windows, threshold=0.7, topn=topn)
    return predictions


def evaluate_predictions(items, pred_by_qid, topn=10):
    totals = defaultdict(float)
    categories = Counter()
    rows = []
    for item in items:
        windows = pred_by_qid.get(item["qid"], [])[:topn]
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
                "qid": item["qid"],
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


def baseline_predictions(items, decoder_name, topn=10):
    return {item["qid"]: DECODERS[decoder_name](item, topn) for item in items}


def write_comparison_html(output_dir, metrics_by_decoder, best_name, current_rows, learned_rows, items_by_qid):
    rows = []
    for name, metrics in sorted(metrics_by_decoder.items(), key=lambda x: x[1]["R1@0.7"], reverse=True):
        rows.append(
            [
                f"<b>{name}</b>" if name == best_name else name,
                fmt_pct(metrics["R1@0.5"]),
                fmt_pct(metrics["R1@0.7"]),
                fmt_pct(metrics["R3@0.7"]),
                fmt_pct(metrics["R5@0.7"]),
                f"{metrics['best_iou_top5']:.4f}",
                f"{metrics['top1_iou']:.4f}",
            ]
        )
    category_names = ["good", "boundary_error", "candidate_exists", "evidence_good_decode_bad", "semantic_miss"]
    cat_rows = []
    for name, metrics in sorted(metrics_by_decoder.items(), key=lambda x: x[1]["R1@0.7"], reverse=True):
        n = max(metrics["num_samples"], 1)
        cat_rows.append(
            [name]
            + [f"{metrics['categories'].get(cat, 0)} ({100 * metrics['categories'].get(cat, 0) / n:.1f}%)" for cat in category_names]
        )

    current_by_qid = {row["qid"]: row for row in current_rows}
    improvements = []
    regressions = []
    for row in learned_rows:
        current = current_by_qid[row["qid"]]
        delta = row["top1_iou"] - current["top1_iou"]
        if delta > 0.2:
            improvements.append((delta, row["qid"], current, row))
        elif delta < -0.2:
            regressions.append((delta, row["qid"], current, row))
    svg_root = os.path.join(output_dir, "comparison_svgs")
    if os.path.exists(svg_root):
        import shutil

        shutil.rmtree(svg_root)
    for folder, records, reverse in [("improvements", improvements, True), ("regressions", regressions, False)]:
        folder_path = os.path.join(svg_root, folder)
        os.makedirs(folder_path, exist_ok=True)
        for idx, (_, qid, current, learned) in enumerate(sorted(records, key=lambda x: x[0], reverse=reverse)[:12]):
            with open(os.path.join(folder_path, f"{idx:03d}_{qid}.svg"), "w", encoding="utf-8") as f:
                f.write(svg_for_comparison(items_by_qid[qid], current["windows"], learned["windows"]))

    def image_tags(folder):
        folder_path = os.path.join(svg_root, folder)
        if not os.path.isdir(folder_path):
            return "<p>No cases.</p>"
        tags = []
        for name in sorted(os.listdir(folder_path)):
            tags.append(f'<img src="comparison_svgs/{folder}/{name}" alt="{name}"/>')
        return "".join(tags)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Learned Evidence Decoder V1</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #9467bd; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eeeeea; }}
    img {{ width: 100%; max-width: 1180px; display: block; margin: 14px 0; border: 1px solid #ddd; background: white; }}
  </style>
</head>
<body>
  <h1>Learned Evidence Decoder V1</h1>
  <div class="note">
    Best decoder by R1@0.7: <b>{best_name}</b>. This experiment trains a small MLP to predict candidate boundary quality from evidence-window features.
  </div>
  <h2>Metric Comparison</h2>
  {html_table(["Decoder", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Best IoU Top5", "Mean Top1 IoU"], rows)}
  <h2>Failure Category Comparison</h2>
  {html_table(["Decoder"] + category_names, cat_rows)}
  <h2>Learned Decoder Improvements Over Current</h2>
  <p>Green = GT, red = current decoder top1, purple = learned decoder top1, blue = evidence curve.</p>
  {image_tags("improvements")}
  <h2>Learned Decoder Regressions Compared With Current</h2>
  {image_tags("regressions")}
</body>
</html>"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_evidence_path", required=True)
    parser.add_argument("--val_evidence_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_pairwise", type=float, default=0.2)
    parser.add_argument("--topn_per_source", type=int, default=10)
    parser.add_argument(
        "--source_quotas",
        default="",
        help="Optional comma-separated source quotas, e.g. peak_drop=3,dense_contrast=4. Missing sources use --topn_per_source.",
    )
    parser.add_argument("--feature_version", choices=["basic", "shape_v2"], default="basic")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_source_quotas(text):
    if not text:
        return None
    quotas = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"invalid source quota '{chunk}', expected name=value")
        name, value = chunk.split("=", 1)
        name = name.strip()
        if name not in SOURCE_NAMES:
            raise ValueError(f"unknown source quota name '{name}'")
        quotas[name] = int(value)
    return quotas


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
    print("feature version:", args.feature_version)
    print("source quotas:", source_quotas if source_quotas else "uniform")

    train_dataset = CandidateDataset(train_rows)
    val_dataset = CandidateDataset(val_rows, feature_mean=train_dataset.feature_mean, feature_std=train_dataset.feature_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CandidateScorer(train_dataset.features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_metrics = None
    best_rows = None
    best_epoch = 0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.lambda_pairwise)
        scored_val = score_rows(model, val_dataset, val_rows, device)
        pred_by_qid = rows_to_predictions(scored_val, topn=10)
        val_metrics, learned_rows = evaluate_predictions(val_items, pred_by_qid, topn=10)
        print(
            f"[epoch {epoch + 1}] loss={train_metrics['loss']:.4f} "
            f"R1@0.7={100 * val_metrics['R1@0.7']:.2f} "
            f"R1@0.5={100 * val_metrics['R1@0.5']:.2f}"
        )
        if best_metrics is None or model_selection_key(val_metrics) > model_selection_key(best_metrics):
            best_metrics = val_metrics
            best_rows = learned_rows
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

    best_path = os.path.join(args.output_dir, "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = int(ckpt.get("epoch", best_epoch))
        scored_val = score_rows(model, val_dataset, val_rows, device)
        learned_pred = rows_to_predictions(scored_val, topn=10)
        learned_metrics, learned_rows = evaluate_predictions(val_items, learned_pred, topn=10)
        best_metrics = learned_metrics
        best_rows = learned_rows
    else:
        scored_val = score_rows(model, val_dataset, val_rows, device)
        learned_pred = rows_to_predictions(scored_val, topn=10)
        learned_metrics, learned_rows = evaluate_predictions(val_items, learned_pred, topn=10)

    metrics_by_decoder = {}
    rows_by_decoder = {}
    for name in ["current_start_end", "peak_drop", "threshold_mean", "dense_contrast"]:
        pred = baseline_predictions(val_items, name, topn=10)
        metrics, rows = evaluate_predictions(val_items, pred, topn=10)
        metrics_by_decoder[name] = metrics
        rows_by_decoder[name] = rows
    metrics_by_decoder["learned_scorer"] = learned_metrics
    rows_by_decoder["learned_scorer"] = learned_rows

    best_name = max(metrics_by_decoder, key=lambda name: metrics_by_decoder[name]["R1@0.7"])
    save_json(
        {
            "best_decoder": best_name,
            "best_epoch": best_epoch,
            "train_candidates": len(train_rows),
            "val_candidates": len(val_rows),
            "metrics_by_decoder": metrics_by_decoder,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "summary.json"),
    )
    save_json(best_rows, os.path.join(args.output_dir, "learned_case_rows.json"))
    save_jsonl(
        [
            {
                "qid": qid,
                "pred_relevant_windows": windows,
            }
            for qid, windows in learned_pred.items()
        ],
        os.path.join(args.output_dir, "learned_predictions.jsonl"),
    )
    write_comparison_html(
        output_dir=args.output_dir,
        metrics_by_decoder=metrics_by_decoder,
        best_name=best_name,
        current_rows=rows_by_decoder["current_start_end"],
        learned_rows=rows_by_decoder["learned_scorer"],
        items_by_qid={item["qid"]: item for item in val_items},
    )
    print(json.dumps({"best_decoder": best_name, "learned_metrics": learned_metrics}, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
