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
from torch.utils.data import DataLoader, TensorDataset

from evidence_decoder_experiment import categorize, evidence_gap, nms_windows, smooth_scores
from metrics import best_iou_among_topk, best_iou_for_window, recall_at_1_iou, recall_at_k_iou
from train_learned_evidence_decoder import (
    CandidateDataset,
    CandidateScorer,
    SOURCE_NAMES,
    build_rows,
    load_items,
    parse_source_quotas,
    score_rows,
)


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
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * max(0.0, min(100.0, float(pct))) / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(values[lo])
    alpha = k - lo
    return float(values[lo] * (1.0 - alpha) + values[hi] * alpha)


def safe_div(num, den):
    return float(num) / float(den) if abs(float(den)) > 1e-8 else 0.0


def fmt_pct(value):
    return f"{100.0 * float(value):.2f}%"


def fmt_float(value):
    return f"{float(value):.4f}"


def table(headers, rows):
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def load_decoder_checkpoint(path, device):
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
        "path": path,
        "args": args,
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "model": model,
    }


def nms_rows(rows, topn=10, threshold=0.7):
    windows = [[*row["window"][:2], row["pred_quality"]] for row in rows]
    kept_windows = nms_windows(windows, threshold=threshold, topn=topn)
    used = set()
    kept_rows = []
    for window in kept_windows:
        best_idx = None
        best_delta = None
        for idx, row in enumerate(rows):
            if idx in used:
                continue
            row_window = row["window"]
            if int(row_window[0]) == int(window[0]) and int(row_window[1]) == int(window[1]):
                delta = abs(float(row["pred_quality"]) - float(window[2]))
                if best_delta is None or delta < best_delta:
                    best_idx = idx
                    best_delta = delta
        if best_idx is not None:
            used.add(best_idx)
            kept = dict(rows[best_idx])
            kept["window"] = [float(window[0]), float(window[1]), float(window[2])]
            kept_rows.append(kept)
    return kept_rows


def apply_decoder(items, decoder, device, mode_name):
    args = decoder["args"]
    source_quotas = parse_source_quotas(args.get("source_quotas", ""))
    rows = build_rows(
        items,
        topn_per_source=int(args.get("topn_per_source", 10)),
        feature_version=args.get("feature_version", "basic"),
        source_quotas=source_quotas,
    )
    dataset = CandidateDataset(rows, feature_mean=decoder["feature_mean"], feature_std=decoder["feature_std"])
    scored_rows = score_rows(decoder["model"], dataset, rows, device)
    grouped = defaultdict(list)
    for row in scored_rows:
        grouped[row["qid"]].append(row)

    by_qid = {}
    pred_by_qid = {}
    for item in items:
        qid = item["qid"]
        ranked_rows = sorted(grouped.get(qid, []), key=lambda row: row["pred_quality"], reverse=True)
        kept_rows = nms_rows(ranked_rows, topn=10, threshold=0.7)
        windows = [row["window"] for row in kept_rows]
        pred_by_qid[qid] = windows
        by_qid[qid] = {
            "mode": mode_name,
            "qid": qid,
            "windows": windows,
            "rows": kept_rows,
            "raw_candidate_count": len(ranked_rows),
        }
    metrics, case_rows = evaluate_predictions(items, pred_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for qid, record in by_qid.items():
        record.update(case_by_qid.get(qid, {}))
    return by_qid, metrics


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


def global_evidence_features(item):
    scores = [float(value) for value in item.get("evidence_scores", [])]
    work = smooth_scores(scores, kernel=5)
    if not work:
        work = [0.0]
    top_values = sorted(work, reverse=True)[:5]
    return [
        mean(work),
        std(work),
        max(work),
        percentile(work, 50),
        percentile(work, 70),
        percentile(work, 80),
        percentile(work, 90),
        mean(top_values),
        max(work) - mean(work),
        percentile(work, 90) - percentile(work, 50),
    ]


def source_one_hot(source):
    values = [0.0] * len(SOURCE_NAMES)
    if source in SOURCE_NAMES:
        values[SOURCE_NAMES.index(source)] = 1.0
    return values


def mode_features(mode_record, duration):
    rows = mode_record.get("rows", [])
    windows = mode_record.get("windows", [])
    scores = [float(row.get("pred_quality", 0.0)) for row in rows[:5]]
    top = rows[0] if rows else {}
    top_window = windows[0] if windows else [0.0, 0.0, 0.0]
    st, ed = float(top_window[0]), float(top_window[1])
    length = max(ed - st, 0.0)
    center = 0.5 * (st + ed)
    top_score = float(top_window[2]) if len(top_window) > 2 else 0.0
    second_score = float(windows[1][2]) if len(windows) > 1 and len(windows[1]) > 2 else 0.0
    return [
        top_score,
        top_score - second_score,
        mean(scores),
        std(scores),
        safe_div(length, duration),
        safe_div(center, duration),
        safe_div(float(top.get("source_rank", 0)), 10.0),
        safe_div(float(mode_record.get("raw_candidate_count", 0)), 100.0),
    ] + source_one_hot(top.get("source", ""))


def window_iou(a, b):
    if not a or not b:
        return 0.0
    inter = max(0.0, min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0])))
    union = (float(a[1]) - float(a[0])) + (float(b[1]) - float(b[0])) - inter
    return inter / max(union, 1e-6)


def selector_feature_row(item, coverage_record, precision_record):
    scores = [float(value) for value in item.get("evidence_scores", [])]
    duration = max(float(item.get("duration", len(scores))), float(len(scores)), 1.0)
    cov_top = coverage_record.get("windows", [[0.0, 0.0, 0.0]])[0] if coverage_record.get("windows") else [0.0, 0.0, 0.0]
    pre_top = precision_record.get("windows", [[0.0, 0.0, 0.0]])[0] if precision_record.get("windows") else [0.0, 0.0, 0.0]
    cov_features = mode_features(coverage_record, duration)
    pre_features = mode_features(precision_record, duration)
    cross = [
        float(pre_top[2]) - float(cov_top[2]),
        abs((float(pre_top[0]) + float(pre_top[1])) * 0.5 - (float(cov_top[0]) + float(cov_top[1])) * 0.5) / duration,
        abs((float(pre_top[1]) - float(pre_top[0])) - (float(cov_top[1]) - float(cov_top[0]))) / duration,
        window_iou(cov_top, pre_top),
    ]
    return global_evidence_features(item) + cov_features + pre_features + cross


def target_prefers_precision(coverage_record, precision_record):
    cov_key = (
        float(coverage_record.get("top1_iou", 0.0)) >= 0.7,
        float(coverage_record.get("top1_iou", 0.0)) >= 0.5,
        float(coverage_record.get("top1_iou", 0.0)),
        float(coverage_record.get("top5_iou", 0.0)),
    )
    pre_key = (
        float(precision_record.get("top1_iou", 0.0)) >= 0.7,
        float(precision_record.get("top1_iou", 0.0)) >= 0.5,
        float(precision_record.get("top1_iou", 0.0)),
        float(precision_record.get("top5_iou", 0.0)),
    )
    return 1.0 if pre_key > cov_key else 0.0


class ModeSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=48, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_selector_dataset(items, coverage_by_qid, precision_by_qid):
    features = []
    labels = []
    qids = []
    for item in items:
        qid = item["qid"]
        features.append(selector_feature_row(item, coverage_by_qid[qid], precision_by_qid[qid]))
        labels.append(target_prefers_precision(coverage_by_qid[qid], precision_by_qid[qid]))
        qids.append(qid)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, y, qids


def train_selector(x_train, y_train, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    mean_vec = x_train.mean(dim=0)
    std_vec = x_train.std(dim=0, unbiased=False).clamp(min=1e-6)
    x_norm = (x_train - mean_vec) / std_vec
    dataset = TensorDataset(x_norm, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ModeSelector(input_dim=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(y_train.sum().item())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([safe_div(neg, pos) if pos > 0 else 1.0], dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, mean_vec, std_vec


@torch.no_grad()
def predict_selector(model, mean_vec, std_vec, x):
    model.eval()
    logits = model((x - mean_vec) / std_vec)
    return torch.sigmoid(logits).cpu().tolist()


def rows_to_pred_by_qid(records_by_qid, qids):
    return {qid: records_by_qid[qid].get("windows", []) for qid in qids}


def select_predictions(items, coverage_by_qid, precision_by_qid, probs, threshold):
    pred_by_qid = {}
    selector_rows = []
    for item, prob in zip(items, probs):
        qid = item["qid"]
        choose_precision = float(prob) >= threshold
        chosen = precision_by_qid[qid] if choose_precision else coverage_by_qid[qid]
        pred_by_qid[qid] = chosen.get("windows", [])
        selector_rows.append(
            {
                "qid": qid,
                "query": item.get("query", ""),
                "chosen_mode": "precision" if choose_precision else "coverage",
                "precision_probability": float(prob),
                "coverage_top1_iou": float(coverage_by_qid[qid].get("top1_iou", 0.0)),
                "precision_top1_iou": float(precision_by_qid[qid].get("top1_iou", 0.0)),
                "coverage_category": coverage_by_qid[qid].get("category", ""),
                "precision_category": precision_by_qid[qid].get("category", ""),
                "windows": chosen.get("windows", []),
            }
        )
    metrics, case_rows = evaluate_predictions(items, pred_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in selector_rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, selector_rows


def tune_threshold(items, coverage_by_qid, precision_by_qid, probs):
    best_threshold = 0.5
    best_key = None
    for step in range(5, 96):
        threshold = step / 100.0
        metrics, _ = select_predictions(items, coverage_by_qid, precision_by_qid, probs, threshold)
        key = (
            float(metrics.get("R1@0.7", 0.0)),
            float(metrics.get("R1@0.5", 0.0)),
            float(metrics.get("top1_iou", 0.0)),
            -abs(threshold - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def oracle_select(items, coverage_by_qid, precision_by_qid):
    pred_by_qid = {}
    rows = []
    for item in items:
        qid = item["qid"]
        choose_precision = target_prefers_precision(coverage_by_qid[qid], precision_by_qid[qid]) > 0.5
        chosen = precision_by_qid[qid] if choose_precision else coverage_by_qid[qid]
        pred_by_qid[qid] = chosen.get("windows", [])
        rows.append({"qid": qid, "chosen_mode": "precision" if choose_precision else "coverage"})
    metrics, case_rows = evaluate_predictions(items, pred_by_qid)
    return metrics, rows, case_rows


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

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Two-Mode Decoder Selector</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #7c3aed; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eceff3; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Two-Mode Decoder Selector</h1>
  <div class="note">
    Coverage mode: shape_v2 top2. Precision mode: source gate v2. Selector threshold learned on train: <b>{summary["threshold"]:.2f}</b>.
  </div>
  <h2>Validation Metrics</h2>
  {table(["Decoder", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Selector Usage</h2>
  {table(["Split", "Coverage", "Precision"], [
      ["train", str(summary["train_mode_counts"].get("coverage", 0)), str(summary["train_mode_counts"].get("precision", 0))],
      ["val", str(summary["val_mode_counts"].get("coverage", 0)), str(summary["val_mode_counts"].get("precision", 0))],
  ])}
</body>
</html>"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_evidence_path", required=True)
    parser.add_argument("--val_evidence_path", required=True)
    parser.add_argument("--coverage_ckpt", required=True)
    parser.add_argument("--precision_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_items = load_items(args.train_evidence_path)
    val_items = load_items(args.val_evidence_path)
    coverage = load_decoder_checkpoint(args.coverage_ckpt, device)
    precision = load_decoder_checkpoint(args.precision_ckpt, device)

    print("applying coverage decoder to train...")
    train_cov, train_cov_metrics = apply_decoder(train_items, coverage, device, "coverage")
    print("applying precision decoder to train...")
    train_pre, train_pre_metrics = apply_decoder(train_items, precision, device, "precision")
    print("applying coverage decoder to val...")
    val_cov, val_cov_metrics = apply_decoder(val_items, coverage, device, "coverage")
    print("applying precision decoder to val...")
    val_pre, val_pre_metrics = apply_decoder(val_items, precision, device, "precision")

    x_train, y_train, train_qids = build_selector_dataset(train_items, train_cov, train_pre)
    x_val, y_val, val_qids = build_selector_dataset(val_items, val_cov, val_pre)
    print("selector train labels:", {"precision": int(y_train.sum().item()), "coverage": int(len(y_train) - y_train.sum().item())})

    selector, feature_mean, feature_std = train_selector(x_train, y_train, args.epochs, args.batch_size, args.lr, args.seed)
    train_probs = predict_selector(selector, feature_mean, feature_std, x_train)
    val_probs = predict_selector(selector, feature_mean, feature_std, x_val)
    threshold = tune_threshold(train_items, train_cov, train_pre, train_probs)

    train_sel_metrics, train_sel_rows = select_predictions(train_items, train_cov, train_pre, train_probs, threshold)
    val_sel_metrics, val_sel_rows = select_predictions(val_items, val_cov, val_pre, val_probs, threshold)
    val_oracle_metrics, val_oracle_choices, val_oracle_rows = oracle_select(val_items, val_cov, val_pre)

    train_mode_counts = Counter(row["chosen_mode"] for row in train_sel_rows)
    val_mode_counts = Counter(row["chosen_mode"] for row in val_sel_rows)
    summary = {
        "threshold": threshold,
        "train_mode_counts": dict(train_mode_counts),
        "val_mode_counts": dict(val_mode_counts),
        "selector_train_labels": {
            "precision": int(y_train.sum().item()),
            "coverage": int(len(y_train) - y_train.sum().item()),
        },
        "metrics": {
            "coverage_shape_v2_top2": val_cov_metrics,
            "precision_source_gate_v2": val_pre_metrics,
            "two_mode_selector": val_sel_metrics,
            "two_mode_oracle": val_oracle_metrics,
        },
        "args": vars(args),
    }
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(val_sel_rows, os.path.join(args.output_dir, "selector_case_rows.json"))
    save_json(val_oracle_rows, os.path.join(args.output_dir, "oracle_case_rows.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in val_sel_rows],
        os.path.join(args.output_dir, "selector_predictions.jsonl"),
    )
    torch.save(
        {
            "model_state_dict": selector.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "threshold": threshold,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "selector.pt"),
    )
    write_html(args.output_dir, summary)
    print(json.dumps(summary, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
