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

from evidence_decoder_experiment import evidence_gap, smooth_scores
from train_two_mode_decoder_selector import (
    apply_decoder,
    evaluate_predictions,
    fmt_float,
    fmt_pct,
    global_evidence_features,
    load_decoder_checkpoint,
    mode_features,
    safe_div,
    save_json,
    save_jsonl,
    table,
    window_iou,
)
from train_learned_evidence_decoder import load_items


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def align_items(left_items, right_items):
    right_by_qid = {item["qid"]: item for item in right_items}
    pairs = []
    for item in left_items:
        qid = item["qid"]
        if qid in right_by_qid:
            pairs.append((item, right_by_qid[qid]))
    return pairs


def top_window(record):
    windows = record.get("windows", [])
    return windows[0] if windows else [0.0, 0.0, 0.0]


def window_len(window):
    return max(float(window[1]) - float(window[0]), 0.0) if window else 0.0


def window_center(window):
    return 0.5 * (float(window[0]) + float(window[1])) if window else 0.0


def score_margin(record):
    windows = record.get("windows", [])
    if not windows:
        return 0.0
    if len(windows) == 1:
        return float(windows[0][2]) if len(windows[0]) > 2 else 0.0
    return float(windows[0][2]) - float(windows[1][2])


def curve_similarity(left_item, right_item):
    left = smooth_scores([float(value) for value in left_item.get("evidence_scores", [])], kernel=5)
    right = smooth_scores([float(value) for value in right_item.get("evidence_scores", [])], kernel=5)
    n = min(len(left), len(right))
    if n <= 1:
        return [0.0, 0.0, 0.0, 0.0]
    left = left[:n]
    right = right[:n]
    left_mean = mean(left)
    right_mean = mean(right)
    left_std = std(left)
    right_std = std(right)
    corr = 0.0
    if left_std > 1e-8 and right_std > 1e-8:
        corr = sum((l - left_mean) * (r - right_mean) for l, r in zip(left, right)) / (n * left_std * right_std)
    abs_delta = mean(abs(l - r) for l, r in zip(left, right))
    max_delta = max(right) - max(left)
    mean_delta = right_mean - left_mean
    return [corr, abs_delta, max_delta, mean_delta]


def quality_key(record):
    top1 = float(record.get("top1_iou", 0.0))
    top5 = float(record.get("top5_iou", 0.0))
    return (top1 >= 0.7, top1 >= 0.5, top1, top5)


def target_prefers_adapter(ms_record, adapter_record):
    return 1.0 if quality_key(adapter_record) > quality_key(ms_record) else 0.0


def representation_feature_row(ms_item, adapter_item, ms_record, adapter_record):
    scores = [float(value) for value in ms_item.get("evidence_scores", [])]
    duration = max(float(ms_item.get("duration", len(scores))), float(len(scores)), 1.0)
    ms_top = top_window(ms_record)
    adapter_top = top_window(adapter_record)
    cross = [
        float(adapter_top[2]) - float(ms_top[2]),
        score_margin(adapter_record) - score_margin(ms_record),
        abs(window_center(adapter_top) - window_center(ms_top)) / duration,
        abs(window_len(adapter_top) - window_len(ms_top)) / duration,
        window_iou(ms_top, adapter_top),
        evidence_gap(adapter_item) - evidence_gap(ms_item),
        evidence_gap(ms_item),
        evidence_gap(adapter_item),
    ]
    return (
        global_evidence_features(ms_item)
        + global_evidence_features(adapter_item)
        + mode_features(ms_record, duration)
        + mode_features(adapter_record, duration)
        + curve_similarity(ms_item, adapter_item)
        + cross
    )


def build_selector_dataset(ms_items, adapter_items, ms_records, adapter_records):
    features = []
    labels = []
    qids = []
    for ms_item, adapter_item in align_items(ms_items, adapter_items):
        qid = ms_item["qid"]
        if qid not in ms_records or qid not in adapter_records:
            continue
        features.append(representation_feature_row(ms_item, adapter_item, ms_records[qid], adapter_records[qid]))
        labels.append(target_prefers_adapter(ms_records[qid], adapter_records[qid]))
        qids.append(qid)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), qids


class RepresentationSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=48, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_selector(x_train, y_train, args):
    torch.manual_seed(args.seed)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp(min=1e-6)
    x_norm = (x_train - feature_mean) / feature_std
    loader = DataLoader(TensorDataset(x_norm, y_train), batch_size=args.batch_size, shuffle=True)
    model = RepresentationSelector(x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = float(y_train.sum().item())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([safe_div(neg, pos) if pos > 0 else 1.0], dtype=torch.float32)
    for _ in range(args.epochs):
        model.train()
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
    return model, feature_mean, feature_std


@torch.no_grad()
def predict_probs(model, feature_mean, feature_std, x):
    model.eval()
    return torch.sigmoid(model((x - feature_mean) / feature_std)).cpu().tolist()


def select_predictions(ms_items, ms_records, adapter_records, probs, threshold, budget=None):
    candidates = []
    for item, prob in zip(ms_items, probs):
        qid = item["qid"]
        if qid not in ms_records or qid not in adapter_records:
            continue
        if float(prob) >= threshold:
            candidates.append((qid, float(prob)))
    if budget is not None:
        candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
        candidates = candidates[: int(math.ceil(float(budget) * len(ms_items)))]
    adapter_qids = set(qid for qid, _ in candidates)

    pred_by_qid = {}
    rows = []
    for item, prob in zip(ms_items, probs):
        qid = item["qid"]
        if qid not in ms_records or qid not in adapter_records:
            continue
        use_adapter = qid in adapter_qids
        chosen = adapter_records[qid] if use_adapter else ms_records[qid]
        pred_by_qid[qid] = chosen.get("windows", [])
        rows.append(
            {
                "qid": qid,
                "query": item.get("query", ""),
                "vid": item.get("vid", ""),
                "chosen_representation": "adapter" if use_adapter else "ms_clap",
                "adapter_probability": float(prob),
                "ms_top1_iou": float(ms_records[qid].get("top1_iou", 0.0)),
                "adapter_top1_iou": float(adapter_records[qid].get("top1_iou", 0.0)),
                "ms_category": ms_records[qid].get("category", ""),
                "adapter_category": adapter_records[qid].get("category", ""),
                "windows": chosen.get("windows", []),
            }
        )

    metrics, case_rows = evaluate_predictions(ms_items, pred_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, rows


def summarize_interventions(rows):
    changed = [row for row in rows if row.get("chosen_representation") == "adapter"]
    improved = 0
    regressed = 0
    same = 0
    recovered_semantic = 0
    good_regressed = 0
    for row in changed:
        ms_key = (row["ms_top1_iou"] >= 0.7, row["ms_top1_iou"] >= 0.5, row["ms_top1_iou"])
        new_key = (row["top1_iou"] >= 0.7, row["top1_iou"] >= 0.5, row["top1_iou"])
        if new_key > ms_key:
            improved += 1
        elif new_key < ms_key:
            regressed += 1
        else:
            same += 1
        if row.get("ms_category") == "semantic_miss" and row.get("category") != "semantic_miss":
            recovered_semantic += 1
        if row.get("ms_category") == "good" and row.get("category") != "good":
            good_regressed += 1
    return {
        "changed": len(changed),
        "improved": improved,
        "regressed": regressed,
        "same": same,
        "recovered_semantic_miss": recovered_semantic,
        "good_regressed": good_regressed,
        "intervention_precision": safe_div(improved, len(changed)),
    }


def tune_threshold(ms_items, ms_records, adapter_records, probs):
    best = None
    for step in range(5, 96):
        threshold = step / 100.0
        metrics, rows = select_predictions(ms_items, ms_records, adapter_records, probs, threshold)
        summary = summarize_interventions(rows)
        key = (
            metrics["R1@0.7"],
            metrics["R1@0.5"],
            metrics["top1_iou"],
            -metrics["categories"].get("semantic_miss", 0),
            -summary["good_regressed"],
        )
        if best is None or key > best["key"]:
            best = {"threshold": threshold, "metrics": metrics, "rows": rows, "summary": summary, "key": key}
    return best


def tune_conservative_gate(ms_items, ms_records, adapter_records, probs):
    best = None
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        for budget in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
            metrics, rows = select_predictions(ms_items, ms_records, adapter_records, probs, threshold, budget=budget)
            summary = summarize_interventions(rows)
            key = (
                metrics["R1@0.7"],
                metrics["R1@0.5"],
                metrics["top1_iou"],
                -metrics["categories"].get("semantic_miss", 0),
                -summary["good_regressed"],
                summary["intervention_precision"],
                -summary["changed"],
            )
            if best is None or key > best["key"]:
                best = {
                    "threshold": threshold,
                    "budget": budget,
                    "metrics": metrics,
                    "rows": rows,
                    "summary": summary,
                    "key": key,
                }
    return best


def oracle_select(ms_items, ms_records, adapter_records, budget=None):
    candidates = []
    for item in ms_items:
        qid = item["qid"]
        if qid not in ms_records or qid not in adapter_records:
            continue
        if quality_key(adapter_records[qid]) > quality_key(ms_records[qid]):
            improvement = float(adapter_records[qid].get("top1_iou", 0.0)) - float(ms_records[qid].get("top1_iou", 0.0))
            candidates.append((qid, improvement))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    if budget is not None:
        candidates = candidates[: int(math.ceil(float(budget) * len(ms_items)))]
    adapter_qids = set(qid for qid, _ in candidates)
    pred_by_qid = {}
    rows = []
    for item in ms_items:
        qid = item["qid"]
        if qid not in ms_records or qid not in adapter_records:
            continue
        use_adapter = qid in adapter_qids
        chosen = adapter_records[qid] if use_adapter else ms_records[qid]
        pred_by_qid[qid] = chosen.get("windows", [])
        rows.append(
            {
                "qid": qid,
                "chosen_representation": "adapter" if use_adapter else "ms_clap",
                "ms_top1_iou": float(ms_records[qid].get("top1_iou", 0.0)),
                "adapter_top1_iou": float(adapter_records[qid].get("top1_iou", 0.0)),
                "ms_category": ms_records[qid].get("category", ""),
                "adapter_category": adapter_records[qid].get("category", ""),
                "windows": chosen.get("windows", []),
            }
        )
    metrics, case_rows = evaluate_predictions(ms_items, pred_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, rows, summarize_interventions(rows)


def mode_counts(rows):
    return dict(Counter(row["chosen_representation"] for row in rows))


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
  <title>Representation Fusion V1</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #0f7bc1; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eceff3; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Representation Fusion V1</h1>
  <div class="note">
    Default representation is MS-CLAP. The temporal adapter is used only when the learned selector/gate chooses it.
    Selector threshold: <b>{summary["selector_threshold"]:.2f}</b>.
    Conservative gate: <code>{escape(json.dumps(summary["conservative_params"]))}</code>.
  </div>
  <h2>Validation Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Adapter Interventions</h2>
  {table(["System", "Changed", "Improved", "Regressed", "Recovered Semantic Miss", "Good Regressed", "Precision"], intervention_rows)}
  <h2>Mode Usage</h2>
  {table(["System", "MS-CLAP", "Adapter"], [
      ["selector", str(summary["mode_counts"]["selector"].get("ms_clap", 0)), str(summary["mode_counts"]["selector"].get("adapter", 0))],
      ["conservative", str(summary["mode_counts"]["conservative"].get("ms_clap", 0)), str(summary["mode_counts"]["conservative"].get("adapter", 0))],
      ["oracle", str(summary["mode_counts"]["oracle"].get("ms_clap", 0)), str(summary["mode_counts"]["oracle"].get("adapter", 0))],
  ])}
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
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

    x_train, y_train, _ = build_selector_dataset(
        train_ms_items,
        train_adapter_items,
        train_ms_records,
        train_adapter_records,
    )
    x_val, y_val, _ = build_selector_dataset(
        val_ms_items,
        val_adapter_items,
        val_ms_records,
        val_adapter_records,
    )
    print(
        "selector labels:",
        {
            "train_adapter_better": int(y_train.sum().item()),
            "train_ms_better_or_tie": int(len(y_train) - y_train.sum().item()),
            "val_adapter_better": int(y_val.sum().item()),
            "val_ms_better_or_tie": int(len(y_val) - y_val.sum().item()),
        },
    )

    selector, feature_mean, feature_std = train_selector(x_train, y_train, args)
    train_probs = predict_probs(selector, feature_mean, feature_std, x_train)
    val_probs = predict_probs(selector, feature_mean, feature_std, x_val)

    tuned = tune_threshold(train_ms_items, train_ms_records, train_adapter_records, train_probs)
    selector_metrics, selector_rows = select_predictions(
        val_ms_items,
        val_ms_records,
        val_adapter_records,
        val_probs,
        tuned["threshold"],
    )
    selector_summary = summarize_interventions(selector_rows)

    conservative_train = tune_conservative_gate(train_ms_items, train_ms_records, train_adapter_records, train_probs)
    conservative_metrics, conservative_rows = select_predictions(
        val_ms_items,
        val_ms_records,
        val_adapter_records,
        val_probs,
        conservative_train["threshold"],
        budget=conservative_train["budget"],
    )
    conservative_summary = summarize_interventions(conservative_rows)

    oracle_metrics, oracle_rows, oracle_summary = oracle_select(val_ms_items, val_ms_records, val_adapter_records)
    oracle_budget_metrics, oracle_budget_rows, oracle_budget_summary = oracle_select(
        val_ms_items,
        val_ms_records,
        val_adapter_records,
        budget=0.2,
    )

    summary = {
        "selector_threshold": tuned["threshold"],
        "conservative_params": {
            "threshold": conservative_train["threshold"],
            "budget": conservative_train["budget"],
        },
        "selector_labels": {
            "train_adapter_better": int(y_train.sum().item()),
            "train_ms_better_or_tie": int(len(y_train) - y_train.sum().item()),
            "val_adapter_better": int(y_val.sum().item()),
            "val_ms_better_or_tie": int(len(y_val) - y_val.sum().item()),
        },
        "metrics": {
            "ms_clap_shape_v2_top2": val_ms_metrics,
            "adapter_shape_v2_top2": val_adapter_metrics,
            "learned_selector": selector_metrics,
            "conservative_adapter_gate": conservative_metrics,
            "oracle_ms_or_adapter": oracle_metrics,
            "oracle_budget_20pct": oracle_budget_metrics,
        },
        "interventions": {
            "learned_selector": selector_summary,
            "conservative_adapter_gate": conservative_summary,
            "oracle_ms_or_adapter": oracle_summary,
            "oracle_budget_20pct": oracle_budget_summary,
        },
        "mode_counts": {
            "selector": mode_counts(selector_rows),
            "conservative": mode_counts(conservative_rows),
            "oracle": mode_counts(oracle_rows),
            "oracle_budget_20pct": mode_counts(oracle_budget_rows),
        },
        "args": vars(args),
    }

    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(selector_rows, os.path.join(args.output_dir, "selector_case_rows.json"))
    save_json(conservative_rows, os.path.join(args.output_dir, "conservative_case_rows.json"))
    save_json(oracle_rows, os.path.join(args.output_dir, "oracle_case_rows.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in conservative_rows],
        os.path.join(args.output_dir, "conservative_predictions.jsonl"),
    )
    torch.save(
        {
            "model_state_dict": selector.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "selector_threshold": tuned["threshold"],
            "conservative_params": summary["conservative_params"],
            "args": vars(args),
        },
        os.path.join(args.output_dir, "selector.pt"),
    )
    write_html(args.output_dir, summary)
    print(json.dumps(summary, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
