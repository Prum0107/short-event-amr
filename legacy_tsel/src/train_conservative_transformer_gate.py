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

from train_transformer_assisted_selector import (
    MODE_NAMES,
    apply_transformer_decoder,
    load_transformer_checkpoint,
    pair_features,
    top_window,
)
from train_two_mode_decoder_selector import (
    ModeSelector,
    apply_decoder,
    build_selector_dataset,
    evaluate_predictions,
    fmt_float,
    fmt_pct,
    global_evidence_features,
    load_decoder_checkpoint,
    mode_features,
    predict_selector,
    safe_div,
    save_json,
    save_jsonl,
    select_predictions,
    table,
    window_iou,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rows_by_qid(rows):
    return {row["qid"]: row for row in rows}


def record_dict_to_lists(records):
    return {mode: list(by_qid.values()) for mode, by_qid in records.items()}


def record_lists_to_dict(records):
    return {mode: rows_by_qid(rows) for mode, rows in records.items()}


def load_two_mode_selector(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    feature_mean = ckpt["feature_mean"].cpu()
    feature_std = ckpt["feature_std"].cpu()
    args = ckpt.get("args", {})
    model = ModeSelector(input_dim=int(feature_mean.numel()), hidden_dim=int(args.get("hidden_dim", 48))).cpu()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "threshold": float(ckpt.get("threshold", 0.5)),
        "args": args,
    }


def apply_two_mode_selector(items, coverage_by_qid, precision_by_qid, selector, device):
    x, _, _ = build_selector_dataset(items, coverage_by_qid, precision_by_qid)
    model = selector["model"].cpu()
    probs = predict_selector(model, selector["feature_mean"], selector["feature_std"], x)
    metrics, rows = select_predictions(
        items,
        coverage_by_qid,
        precision_by_qid,
        probs,
        selector["threshold"],
    )
    return rows_by_qid(rows), metrics


def build_or_load_records(split, items, coverage, precision, transformer, selector, device, cache_dir):
    cache_path = os.path.join(cache_dir, f"{split}_mode_records.json")
    metric_path = os.path.join(cache_dir, f"{split}_mode_metrics.json")
    if os.path.exists(cache_path) and os.path.exists(metric_path):
        print(f"loading cached {split} records...")
        return record_lists_to_dict(load_json(cache_path)), load_json(metric_path)

    print(f"applying coverage decoder to {split}...")
    cov, cov_metrics = apply_decoder(items, coverage, device, "coverage")
    print(f"applying precision decoder to {split}...")
    pre, pre_metrics = apply_decoder(items, precision, device, "precision")
    print(f"applying transformer decoder to {split}...")
    tr, tr_metrics = apply_transformer_decoder(items, transformer, device, "transformer")
    print(f"applying two-mode selector to {split}...")
    default, default_metrics = apply_two_mode_selector(items, cov, pre, selector, device)

    records = {"coverage": cov, "precision": pre, "transformer": tr, "default": default}
    metrics = {
        "coverage_shape_v2_top2": cov_metrics,
        "precision_source_gate_v2": pre_metrics,
        "transformer_source_gate": tr_metrics,
        "two_mode_default": default_metrics,
    }
    save_json(record_dict_to_lists(records), cache_path)
    save_json(metrics, metric_path)
    return records, metrics


def window_len(window):
    if not window:
        return 0.0
    return max(float(window[1]) - float(window[0]), 0.0)


def window_center(window):
    if not window:
        return 0.0
    return 0.5 * (float(window[0]) + float(window[1]))


def score_margin(record):
    windows = record.get("windows", [])
    if not windows:
        return 0.0
    if len(windows) == 1:
        return float(windows[0][2])
    return float(windows[0][2]) - float(windows[1][2])


def top_score(record):
    top = top_window(record)
    return float(top[2]) if len(top) > 2 else 0.0


def top_support(record):
    windows = record.get("windows", [])
    if len(windows) <= 1:
        return 0.0
    top = windows[0]
    return max(window_iou(top, other) for other in windows[1:5])


def quality_key(record):
    top1 = float(record.get("top1_iou", 0.0))
    top5 = float(record.get("top5_iou", 0.0))
    return (top1 >= 0.7, top1 >= 0.5, top1, top5)


def transformer_better(default_record, transformer_record):
    return quality_key(transformer_record) > quality_key(default_record)


def gate_stats_for_item(item, records):
    cov = records["coverage"]
    pre = records["precision"]
    tr = records["transformer"]
    default = records["default"]
    scores = [float(value) for value in item.get("evidence_scores", [])]
    duration = max(float(item.get("duration", len(scores))), float(len(scores)), 1.0)
    default_top = top_window(default)
    tr_top = top_window(tr)
    cov_top = top_window(cov)
    pre_top = top_window(pre)
    tr_default_iou = window_iou(tr_top, default_top)
    tr_cov_iou = window_iou(tr_top, cov_top)
    tr_pre_iou = window_iou(tr_top, pre_top)
    cov_pre_iou = window_iou(cov_top, pre_top)
    tr_best_agreement = max(tr_default_iou, tr_cov_iou, tr_pre_iou)
    default_prob = float(default.get("precision_probability", 0.5))
    default_confidence = abs(default_prob - 0.5) * 2.0
    center_delta = abs(window_center(tr_top) - window_center(default_top)) / duration
    length_delta = abs(window_len(tr_top) - window_len(default_top)) / duration
    score_advantage = top_score(tr) - top_score(default)
    return {
        "default_precision_probability": default_prob,
        "default_confidence": default_confidence,
        "two_mode_uncertainty": 1.0 - default_confidence,
        "cov_pre_iou": cov_pre_iou,
        "tr_default_iou": tr_default_iou,
        "tr_cov_iou": tr_cov_iou,
        "tr_pre_iou": tr_pre_iou,
        "tr_best_agreement": tr_best_agreement,
        "tr_margin": score_margin(tr),
        "default_margin": score_margin(default),
        "tr_score": top_score(tr),
        "default_score": top_score(default),
        "score_advantage": score_advantage,
        "tr_support": top_support(tr),
        "center_delta": center_delta,
        "length_delta": length_delta,
        "duration": duration,
    }


def intervention_score(stats):
    return (
        1.4 * stats["tr_best_agreement"]
        + 0.8 * stats["two_mode_uncertainty"]
        + 0.6 * max(0.0, stats["score_advantage"])
        + 18.0 * max(0.0, stats["tr_margin"])
        + 0.3 * (1.0 - stats["cov_pre_iou"])
        + 0.2 * stats["tr_support"]
    )


def rule_candidates(items, records_by_mode, params):
    candidates = []
    for item in items:
        qid = item["qid"]
        records = {mode: records_by_mode[mode][qid] for mode in ["coverage", "precision", "transformer", "default"]}
        stats = gate_stats_for_item(item, records)
        uncertain_or_disagree = (
            stats["two_mode_uncertainty"] >= params["min_uncertainty"]
            or stats["cov_pre_iou"] <= params["max_cov_pre_iou"]
        )
        eligible = (
            stats["tr_best_agreement"] >= params["min_agreement"]
            and stats["tr_margin"] >= params["min_margin"]
            and stats["center_delta"] <= params["max_center_delta"]
            and uncertain_or_disagree
            and top_window(records["transformer"])
        )
        if eligible:
            candidates.append(
                {
                    "qid": qid,
                    "score": intervention_score(stats),
                    "stats": stats,
                }
            )
    return sorted(candidates, key=lambda row: row["score"], reverse=True)


def select_rule_qids(items, records_by_mode, params):
    candidates = rule_candidates(items, records_by_mode, params)
    budget = int(math.ceil(float(params["budget"]) * len(items)))
    return set(row["qid"] for row in candidates[:budget])


def apply_interventions(items, records_by_mode, intervention_qids, label):
    pred_by_qid = {}
    rows = []
    for item in items:
        qid = item["qid"]
        use_transformer = qid in intervention_qids
        source = "transformer" if use_transformer else "default"
        chosen = records_by_mode[source][qid]
        default = records_by_mode["default"][qid]
        transformer = records_by_mode["transformer"][qid]
        stats = gate_stats_for_item(
            item,
            {mode: records_by_mode[mode][qid] for mode in ["coverage", "precision", "transformer", "default"]},
        )
        pred_by_qid[qid] = chosen.get("windows", [])
        rows.append(
            {
                "qid": qid,
                "query": item.get("query", ""),
                "gate_label": label,
                "chosen_source": source,
                "intervened": use_transformer,
                "default_top1_iou": float(default.get("top1_iou", 0.0)),
                "transformer_top1_iou": float(transformer.get("top1_iou", 0.0)),
                "default_category": default.get("category", ""),
                "transformer_category": transformer.get("category", ""),
                "gate_stats": stats,
                "windows": chosen.get("windows", []),
            }
        )
    metrics, case_rows = evaluate_predictions(items, pred_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, rows


def summarize_interventions(rows):
    changed = [row for row in rows if row.get("intervened")]
    improved = 0
    regressed = 0
    same = 0
    recovered_semantic = 0
    good_regressed = 0
    for row in changed:
        default_key = (
            row["default_top1_iou"] >= 0.7,
            row["default_top1_iou"] >= 0.5,
            row["default_top1_iou"],
        )
        new_key = (
            row["top1_iou"] >= 0.7,
            row["top1_iou"] >= 0.5,
            row["top1_iou"],
        )
        if new_key > default_key:
            improved += 1
        elif new_key < default_key:
            regressed += 1
        else:
            same += 1
        if row.get("default_category") == "semantic_miss" and row.get("category") != "semantic_miss":
            recovered_semantic += 1
        if row.get("default_category") == "good" and row.get("category") != "good":
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


def tune_rule_gate(items, records_by_mode):
    best_params = None
    best_metrics = None
    best_rows = None
    best_summary = None
    best_key = None
    grid = []
    for min_agreement in [0.25, 0.4, 0.55, 0.7]:
        for min_uncertainty in [0.25, 0.45, 0.65]:
            for max_cov_pre_iou in [0.35, 0.55, 0.75]:
                for min_margin in [0.0, 0.006, 0.012]:
                    for max_center_delta in [0.4, 0.7]:
                        for budget in [0.05, 0.1, 0.15]:
                            grid.append(
                                {
                                    "min_agreement": min_agreement,
                                    "min_uncertainty": min_uncertainty,
                                    "max_cov_pre_iou": max_cov_pre_iou,
                                    "min_margin": min_margin,
                                    "max_center_delta": max_center_delta,
                                    "budget": budget,
                                }
                            )
    for params in grid:
        qids = select_rule_qids(items, records_by_mode, params)
        metrics, rows = apply_interventions(items, records_by_mode, qids, "rule")
        summary = summarize_interventions(rows)
        key = (
            metrics["R1@0.7"],
            metrics["R1@0.5"],
            metrics["top1_iou"],
            -metrics["categories"].get("semantic_miss", 0),
            summary["intervention_precision"],
            -summary["changed"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_params = params
            best_metrics = metrics
            best_rows = rows
            best_summary = summary
    return best_params, best_metrics, best_rows, best_summary


def gate_feature_row(item, records):
    scores = [float(value) for value in item.get("evidence_scores", [])]
    duration = max(float(item.get("duration", len(scores))), float(len(scores)), 1.0)
    stats = gate_stats_for_item(item, records)
    stat_values = [
        stats["default_precision_probability"],
        stats["default_confidence"],
        stats["two_mode_uncertainty"],
        stats["cov_pre_iou"],
        stats["tr_default_iou"],
        stats["tr_cov_iou"],
        stats["tr_pre_iou"],
        stats["tr_best_agreement"],
        stats["tr_margin"],
        stats["default_margin"],
        stats["score_advantage"],
        stats["tr_support"],
        stats["center_delta"],
        stats["length_delta"],
    ]
    mode_blocks = []
    for mode in ["default", "transformer", "coverage", "precision"]:
        mode_blocks.extend(mode_features(records[mode], duration))
    return global_evidence_features(item) + stat_values + mode_blocks


def build_gate_dataset(items, records_by_mode):
    features = []
    labels = []
    qids = []
    for item in items:
        qid = item["qid"]
        records = {mode: records_by_mode[mode][qid] for mode in ["coverage", "precision", "transformer", "default"]}
        features.append(gate_feature_row(item, records))
        labels.append(1.0 if transformer_better(records["default"], records["transformer"]) else 0.0)
        qids.append(qid)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), qids


class SmallGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.05):
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


def train_small_gate(x_train, y_train, args):
    torch.manual_seed(args.seed)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp(min=1e-6)
    x_norm = (x_train - feature_mean) / feature_std
    loader = DataLoader(TensorDataset(x_norm, y_train), batch_size=args.batch_size, shuffle=True)
    model = SmallGate(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
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
def predict_gate_probs(model, feature_mean, feature_std, x):
    model.eval()
    return torch.sigmoid(model((x - feature_mean) / feature_std)).cpu().tolist()


def learned_intervention_qids(items, records_by_mode, probs, threshold, min_agreement, budget):
    candidates = []
    for item, prob in zip(items, probs):
        qid = item["qid"]
        records = {mode: records_by_mode[mode][qid] for mode in ["coverage", "precision", "transformer", "default"]}
        stats = gate_stats_for_item(item, records)
        if prob >= threshold and stats["tr_best_agreement"] >= min_agreement:
            candidates.append({"qid": qid, "score": float(prob), "stats": stats})
    candidates = sorted(candidates, key=lambda row: row["score"], reverse=True)
    max_changes = int(math.ceil(float(budget) * len(items)))
    return set(row["qid"] for row in candidates[:max_changes])


def tune_learned_gate(items, records_by_mode, probs):
    best_params = None
    best_metrics = None
    best_rows = None
    best_summary = None
    best_key = None
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
        for min_agreement in [0.2, 0.35, 0.5, 0.65]:
            for budget in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15]:
                qids = learned_intervention_qids(items, records_by_mode, probs, threshold, min_agreement, budget)
                metrics, rows = apply_interventions(items, records_by_mode, qids, "learned")
                summary = summarize_interventions(rows)
                key = (
                    metrics["R1@0.7"],
                    metrics["R1@0.5"],
                    metrics["top1_iou"],
                    -metrics["categories"].get("semantic_miss", 0),
                    summary["intervention_precision"],
                    -summary["changed"],
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_params = {"threshold": threshold, "min_agreement": min_agreement, "budget": budget}
                    best_metrics = metrics
                    best_rows = rows
                    best_summary = summary
    return best_params, best_metrics, best_rows, best_summary


def intervention_oracle(items, records_by_mode, budget=None):
    candidates = []
    for item in items:
        qid = item["qid"]
        default = records_by_mode["default"][qid]
        transformer = records_by_mode["transformer"][qid]
        if transformer_better(default, transformer):
            improvement = float(transformer.get("top1_iou", 0.0)) - float(default.get("top1_iou", 0.0))
            candidates.append((qid, improvement))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    if budget is not None:
        candidates = candidates[: int(math.ceil(float(budget) * len(items)))]
    qids = set(qid for qid, _ in candidates)
    return apply_interventions(items, records_by_mode, qids, "oracle") + (len(qids),)


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
                str(data["recovered_semantic_miss"]),
                str(data["good_regressed"]),
                fmt_pct(data["intervention_precision"]),
            ]
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Conservative Transformer Gate V5.1</title>
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
  <h1>Conservative Transformer Gate V5.1</h1>
  <div class="note">
    Default prediction is the two-mode selector. The Transformer may intervene only under a small budget.
    Rule params: <code>{escape(json.dumps(summary["rule_params"]))}</code>.
    Learned params: <code>{escape(json.dumps(summary["learned_params"]))}</code>.
  </div>
  <h2>Validation Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Interventions</h2>
  {table(["Gate", "Changed", "Improved", "Regressed", "Recovered Semantic Miss", "Good Regressed", "Intervention Precision"], intervention_rows)}
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
    parser.add_argument("--transformer_ckpt", required=True)
    parser.add_argument("--two_mode_selector_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_items = load_json(args.train_evidence_path)
    val_items = load_json(args.val_evidence_path)
    coverage = load_decoder_checkpoint(args.coverage_ckpt, device)
    precision = load_decoder_checkpoint(args.precision_ckpt, device)
    transformer = load_transformer_checkpoint(args.transformer_ckpt, device)
    selector = load_two_mode_selector(args.two_mode_selector_ckpt, device)

    train_records, train_mode_metrics = build_or_load_records(
        "train",
        train_items,
        coverage,
        precision,
        transformer,
        selector,
        device,
        args.output_dir,
    )
    val_records, val_mode_metrics = build_or_load_records(
        "val",
        val_items,
        coverage,
        precision,
        transformer,
        selector,
        device,
        args.output_dir,
    )

    print("tuning rule gate on train...")
    rule_params, rule_train_metrics, rule_train_rows, rule_train_summary = tune_rule_gate(train_items, train_records)
    rule_val_qids = select_rule_qids(val_items, val_records, rule_params)
    rule_val_metrics, rule_val_rows = apply_interventions(val_items, val_records, rule_val_qids, "rule")
    rule_val_summary = summarize_interventions(rule_val_rows)

    print("training learned gate...")
    x_train, y_train, _ = build_gate_dataset(train_items, train_records)
    x_val, y_val, _ = build_gate_dataset(val_items, val_records)
    small_gate, feature_mean, feature_std = train_small_gate(x_train, y_train, args)
    train_probs = predict_gate_probs(small_gate, feature_mean, feature_std, x_train)
    val_probs = predict_gate_probs(small_gate, feature_mean, feature_std, x_val)
    learned_params, learned_train_metrics, learned_train_rows, learned_train_summary = tune_learned_gate(
        train_items,
        train_records,
        train_probs,
    )
    learned_val_qids = learned_intervention_qids(
        val_items,
        val_records,
        val_probs,
        learned_params["threshold"],
        learned_params["min_agreement"],
        learned_params["budget"],
    )
    learned_val_metrics, learned_val_rows = apply_interventions(val_items, val_records, learned_val_qids, "learned")
    learned_val_summary = summarize_interventions(learned_val_rows)

    oracle_val_metrics, oracle_val_rows, oracle_val_changed = intervention_oracle(val_items, val_records, budget=None)
    oracle_budget_metrics, oracle_budget_rows, oracle_budget_changed = intervention_oracle(
        val_items,
        val_records,
        budget=0.15,
    )
    oracle_val_summary = summarize_interventions(oracle_val_rows)
    oracle_budget_summary = summarize_interventions(oracle_budget_rows)

    summary = {
        "rule_params": rule_params,
        "learned_params": learned_params,
        "train_gate_label_counts": {
            "transformer_better": int(y_train.sum().item()),
            "default_better_or_tie": int(len(y_train) - y_train.sum().item()),
        },
        "val_gate_label_counts": {
            "transformer_better": int(y_val.sum().item()),
            "default_better_or_tie": int(len(y_val) - y_val.sum().item()),
        },
        "metrics": {
            "two_mode_default": val_mode_metrics["two_mode_default"],
            "transformer_direct": val_mode_metrics["transformer_source_gate"],
            "rule_conservative_gate_v51": rule_val_metrics,
            "learned_conservative_gate_v51": learned_val_metrics,
            "default_vs_transformer_oracle": oracle_val_metrics,
            "oracle_budget_15pct": oracle_budget_metrics,
        },
        "train_metrics": {
            "two_mode_default": train_mode_metrics["two_mode_default"],
            "transformer_direct": train_mode_metrics["transformer_source_gate"],
            "rule_conservative_gate_v51": rule_train_metrics,
            "learned_conservative_gate_v51": learned_train_metrics,
        },
        "interventions": {
            "rule_val": rule_val_summary,
            "learned_val": learned_val_summary,
            "oracle_val": oracle_val_summary,
            "oracle_budget_15pct_val": oracle_budget_summary,
        },
        "train_interventions": {
            "rule_train": rule_train_summary,
            "learned_train": learned_train_summary,
        },
        "oracle_changed": oracle_val_changed,
        "oracle_budget_15pct_changed": oracle_budget_changed,
        "args": vars(args),
    }
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(rule_val_rows, os.path.join(args.output_dir, "rule_case_rows.json"))
    save_json(learned_val_rows, os.path.join(args.output_dir, "learned_case_rows.json"))
    save_json(oracle_val_rows, os.path.join(args.output_dir, "oracle_case_rows.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in rule_val_rows],
        os.path.join(args.output_dir, "rule_predictions.jsonl"),
    )
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in learned_val_rows],
        os.path.join(args.output_dir, "learned_predictions.jsonl"),
    )
    torch.save(
        {
            "model_state_dict": small_gate.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "learned_params": learned_params,
            "rule_params": rule_params,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "learned_gate.pt"),
    )
    write_html(args.output_dir, summary)
    print(json.dumps(summary, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
