import argparse
import json
import os
import random
from collections import Counter
from xml.sax.saxutils import escape

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_candidate_transformer_decoder import CandidateContextTransformer, CandidateSetDataset, score_dataset
from train_learned_evidence_decoder import build_rows, parse_source_quotas
from train_two_mode_decoder_selector import (
    apply_decoder,
    evaluate_predictions,
    fmt_float,
    fmt_pct,
    global_evidence_features,
    load_decoder_checkpoint,
    mean,
    mode_features,
    nms_rows,
    safe_div,
    save_json,
    save_jsonl,
    table,
    window_iou,
)


MODE_NAMES = ["coverage", "precision", "transformer"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transformer_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    feature_mean = ckpt["feature_mean"].cpu()
    feature_std = ckpt["feature_std"].cpu()
    model = CandidateContextTransformer(
        input_dim=int(feature_mean.numel()),
        hidden_dim=int(args.get("hidden_dim", 128)),
        num_layers=int(args.get("num_layers", 2)),
        num_heads=int(args.get("num_heads", 4)),
        dropout=float(args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "path": path,
        "args": args,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "model": model,
    }


def apply_transformer_decoder(items, decoder, device, mode_name):
    args = decoder["args"]
    source_quotas = parse_source_quotas(args.get("source_quotas", ""))
    rows = build_rows(
        items,
        topn_per_source=int(args.get("topn_per_source", 2)),
        feature_version=args.get("feature_version", "shape_v2"),
        source_quotas=source_quotas,
    )
    dataset = CandidateSetDataset(
        items,
        rows,
        feature_mean=decoder["feature_mean"],
        feature_std=decoder["feature_std"],
    )
    scored_by_qid = score_dataset(
        decoder["model"],
        dataset,
        device,
        batch_size=int(args.get("batch_size", 64)),
    )

    by_qid = {}
    pred_by_qid = {}
    for item in items:
        qid = item["qid"]
        ranked_rows = sorted(scored_by_qid.get(qid, []), key=lambda row: row["pred_quality"], reverse=True)
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


def top_window(record):
    windows = record.get("windows", [])
    return windows[0] if windows else [0.0, 0.0, 0.0]


def top_source(record):
    rows = record.get("rows", [])
    return rows[0].get("source", "") if rows else ""


def pair_features(left_record, right_record, duration):
    left = top_window(left_record)
    right = top_window(right_record)
    left_len = max(float(left[1]) - float(left[0]), 0.0)
    right_len = max(float(right[1]) - float(right[0]), 0.0)
    left_center = 0.5 * (float(left[0]) + float(left[1]))
    right_center = 0.5 * (float(right[0]) + float(right[1]))
    iou = window_iou(left, right)
    return [
        float(right[2]) - float(left[2]),
        abs(right_center - left_center) / duration,
        abs(right_len - left_len) / duration,
        safe_div(min(left_len, right_len), max(left_len, right_len, 1e-6)),
        iou,
        1.0 if top_source(left_record) == top_source(right_record) and top_source(left_record) else 0.0,
    ]


def selector_feature_row(item, records):
    scores = [float(value) for value in item.get("evidence_scores", [])]
    duration = max(float(item.get("duration", len(scores))), float(len(scores)), 1.0)
    mode_blocks = []
    for name in MODE_NAMES:
        mode_blocks.extend(mode_features(records[name], duration))

    cov_pre = pair_features(records["coverage"], records["precision"], duration)
    cov_tr = pair_features(records["coverage"], records["transformer"], duration)
    pre_tr = pair_features(records["precision"], records["transformer"], duration)
    pair_ious = [cov_pre[4], cov_tr[4], pre_tr[4]]
    pair_block = cov_pre + cov_tr + pre_tr + [max(pair_ious), mean(pair_ious)]
    return global_evidence_features(item) + mode_blocks + pair_block


def mode_quality_key(record):
    top1 = float(record.get("top1_iou", 0.0))
    top5 = float(record.get("top5_iou", 0.0))
    best5 = float(record.get("best_iou_top5", top5))
    return (
        top1 >= 0.7,
        top1 >= 0.5,
        top1,
        best5,
        top5,
    )


def best_mode_index(records):
    # Tie-break toward simpler, already stable modes.
    tie_priority = {"coverage": 2, "precision": 1, "transformer": 0}
    return max(
        range(len(MODE_NAMES)),
        key=lambda idx: mode_quality_key(records[MODE_NAMES[idx]]) + (tie_priority[MODE_NAMES[idx]],),
    )


def build_selector_dataset(items, records_by_mode):
    features = []
    labels = []
    qids = []
    for item in items:
        qid = item["qid"]
        records = {name: records_by_mode[name][qid] for name in MODE_NAMES}
        features.append(selector_feature_row(item, records))
        labels.append(best_mode_index(records))
        qids.append(qid)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), qids


class TransformerAssistedSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(MODE_NAMES)),
        )

    def forward(self, x):
        return self.net(x)


def train_selector(x_train, y_train, epochs, batch_size, lr, seed, hidden_dim, dropout):
    torch.manual_seed(seed)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp(min=1e-6)
    x_norm = (x_train - feature_mean) / feature_std
    dataset = TensorDataset(x_norm, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TransformerAssistedSelector(input_dim=x_train.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = torch.bincount(y_train, minlength=len(MODE_NAMES)).float().clamp(min=1.0)
    class_weights = torch.sqrt(counts.sum() / (len(MODE_NAMES) * counts))
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
    return model, feature_mean, feature_std


@torch.no_grad()
def predict_logits(model, feature_mean, feature_std, x):
    model.eval()
    return model((x - feature_mean) / feature_std).cpu()


def logits_to_choices(logits, biases):
    bias_tensor = torch.tensor(biases, dtype=logits.dtype)
    return torch.argmax(logits + bias_tensor, dim=1).tolist()


def select_predictions(items, records_by_mode, logits, biases):
    choices = logits_to_choices(logits, biases)
    probs = F.softmax(logits, dim=1).tolist()
    pred_by_qid = {}
    selector_rows = []
    for item, choice, prob_row in zip(items, choices, probs):
        qid = item["qid"]
        chosen_mode = MODE_NAMES[int(choice)]
        chosen = records_by_mode[chosen_mode][qid]
        pred_by_qid[qid] = chosen.get("windows", [])
        row = {
            "qid": qid,
            "query": item.get("query", ""),
            "chosen_mode": chosen_mode,
            "mode_probabilities": {name: float(prob_row[idx]) for idx, name in enumerate(MODE_NAMES)},
            "coverage_top1_iou": float(records_by_mode["coverage"][qid].get("top1_iou", 0.0)),
            "precision_top1_iou": float(records_by_mode["precision"][qid].get("top1_iou", 0.0)),
            "transformer_top1_iou": float(records_by_mode["transformer"][qid].get("top1_iou", 0.0)),
            "coverage_category": records_by_mode["coverage"][qid].get("category", ""),
            "precision_category": records_by_mode["precision"][qid].get("category", ""),
            "transformer_category": records_by_mode["transformer"][qid].get("category", ""),
            "windows": chosen.get("windows", []),
        }
        selector_rows.append(row)
    metrics, case_rows = evaluate_predictions(items, pred_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in selector_rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, selector_rows


def metric_key(metrics, biases):
    return (
        float(metrics.get("R1@0.7", 0.0)),
        float(metrics.get("R1@0.5", 0.0)),
        float(metrics.get("top1_iou", 0.0)),
        -float(metrics.get("categories", {}).get("semantic_miss", 0)),
        -abs(float(biases[1])),
        -abs(float(biases[2])),
    )


def tune_biases(items, records_by_mode, logits):
    best_biases = [0.0, 0.0, 0.0]
    best_key = None
    grid = [step / 20.0 for step in range(-24, 25)]
    for precision_bias in grid:
        for transformer_bias in grid:
            biases = [0.0, precision_bias, transformer_bias]
            metrics, _ = select_predictions(items, records_by_mode, logits, biases)
            key = metric_key(metrics, biases)
            if best_key is None or key > best_key:
                best_key = key
                best_biases = biases
    return best_biases


def oracle_select(items, records_by_mode):
    pred_by_qid = {}
    rows = []
    choices = []
    for item in items:
        qid = item["qid"]
        records = {name: records_by_mode[name][qid] for name in MODE_NAMES}
        choice = best_mode_index(records)
        chosen_mode = MODE_NAMES[choice]
        pred_by_qid[qid] = records[chosen_mode].get("windows", [])
        rows.append({"qid": qid, "chosen_mode": chosen_mode})
        choices.append(chosen_mode)
    metrics, case_rows = evaluate_predictions(items, pred_by_qid)
    return metrics, rows, case_rows, Counter(choices)


def load_baseline_metrics(path):
    if not path or not os.path.exists(path):
        return {}
    data = load_json(path)
    metrics = data.get("metrics", {})
    if "two_mode_selector" in metrics:
        return {"two_mode_selector_baseline": metrics["two_mode_selector"]}
    return {}


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
  <title>Transformer-Assisted Decoder Selector</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #0f766e; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #e9efef; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Transformer-Assisted Decoder Selector</h1>
  <div class="note">
    V5 tests whether candidate-context Transformer predictions contain useful auxiliary evidence for choosing between coverage and precision decoding.
    Logit biases tuned on train: coverage={summary["biases"][0]:.2f}, precision={summary["biases"][1]:.2f}, transformer={summary["biases"][2]:.2f}.
  </div>
  <h2>Validation Metrics</h2>
  {table(["Decoder", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Selector Usage</h2>
  {table(["Split", "Coverage", "Precision", "Transformer"], [
      ["train labels", str(summary["train_label_counts"].get("coverage", 0)), str(summary["train_label_counts"].get("precision", 0)), str(summary["train_label_counts"].get("transformer", 0))],
      ["train chosen", str(summary["train_mode_counts"].get("coverage", 0)), str(summary["train_mode_counts"].get("precision", 0)), str(summary["train_mode_counts"].get("transformer", 0))],
      ["val oracle labels", str(summary["val_oracle_counts"].get("coverage", 0)), str(summary["val_oracle_counts"].get("precision", 0)), str(summary["val_oracle_counts"].get("transformer", 0))],
      ["val chosen", str(summary["val_mode_counts"].get("coverage", 0)), str(summary["val_mode_counts"].get("precision", 0)), str(summary["val_mode_counts"].get("transformer", 0))],
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
    parser.add_argument("--transformer_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_summary_path", default="")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.12)
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

    print("applying coverage decoder to train...")
    train_cov, train_cov_metrics = apply_decoder(train_items, coverage, device, "coverage")
    print("applying precision decoder to train...")
    train_pre, train_pre_metrics = apply_decoder(train_items, precision, device, "precision")
    print("applying transformer decoder to train...")
    train_tr, train_tr_metrics = apply_transformer_decoder(train_items, transformer, device, "transformer")
    print("applying coverage decoder to val...")
    val_cov, val_cov_metrics = apply_decoder(val_items, coverage, device, "coverage")
    print("applying precision decoder to val...")
    val_pre, val_pre_metrics = apply_decoder(val_items, precision, device, "precision")
    print("applying transformer decoder to val...")
    val_tr, val_tr_metrics = apply_transformer_decoder(val_items, transformer, device, "transformer")

    train_records = {"coverage": train_cov, "precision": train_pre, "transformer": train_tr}
    val_records = {"coverage": val_cov, "precision": val_pre, "transformer": val_tr}
    x_train, y_train, _ = build_selector_dataset(train_items, train_records)
    x_val, y_val, _ = build_selector_dataset(val_items, val_records)
    train_label_counts = Counter(MODE_NAMES[int(label)] for label in y_train.tolist())
    val_label_counts = Counter(MODE_NAMES[int(label)] for label in y_val.tolist())
    print("selector train labels:", dict(train_label_counts))
    print("selector val oracle labels:", dict(val_label_counts))

    selector, feature_mean, feature_std = train_selector(
        x_train,
        y_train,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
        args.hidden_dim,
        args.dropout,
    )
    train_logits = predict_logits(selector, feature_mean, feature_std, x_train)
    val_logits = predict_logits(selector, feature_mean, feature_std, x_val)
    biases = tune_biases(train_items, train_records, train_logits)
    print("selected train biases:", biases)

    train_sel_metrics, train_sel_rows = select_predictions(train_items, train_records, train_logits, biases)
    val_sel_metrics, val_sel_rows = select_predictions(val_items, val_records, val_logits, biases)
    val_oracle_metrics, val_oracle_choices, val_oracle_rows, val_oracle_counts = oracle_select(val_items, val_records)

    train_mode_counts = Counter(row["chosen_mode"] for row in train_sel_rows)
    val_mode_counts = Counter(row["chosen_mode"] for row in val_sel_rows)
    metrics = {
        "coverage_shape_v2_top2": val_cov_metrics,
        "precision_source_gate_v2": val_pre_metrics,
        "transformer_source_gate": val_tr_metrics,
    }
    metrics.update(load_baseline_metrics(args.baseline_summary_path))
    metrics.update(
        {
            "transformer_assisted_selector_v5": val_sel_metrics,
            "three_mode_oracle": val_oracle_metrics,
        }
    )
    summary = {
        "biases": biases,
        "train_label_counts": dict(train_label_counts),
        "val_label_counts": dict(val_label_counts),
        "train_mode_counts": dict(train_mode_counts),
        "val_mode_counts": dict(val_mode_counts),
        "val_oracle_counts": dict(val_oracle_counts),
        "metrics": metrics,
        "train_metrics": {
            "coverage_shape_v2_top2": train_cov_metrics,
            "precision_source_gate_v2": train_pre_metrics,
            "transformer_source_gate": train_tr_metrics,
            "transformer_assisted_selector_v5": train_sel_metrics,
        },
        "args": vars(args),
    }
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(val_sel_rows, os.path.join(args.output_dir, "selector_case_rows.json"))
    save_json(val_oracle_rows, os.path.join(args.output_dir, "oracle_case_rows.json"))
    save_json(val_oracle_choices, os.path.join(args.output_dir, "oracle_choices.json"))
    save_jsonl(
        [{"qid": row["qid"], "pred_relevant_windows": row.get("windows", [])} for row in val_sel_rows],
        os.path.join(args.output_dir, "selector_predictions.jsonl"),
    )
    torch.save(
        {
            "model_state_dict": selector.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "biases": biases,
            "args": vars(args),
            "mode_names": MODE_NAMES,
        },
        os.path.join(args.output_dir, "selector.pt"),
    )
    write_html(args.output_dir, summary)
    print(json.dumps(summary, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
