import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from xml.sax.saxutils import escape

import torch
from torch.utils.data import DataLoader

from evidence_decoder_experiment import evidence_gap, save_json, save_jsonl, svg_for_comparison
from train_learned_evidence_decoder import (
    CandidateDataset,
    CandidateScorer,
    build_rows,
    candidate_iou,
    extract_candidate_features,
    generate_candidates,
    load_items,
    make_feature_context,
    model_selection_key,
    parse_source_quotas,
    score_rows,
    train_epoch,
)
from train_representation_fusion_selector import curve_similarity, quality_key, score_margin, top_window, window_center, window_len
from train_two_mode_decoder_selector import (
    apply_decoder,
    evaluate_predictions,
    fmt_float,
    fmt_pct,
    global_evidence_features,
    load_decoder_checkpoint,
    mode_features,
    nms_rows,
    safe_div,
    table,
    window_iou,
)


REPRESENTATIONS = ["ms_clap", "adapter"]


def align_items(ms_items, adapter_items):
    adapter_by_qid = {item["qid"]: item for item in adapter_items}
    pairs = []
    for ms_item in ms_items:
        qid = ms_item["qid"]
        if qid in adapter_by_qid:
            pairs.append((ms_item, adapter_by_qid[qid]))
    return pairs


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(value)))


def center_delta(a, b, duration):
    return safe_div(abs(window_center(a) - window_center(b)), duration)


def length_delta(a, b, duration):
    return safe_div(abs(window_len(a) - window_len(b)), duration)


def top_score(window):
    return float(window[2]) if window and len(window) > 2 else 0.0


def representation_one_hot(name):
    return [1.0 if name == rep else 0.0 for rep in REPRESENTATIONS]


def pair_context_features(ms_item, adapter_item, ms_record, adapter_record):
    scores = [float(value) for value in ms_item.get("evidence_scores", [])]
    duration = max(float(ms_item.get("duration", len(scores))), float(len(scores)), 1.0)
    ms_top = top_window(ms_record)
    adapter_top = top_window(adapter_record)
    cross_top = [
        top_score(adapter_top) - top_score(ms_top),
        score_margin(adapter_record) - score_margin(ms_record),
        center_delta(ms_top, adapter_top, duration),
        length_delta(ms_top, adapter_top, duration),
        window_iou(ms_top, adapter_top),
        evidence_gap(ms_item),
        evidence_gap(adapter_item),
        evidence_gap(adapter_item) - evidence_gap(ms_item),
    ]
    return (
        global_evidence_features(ms_item)
        + global_evidence_features(adapter_item)
        + mode_features(ms_record, duration)
        + mode_features(adapter_record, duration)
        + curve_similarity(ms_item, adapter_item)
        + cross_top
    )


def candidate_cross_features(candidate, representation, ms_record, adapter_record, duration):
    window = candidate["window"]
    ms_top = top_window(ms_record)
    adapter_top = top_window(adapter_record)
    own_top = ms_top if representation == "ms_clap" else adapter_top
    other_top = adapter_top if representation == "ms_clap" else ms_top
    return (
        representation_one_hot(representation)
        + [
            window_iou(window, ms_top),
            window_iou(window, adapter_top),
            window_iou(window, own_top),
            window_iou(window, other_top),
            center_delta(window, ms_top, duration),
            center_delta(window, adapter_top, duration),
            center_delta(window, own_top, duration),
            center_delta(window, other_top, duration),
            length_delta(window, ms_top, duration),
            length_delta(window, adapter_top, duration),
            length_delta(window, own_top, duration),
            length_delta(window, other_top, duration),
            top_score(window) - top_score(ms_top),
            top_score(window) - top_score(adapter_top),
            top_score(window) - top_score(own_top),
            top_score(window) - top_score(other_top),
            float(window_iou(window, ms_top) >= 0.7),
            float(window_iou(window, adapter_top) >= 0.7),
            float(window_iou(window, own_top) >= 0.7),
            float(window_iou(window, other_top) >= 0.7),
        ]
    )


def risk_aware_candidate_target(raw_iou, ms_record):
    target = float(raw_iou)
    if raw_iou >= 0.7:
        target += 0.25
    elif raw_iou >= 0.5:
        target += 0.10
    elif raw_iou < 0.1:
        target -= 0.05
    if ms_record.get("category") == "semantic_miss" and raw_iou >= 0.5:
        target += 0.10
    if float(ms_record.get("top1_iou", 0.0)) >= 0.7 and raw_iou < 0.7:
        target -= 0.25
    return clamp(target)


def build_fusion_rows(
    ms_items,
    adapter_items,
    ms_records,
    adapter_records,
    topn_per_source=2,
    feature_version="shape_v2",
    source_quotas=None,
):
    rows = []
    for ms_item, adapter_item in align_items(ms_items, adapter_items):
        qid = ms_item["qid"]
        if qid not in ms_records or qid not in adapter_records:
            continue
        scores = [float(value) for value in ms_item.get("evidence_scores", [])]
        duration = max(float(ms_item.get("duration", len(scores))), float(len(scores)), 1.0)
        shared_features = pair_context_features(ms_item, adapter_item, ms_records[qid], adapter_records[qid])
        for representation, item in [("ms_clap", ms_item), ("adapter", adapter_item)]:
            context = make_feature_context(item)
            candidates = generate_candidates(item, topn_per_source=topn_per_source, source_quotas=source_quotas)
            for candidate in candidates:
                raw_iou = candidate_iou(candidate["window"], ms_item["gt_windows"])
                features = (
                    extract_candidate_features(item, candidate, feature_version=feature_version, context=context)
                    + shared_features
                    + candidate_cross_features(candidate, representation, ms_records[qid], adapter_records[qid], duration)
                )
                rows.append(
                    {
                        "qid": qid,
                        "query": ms_item.get("query", ""),
                        "vid": ms_item.get("vid", ""),
                        "representation": representation,
                        "candidate_source": candidate["source"],
                        "source_rank": candidate["source_rank"],
                        "window": candidate["window"],
                        "features": features,
                        "target_raw_iou": raw_iou,
                        "target_iou": risk_aware_candidate_target(raw_iou, ms_records[qid]),
                    }
                )
    return rows


def scored_rows_to_records(items, scored_rows, ms_records, adapter_records, topn=10):
    grouped = defaultdict(list)
    for row in scored_rows:
        grouped[row["qid"]].append(row)

    pred_by_qid = {}
    rows = []
    for item in items:
        qid = item["qid"]
        ranked_rows = sorted(grouped.get(qid, []), key=lambda row: row.get("pred_quality", 0.0), reverse=True)
        kept_rows = nms_rows(ranked_rows, topn=topn, threshold=0.7)
        windows = [row["window"] for row in kept_rows]
        top = kept_rows[0] if kept_rows else {}
        ms_top = top_window(ms_records[qid])
        pred_by_qid[qid] = windows
        rows.append(
            {
                "qid": qid,
                "query": item.get("query", ""),
                "vid": item.get("vid", ""),
                "chosen_representation": top.get("representation", ""),
                "candidate_source": top.get("candidate_source", ""),
                "candidate_source_rank": int(top.get("source_rank", -1)) if top else -1,
                "changed_from_ms": bool(windows and window_iou(windows[0], ms_top) < 0.95),
                "ms_top1_iou": float(ms_records[qid].get("top1_iou", 0.0)),
                "adapter_top1_iou": float(adapter_records[qid].get("top1_iou", 0.0)),
                "ms_category": ms_records[qid].get("category", ""),
                "adapter_category": adapter_records[qid].get("category", ""),
                "windows": windows,
                "top_rows": kept_rows[:5],
                "raw_candidate_count": len(ranked_rows),
            }
        )

    metrics, case_rows = evaluate_predictions(items, pred_by_qid, topn=topn)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, rows


def oracle_from_rows(items, rows, ms_records, adapter_records, topn=10, representation=None):
    scored_rows = []
    for row in rows:
        if representation is not None and row.get("representation") != representation:
            continue
        item = dict(row)
        item["pred_quality"] = float(row.get("target_raw_iou", 0.0))
        scored_rows.append(item)
    return scored_rows_to_records(items, scored_rows, ms_records, adapter_records, topn=topn)


def apply_oracle_budget(items, oracle_rows, ms_records, budget=0.2):
    oracle_by_qid = {row["qid"]: row for row in oracle_rows}
    candidates = []
    for item in items:
        qid = item["qid"]
        if qid not in oracle_by_qid or qid not in ms_records:
            continue
        oracle_key = quality_key(oracle_by_qid[qid])
        ms_key = quality_key(ms_records[qid])
        if oracle_key > ms_key:
            candidates.append((qid, float(oracle_by_qid[qid].get("top1_iou", 0.0)) - float(ms_records[qid].get("top1_iou", 0.0))))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    keep = set(qid for qid, _ in candidates[: int(math.ceil(float(budget) * len(items)))])

    pred_by_qid = {}
    rows = []
    for item in items:
        qid = item["qid"]
        use_oracle = qid in keep
        chosen = oracle_by_qid[qid] if use_oracle else ms_records[qid]
        pred_by_qid[qid] = chosen.get("windows", [])
        rows.append(
            {
                "qid": qid,
                "chosen_representation": "merged_oracle" if use_oracle else "ms_clap",
                "changed_from_ms": use_oracle,
                "ms_top1_iou": float(ms_records[qid].get("top1_iou", 0.0)),
                "adapter_top1_iou": 0.0,
                "ms_category": ms_records[qid].get("category", ""),
                "adapter_category": "",
                "windows": chosen.get("windows", []),
            }
        )
    metrics, case_rows = evaluate_predictions(items, pred_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, rows


def candidate_confidence(row):
    top_rows = row.get("top_rows", [])
    if not top_rows:
        return 0.0
    top = float(top_rows[0].get("pred_quality", 0.0))
    second = float(top_rows[1].get("pred_quality", 0.0)) if len(top_rows) > 1 else 0.0
    return top + 0.25 * (top - second)


def candidate_margin(row):
    top_rows = row.get("top_rows", [])
    if len(top_rows) < 2:
        return float(top_rows[0].get("pred_quality", 0.0)) if top_rows else 0.0
    return float(top_rows[0].get("pred_quality", 0.0)) - float(top_rows[1].get("pred_quality", 0.0))


def apply_candidate_gate(items, fusion_rows, ms_records, params):
    fusion_by_qid = {row["qid"]: row for row in fusion_rows}
    candidates = []
    for item in items:
        qid = item["qid"]
        row = fusion_by_qid.get(qid)
        if not row or not row.get("changed_from_ms"):
            continue
        confidence = candidate_confidence(row)
        margin = candidate_margin(row)
        if confidence >= params["min_confidence"] and margin >= params["min_margin"]:
            score = confidence + params["margin_alpha"] * margin
            candidates.append((qid, score))
    candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    budget = int(math.ceil(float(params["budget"]) * len(items)))
    keep = set(qid for qid, _ in candidates[:budget])

    pred_by_qid = {}
    rows = []
    for item in items:
        qid = item["qid"]
        use_fusion = qid in keep
        chosen = fusion_by_qid[qid] if use_fusion else ms_records[qid]
        pred_by_qid[qid] = chosen.get("windows", [])
        top_rows = chosen.get("top_rows", []) if use_fusion else []
        rows.append(
            {
                "qid": qid,
                "query": item.get("query", ""),
                "vid": item.get("vid", ""),
                "chosen_representation": chosen.get("chosen_representation", "ms_clap") if use_fusion else "ms_clap",
                "candidate_source": chosen.get("candidate_source", "") if use_fusion else "",
                "candidate_source_rank": chosen.get("candidate_source_rank", -1) if use_fusion else -1,
                "changed_from_ms": use_fusion,
                "candidate_confidence": candidate_confidence(chosen) if use_fusion else 0.0,
                "candidate_margin": candidate_margin(chosen) if use_fusion else 0.0,
                "ms_top1_iou": float(ms_records[qid].get("top1_iou", 0.0)),
                "adapter_top1_iou": float(fusion_by_qid[qid].get("adapter_top1_iou", 0.0)),
                "ms_category": ms_records[qid].get("category", ""),
                "adapter_category": fusion_by_qid[qid].get("adapter_category", ""),
                "windows": chosen.get("windows", []),
                "top_rows": top_rows,
            }
        )

    metrics, case_rows = evaluate_predictions(items, pred_by_qid)
    case_by_qid = {row["qid"]: row for row in case_rows}
    for row in rows:
        row.update(case_by_qid.get(row["qid"], {}))
    return metrics, rows


def tune_candidate_gate(items, fusion_rows, ms_records, objective):
    best = None
    for min_confidence in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for min_margin in [-0.05, 0.0, 0.02, 0.05, 0.08, 0.10]:
            for budget in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]:
                params = {
                    "min_confidence": min_confidence,
                    "min_margin": min_margin,
                    "margin_alpha": 0.25,
                    "budget": budget,
                }
                metrics, rows = apply_candidate_gate(items, fusion_rows, ms_records, params)
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
                elif objective == "precision":
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
                else:
                    raise ValueError(f"unknown objective: {objective}")
                if best is None or key > best["key"]:
                    best = {"params": params, "metrics": metrics, "rows": rows, "summary": summary, "key": key}
    return best


def summarize_interventions(rows):
    changed = [row for row in rows if row.get("changed_from_ms")]
    improved = 0
    regressed = 0
    same = 0
    recovered_semantic = 0
    good_regressed = 0
    adapter_top1 = 0
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
        if row.get("chosen_representation") == "adapter":
            adapter_top1 += 1
    return {
        "changed": len(changed),
        "improved": improved,
        "regressed": regressed,
        "same": same,
        "recovered_semantic_miss": recovered_semantic,
        "good_regressed": good_regressed,
        "adapter_top1_changed": adapter_top1,
        "intervention_precision": safe_div(improved, len(changed)),
    }


def mode_counts(rows):
    return dict(Counter(row.get("chosen_representation", "") for row in rows))


def candidate_source_counts(rows):
    return dict(Counter(row.get("candidate_source", "") for row in rows if row.get("candidate_source")))


def write_case_svgs(output_dir, items_by_qid, ms_records, fusion_rows, max_items=12):
    current_by_qid = ms_records
    improvements = []
    regressions = []
    for row in fusion_rows:
        qid = row["qid"]
        current = current_by_qid[qid]
        delta = float(row.get("top1_iou", 0.0)) - float(current.get("top1_iou", 0.0))
        record = (delta, qid, current, row)
        if delta > 0.2:
            improvements.append(record)
        elif delta < -0.2:
            regressions.append(record)

    svg_root = os.path.join(output_dir, "comparison_svgs")
    for folder, records, reverse in [("improvements", improvements, True), ("regressions", regressions, False)]:
        folder_path = os.path.join(svg_root, folder)
        os.makedirs(folder_path, exist_ok=True)
        for idx, (_, qid, current, fusion) in enumerate(sorted(records, key=lambda x: x[0], reverse=reverse)[:max_items]):
            with open(os.path.join(folder_path, f"{idx:03d}_{qid}.svg"), "w", encoding="utf-8") as f:
                f.write(svg_for_comparison(items_by_qid[qid], current["windows"], fusion["windows"]))


def image_tags(output_dir, folder):
    folder_path = os.path.join(output_dir, "comparison_svgs", folder)
    if not os.path.isdir(folder_path):
        return "<p>No cases.</p>"
    tags = []
    for name in sorted(os.listdir(folder_path)):
        tags.append(f'<img src="comparison_svgs/{folder}/{escape(name)}" alt="{escape(name)}"/>')
    return "".join(tags) if tags else "<p>No cases.</p>"


def load_reference_metrics(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    return {
        "representation_fusion_v2_balanced_ref": metrics.get("risk_aware_balanced_gate"),
        "whole_prediction_oracle_ref": metrics.get("oracle_ms_or_adapter"),
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
  <title>Candidate-Level Representation Fusion V1</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #0f766e; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #e8f3f1; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    img {{ width: 100%; max-width: 1180px; display: block; margin: 14px 0; border: 1px solid #ddd; background: white; }}
  </style>
</head>
<body>
  <h1>Candidate-Level Representation Fusion V1</h1>
  <div class="note">
    This experiment merges MS-CLAP and temporal-adapter candidate lists, then trains a risk-aware candidate reranker.
    Candidate target is IoU with threshold bonuses and penalties for regressing reliable MS-CLAP cases.
  </div>
  <h2>Validation Metrics</h2>
  {table(["System", "R1@0.5", "R1@0.7", "R3@0.7", "R5@0.7", "Top1 IoU", "Best IoU@5", "Semantic Miss", "Good"], metric_rows)}
  <h2>Changes Compared With MS-CLAP Baseline</h2>
  {table(["System", "Changed", "Improved", "Regressed", "Recovered Semantic Miss", "Good Regressed", "Adapter Top1", "Precision"], intervention_rows)}
  <h2>Learned Fusion Improvements Over MS-CLAP</h2>
  <p>Green = GT, red = MS-CLAP baseline top1, purple = candidate fusion top1, blue = MS evidence curve.</p>
  {image_tags(output_dir, "improvements")}
  <h2>Learned Fusion Regressions Compared With MS-CLAP</h2>
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
    parser.add_argument("--reference_summary_path", default="")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lambda_pairwise", type=float, default=0.25)
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
    print("train candidates:", len(train_rows))
    print("val candidates:", len(val_rows))
    print("feature dim:", len(train_rows[0]["features"]) if train_rows else 0)

    train_dataset = CandidateDataset(train_rows)
    val_dataset = CandidateDataset(val_rows, feature_mean=train_dataset.feature_mean, feature_std=train_dataset.feature_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = CandidateScorer(train_dataset.features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)

    best_metrics = None
    best_epoch = 0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.lambda_pairwise)
        scored_val = score_rows(model, val_dataset, val_rows, device)
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
    scored_train = score_rows(model, train_dataset, train_rows, device)
    train_fusion_metrics, train_fusion_rows = scored_rows_to_records(
        train_ms_items,
        scored_train,
        train_ms_records,
        train_adapter_records,
    )
    scored_val = score_rows(model, val_dataset, val_rows, device)
    fusion_metrics, fusion_rows = scored_rows_to_records(val_ms_items, scored_val, val_ms_records, val_adapter_records)

    balanced_gate_train = tune_candidate_gate(train_ms_items, train_fusion_rows, train_ms_records, objective="balanced")
    precision_gate_train = tune_candidate_gate(train_ms_items, train_fusion_rows, train_ms_records, objective="precision")
    if precision_gate_train is None:
        precision_gate_train = balanced_gate_train
    balanced_gate_metrics, balanced_gate_rows = apply_candidate_gate(
        val_ms_items,
        fusion_rows,
        val_ms_records,
        balanced_gate_train["params"],
    )
    precision_gate_metrics, precision_gate_rows = apply_candidate_gate(
        val_ms_items,
        fusion_rows,
        val_ms_records,
        precision_gate_train["params"],
    )

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
        "candidate_level_fusion_v1": fusion_metrics,
        "candidate_level_balanced_gate_v1": balanced_gate_metrics,
        "candidate_level_precision_gate_v1": precision_gate_metrics,
        "ms_candidate_oracle": ms_oracle_metrics,
        "adapter_candidate_oracle": adapter_oracle_metrics,
        "merged_candidate_oracle": merged_oracle_metrics,
        "merged_candidate_oracle_20pct_budget": budget_oracle_metrics,
    }
    metrics.update(load_reference_metrics(args.reference_summary_path))

    interventions = {
        "candidate_level_fusion_v1": summarize_interventions(fusion_rows),
        "candidate_level_balanced_gate_v1": summarize_interventions(balanced_gate_rows),
        "candidate_level_precision_gate_v1": summarize_interventions(precision_gate_rows),
        "merged_candidate_oracle": summarize_interventions(merged_oracle_rows),
        "merged_candidate_oracle_20pct_budget": summarize_interventions(budget_oracle_rows),
    }
    summary = {
        "best_epoch": best_epoch,
        "balanced_gate_params": balanced_gate_train["params"],
        "precision_gate_params": precision_gate_train["params"],
        "train_candidates": len(train_rows),
        "val_candidates": len(val_rows),
        "feature_dim": int(train_dataset.features.shape[1]),
        "metrics": metrics,
        "interventions": interventions,
        "mode_counts": {
            "candidate_level_fusion_v1": mode_counts(fusion_rows),
            "candidate_level_balanced_gate_v1": mode_counts(balanced_gate_rows),
            "candidate_level_precision_gate_v1": mode_counts(precision_gate_rows),
            "merged_candidate_oracle": mode_counts(merged_oracle_rows),
            "merged_candidate_oracle_20pct_budget": mode_counts(budget_oracle_rows),
        },
        "candidate_source_counts": {
            "candidate_level_fusion_v1": candidate_source_counts(fusion_rows),
            "candidate_level_balanced_gate_v1": candidate_source_counts(balanced_gate_rows),
            "candidate_level_precision_gate_v1": candidate_source_counts(precision_gate_rows),
            "merged_candidate_oracle": candidate_source_counts(merged_oracle_rows),
        },
        "target_summary": {
            "train_positive_07": sum(1 for row in train_rows if row["target_raw_iou"] >= 0.7),
            "train_positive_05": sum(1 for row in train_rows if row["target_raw_iou"] >= 0.5),
            "val_positive_07": sum(1 for row in val_rows if row["target_raw_iou"] >= 0.7),
            "val_positive_05": sum(1 for row in val_rows if row["target_raw_iou"] >= 0.5),
        },
        "args": vars(args),
    }

    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(fusion_rows, os.path.join(args.output_dir, "fusion_case_rows.json"))
    save_json(balanced_gate_rows, os.path.join(args.output_dir, "balanced_gate_case_rows.json"))
    save_json(precision_gate_rows, os.path.join(args.output_dir, "precision_gate_case_rows.json"))
    save_json(merged_oracle_rows, os.path.join(args.output_dir, "merged_oracle_case_rows.json"))
    save_json(budget_oracle_rows, os.path.join(args.output_dir, "merged_oracle_budget_case_rows.json"))
    save_jsonl(
        [
            {
                "qid": row["qid"],
                "pred_relevant_windows": row.get("windows", []),
                "chosen_representation": row.get("chosen_representation", ""),
            }
            for row in fusion_rows
        ],
        os.path.join(args.output_dir, "fusion_predictions.jsonl"),
    )
    write_case_svgs(args.output_dir, {item["qid"]: item for item in val_ms_items}, val_ms_records, fusion_rows)
    write_html(args.output_dir, summary)
    print(json.dumps({"best_epoch": best_epoch, "fusion_metrics": fusion_metrics}, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
