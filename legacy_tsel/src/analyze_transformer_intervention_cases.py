import argparse
import json
import math
import os
from collections import Counter, defaultdict
from xml.sax.saxutils import escape

from train_conservative_transformer_gate import gate_stats_for_item, quality_key, window_center, window_len
from train_two_mode_decoder_selector import fmt_float, fmt_pct, safe_div, table, window_iou


FEATURES = [
    "two_mode_uncertainty",
    "default_confidence",
    "cov_pre_iou",
    "tr_default_iou",
    "tr_cov_iou",
    "tr_pre_iou",
    "tr_best_agreement",
    "tr_margin",
    "default_margin",
    "score_advantage",
    "tr_support",
    "center_delta",
    "length_delta",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def mean(values):
    return sum(values) / len(values) if values else 0.0


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


def rows_by_qid(rows):
    return {row["qid"]: row for row in rows}


def best_window(row):
    windows = row.get("windows", [])
    return windows[0] if windows else [0.0, 0.0, 0.0]


def relation(default, transformer):
    default_key = quality_key(default)
    transformer_key = quality_key(transformer)
    if transformer_key > default_key:
        return "transformer_better"
    if transformer_key < default_key:
        return "default_better"
    return "tie"


def top1_relation(default, transformer):
    delta = float(transformer.get("top1_iou", 0.0)) - float(default.get("top1_iou", 0.0))
    if delta > 1e-6:
        return "top1_improved"
    if delta < -1e-6:
        return "top1_regressed"
    return "top1_same"


def improvement_type(default, transformer):
    default_cat = default.get("category", "")
    tr_cat = transformer.get("category", "")
    default_iou = float(default.get("top1_iou", 0.0))
    tr_iou = float(transformer.get("top1_iou", 0.0))
    top5_delta = float(transformer.get("top5_iou", 0.0)) - float(default.get("top5_iou", 0.0))

    if default_cat == "semantic_miss" and tr_cat != "semantic_miss":
        return "semantic_recovery"
    if default_cat != "semantic_miss" and tr_cat == "semantic_miss":
        return "semantic_jump_harm"
    if default_cat == "good" and tr_cat != "good":
        return "good_regression"
    if default_cat != "good" and tr_cat == "good":
        return "to_good"
    if default_cat == "candidate_exists" and tr_iou > default_iou:
        return "candidate_to_top1"
    if default_cat == "boundary_error" and tr_iou > default_iou + 0.1:
        return "boundary_tightening"
    if default_cat == "good" and tr_cat == "good" and tr_iou > default_iou:
        return "good_refinement"
    if tr_iou < default_iou - 0.1:
        return "boundary_or_event_regression"
    if top5_delta > 0.1 and tr_iou <= default_iou:
        return "topk_only_gain"
    return "small_or_mixed_change"


def build_case_rows(items, mode_records):
    evidence_by_qid = {item["qid"]: item for item in items}
    records_by_mode = {mode: rows_by_qid(rows) for mode, rows in mode_records.items()}
    case_rows = []
    for qid, item in evidence_by_qid.items():
        records = {
            mode: records_by_mode[mode][qid]
            for mode in ["coverage", "precision", "transformer", "default"]
        }
        stats = gate_stats_for_item(item, records)
        default = records["default"]
        transformer = records["transformer"]
        row = {
            "qid": qid,
            "query": item.get("query", default.get("query", "")),
            "vid": item.get("vid", default.get("vid", "")),
            "gt_windows": item.get("gt_windows", default.get("gt_windows", [])),
            "relation": relation(default, transformer),
            "top1_relation": top1_relation(default, transformer),
            "change_type": improvement_type(default, transformer),
            "default_category": default.get("category", ""),
            "transformer_category": transformer.get("category", ""),
            "coverage_category": records["coverage"].get("category", ""),
            "precision_category": records["precision"].get("category", ""),
            "default_top1_iou": float(default.get("top1_iou", 0.0)),
            "transformer_top1_iou": float(transformer.get("top1_iou", 0.0)),
            "top1_delta": float(transformer.get("top1_iou", 0.0)) - float(default.get("top1_iou", 0.0)),
            "default_top5_iou": float(default.get("top5_iou", 0.0)),
            "transformer_top5_iou": float(transformer.get("top5_iou", 0.0)),
            "top5_delta": float(transformer.get("top5_iou", 0.0)) - float(default.get("top5_iou", 0.0)),
            "default_window": best_window(default),
            "transformer_window": best_window(transformer),
            "coverage_window": best_window(records["coverage"]),
            "precision_window": best_window(records["precision"]),
            "stats": stats,
        }
        row["default_transformer_iou"] = window_iou(row["default_window"], row["transformer_window"])
        row["coverage_precision_iou"] = window_iou(row["coverage_window"], row["precision_window"])
        row["center_delta_seconds"] = abs(window_center(row["default_window"]) - window_center(row["transformer_window"]))
        row["length_delta_seconds"] = abs(window_len(row["default_window"]) - window_len(row["transformer_window"]))
        case_rows.append(row)
    return case_rows


def summarize_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["relation"]].append(row)
    summary = {
        "num_samples": len(rows),
        "relation_counts": dict(Counter(row["relation"] for row in rows)),
        "top1_relation_counts": dict(Counter(row["top1_relation"] for row in rows)),
        "change_type_counts": dict(Counter(row["change_type"] for row in rows)),
        "transition_counts": dict(Counter(f"{row['default_category']} -> {row['transformer_category']}" for row in rows)),
        "group_stats": {},
    }
    for name, group_rows in groups.items():
        summary["group_stats"][name] = {
            "count": len(group_rows),
            "avg_top1_delta": mean([row["top1_delta"] for row in group_rows]),
            "avg_top5_delta": mean([row["top5_delta"] for row in group_rows]),
            "strict_crossings": sum(1 for row in group_rows if row["default_top1_iou"] < 0.7 <= row["transformer_top1_iou"]),
            "loose_crossings": sum(1 for row in group_rows if row["default_top1_iou"] < 0.5 <= row["transformer_top1_iou"]),
            "semantic_recoveries": sum(1 for row in group_rows if row["change_type"] == "semantic_recovery"),
            "good_regressions": sum(1 for row in group_rows if row["change_type"] == "good_regression"),
        }
        for feature in FEATURES:
            values = [float(row["stats"].get(feature, 0.0)) for row in group_rows]
            summary["group_stats"][name][f"avg_{feature}"] = mean(values)
            summary["group_stats"][name][f"p50_{feature}"] = percentile(values, 50)
    return summary


def rule_probe(rows):
    probes = []
    labels = [1 if row["relation"] == "transformer_better" else 0 for row in rows]
    total_pos = sum(labels)
    for feature in FEATURES:
        values = [float(row["stats"].get(feature, 0.0)) for row in rows]
        thresholds = sorted(set(percentile(values, pct) for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]))
        for threshold in thresholds:
            for direction in [">=", "<="]:
                selected = []
                for row, label, value in zip(rows, labels, values):
                    ok = value >= threshold if direction == ">=" else value <= threshold
                    if ok:
                        selected.append(label)
                count = len(selected)
                if count < 10 or count > 120:
                    continue
                hits = sum(selected)
                precision = safe_div(hits, count)
                recall = safe_div(hits, total_pos)
                f1 = safe_div(2 * precision * recall, precision + recall)
                lift = safe_div(precision, safe_div(total_pos, len(rows)))
                probes.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "count": count,
                        "hits": hits,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "lift": lift,
                    }
                )
    return sorted(probes, key=lambda row: (row["precision"], row["hits"], row["f1"]), reverse=True)


def combo_rule_probe(rows):
    labels = [1 if row["relation"] == "transformer_better" else 0 for row in rows]
    total_pos = sum(labels)
    base_rate = safe_div(total_pos, len(rows))
    candidates = [
        ("tr_best_agreement", ">=", 0.2),
        ("tr_best_agreement", ">=", 0.4),
        ("tr_best_agreement", ">=", 0.6),
        ("two_mode_uncertainty", ">=", 0.4),
        ("two_mode_uncertainty", ">=", 0.6),
        ("cov_pre_iou", "<=", 0.4),
        ("cov_pre_iou", "<=", 0.6),
        ("tr_margin", ">=", 0.0),
        ("tr_support", ">=", 0.3),
        ("center_delta", "<=", 0.4),
        ("score_advantage", ">=", 0.0),
    ]
    probes = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            selected = []
            for row, label in zip(rows, labels):
                ok = True
                for feature, direction, threshold in [first, second]:
                    value = float(row["stats"].get(feature, 0.0))
                    ok = ok and (value >= threshold if direction == ">=" else value <= threshold)
                if ok:
                    selected.append(label)
            count = len(selected)
            if count < 10 or count > 120:
                continue
            hits = sum(selected)
            precision = safe_div(hits, count)
            recall = safe_div(hits, total_pos)
            probes.append(
                {
                    "rule": f"{first[0]} {first[1]} {first[2]} AND {second[0]} {second[1]} {second[2]}",
                    "count": count,
                    "hits": hits,
                    "precision": precision,
                    "recall": recall,
                    "lift": safe_div(precision, base_rate),
                }
            )
    return sorted(probes, key=lambda row: (row["precision"], row["hits"], row["recall"]), reverse=True)


def fmt_window(window):
    if not window:
        return ""
    score = f", {float(window[2]):.3f}" if len(window) > 2 else ""
    return f"[{float(window[0]):.1f}, {float(window[1]):.1f}{score}]"


def case_table(rows, title, limit=16):
    body_rows = []
    for row in rows[:limit]:
        body_rows.append(
            [
                escape(row["qid"]),
                escape(row["query"][:110]),
                escape(row["change_type"]),
                escape(f"{row['default_category']} -> {row['transformer_category']}"),
                fmt_float(row["default_top1_iou"]),
                fmt_float(row["transformer_top1_iou"]),
                fmt_float(row["top1_delta"]),
                escape(fmt_window(row["default_window"])),
                escape(fmt_window(row["transformer_window"])),
            ]
        )
    return f"<h2>{escape(title)}</h2>" + table(
        ["QID", "Query", "Type", "Category", "Default IoU", "Transformer IoU", "Delta", "Default Window", "Transformer Window"],
        body_rows,
    )


def write_html(output_dir, summary, rows, probes, combo_probes):
    relation_rows = []
    for name in ["transformer_better", "default_better", "tie"]:
        stats = summary["group_stats"].get(name, {})
        relation_rows.append(
            [
                escape(name),
                str(stats.get("count", 0)),
                fmt_float(stats.get("avg_top1_delta", 0.0)),
                fmt_float(stats.get("avg_top5_delta", 0.0)),
                str(stats.get("strict_crossings", 0)),
                str(stats.get("loose_crossings", 0)),
                fmt_float(stats.get("avg_two_mode_uncertainty", 0.0)),
                fmt_float(stats.get("avg_tr_best_agreement", 0.0)),
                fmt_float(stats.get("avg_tr_margin", 0.0)),
                fmt_float(stats.get("avg_center_delta", 0.0)),
            ]
        )

    change_rows = [[escape(key), str(value)] for key, value in sorted(summary["change_type_counts"].items(), key=lambda x: (-x[1], x[0]))]
    transition_rows = [[escape(key), str(value)] for key, value in sorted(summary["transition_counts"].items(), key=lambda x: (-x[1], x[0]))[:18]]
    probe_rows = [
        [
            escape(row["feature"]),
            escape(row["direction"]),
            fmt_float(row["threshold"]),
            str(row["count"]),
            str(row["hits"]),
            fmt_pct(row["precision"]),
            fmt_pct(row["recall"]),
            fmt_float(row["lift"]),
        ]
        for row in probes[:14]
    ]
    combo_rows = [
        [
            escape(row["rule"]),
            str(row["count"]),
            str(row["hits"]),
            fmt_pct(row["precision"]),
            fmt_pct(row["recall"]),
            fmt_float(row["lift"]),
        ]
        for row in combo_probes[:12]
    ]

    useful = sorted(
        [row for row in rows if row["relation"] == "transformer_better"],
        key=lambda row: (row["top1_delta"], row["transformer_top1_iou"]),
        reverse=True,
    )
    dangerous = sorted(
        [row for row in rows if row["relation"] == "default_better"],
        key=lambda row: (row["top1_delta"], -row["default_top1_iou"]),
    )
    semantic = [row for row in useful if row["change_type"] == "semantic_recovery"]

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Transformer Intervention Case Analysis V5.2</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; background: #f7f7f5; }}
    .note {{ background: white; border-left: 4px solid #155e75; padding: 12px 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #e7eef2; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Transformer Intervention Case Analysis V5.2</h1>
  <div class="note">
    This report compares the two-mode default with the candidate-context Transformer on the same validation items.
    Transformer is better on <b>{summary["relation_counts"].get("transformer_better", 0)}</b> cases, default is better on
    <b>{summary["relation_counts"].get("default_better", 0)}</b> cases, and they tie on
    <b>{summary["relation_counts"].get("tie", 0)}</b> cases.
  </div>
  <h2>Group Signatures</h2>
  {table(["Group", "Count", "Avg Top1 Delta", "Avg Top5 Delta", "Strict Cross", "Loose Cross", "Uncertainty", "Tr Agreement", "Tr Margin", "Center Delta"], relation_rows)}
  <h2>Change Types</h2>
  {table(["Type", "Count"], change_rows)}
  <h2>Category Transitions</h2>
  {table(["Default -> Transformer", "Count"], transition_rows)}
  <h2>Single-Feature Rule Probes</h2>
  {table(["Feature", "Dir", "Threshold", "Selected", "Hits", "Precision", "Recall", "Lift"], probe_rows)}
  <h2>Two-Condition Rule Probes</h2>
  {table(["Rule", "Selected", "Hits", "Precision", "Recall", "Lift"], combo_rows)}
  {case_table(semantic, "Semantic Recovery Examples", limit=10)}
  {case_table(useful, "Largest Transformer Gains", limit=16)}
  {case_table(dangerous, "Largest Transformer Dangers", limit=16)}
</body>
</html>"""
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--mode_records_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    items = load_json(args.evidence_path)
    mode_records = load_json(args.mode_records_path)
    rows = build_case_rows(items, mode_records)
    summary = summarize_groups(rows)
    probes = rule_probe(rows)
    combo_probes = combo_rule_probe(rows)
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_json(rows, os.path.join(args.output_dir, "case_rows.json"))
    save_json(probes, os.path.join(args.output_dir, "single_feature_rule_probes.json"))
    save_json(combo_probes, os.path.join(args.output_dir, "combo_rule_probes.json"))
    write_html(args.output_dir, summary, rows, probes, combo_probes)
    print(json.dumps(summary, indent=2))
    print("saved report to", os.path.join(args.output_dir, "report.html"))


if __name__ == "__main__":
    main()
