#!/usr/bin/env python3
"""Create the v0C temporal over-extension diagnostics on the remote AMR run."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


BASE = Path("/private/research-artifact")
PRED_PATH = BASE / "dcase2026_task6_baseline/results/submission.jsonl"
GT_PATH = BASE / "dcase2026_task6_baseline/data/castella_test_release.jsonl"
OUT_DIR = BASE / "amr_failure_audit/outputs/qd_detr"
CSV_PATH = OUT_DIR / "duration_extension_query.csv"
JSON_PATH = OUT_DIR / "duration_extension_summary.json"
MD_PATH = OUT_DIR / "duration_extension_report.md"

EPS = 1e-9
BIN_LABELS = ["0–2s", "2–5s", "5–10s", "10–20s", "20s+"]
QUANTILES = {"p10": 0.10, "p25": 0.25, "median": 0.50, "p75": 0.75, "p90": 0.90}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def interval(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"invalid {label}: {value!r}")
    start, end = float(value[0]), float(value[1])
    if not (math.isfinite(start) and math.isfinite(end) and end >= start):
        raise ValueError(f"invalid {label}: {value!r}")
    return start, end


def prediction(value: Any, rank: int) -> dict[str, Any]:
    start, end = interval(value, f"prediction rank {rank}")
    score = float(value[2]) if len(value) > 2 and value[2] is not None else None
    return {
        "rank": rank,
        "start": start,
        "end": end,
        "duration": end - start,
        "score": score,
    }


def iou(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return (intersection / union if union > EPS else 0.0), intersection


def bin_for(length: float) -> str:
    if length <= 2.0 + EPS:
        return "0–2s"
    if length <= 5.0 + EPS:
        return "2–5s"
    if length <= 10.0 + EPS:
        return "5–10s"
    if length <= 20.0 + EPS:
        return "10–20s"
    return "20s+"


def focus_group(length: float) -> str | None:
    if length < 2.0 - EPS:
        return "<2s"
    if length < 5.0 - EPS:
        return "2≤GT<5s"
    if 10.0 - EPS < length <= 20.0 + EPS:
        return "10–20s"
    if length > 20.0 + EPS:
        return "20s+"
    return None


def matched_metrics(pred: dict[str, Any], gts: list[tuple[float, float]]) -> dict[str, Any]:
    pred_interval = (pred["start"], pred["end"])
    best_iou = -1.0
    best_index = None
    best_intersection = 0.0
    for index, gt in enumerate(gts):
        current_iou, current_intersection = iou(pred_interval, gt)
        if current_iou > best_iou + EPS:
            best_iou = current_iou
            best_index = index
            best_intersection = current_intersection

    result: dict[str, Any] = {
        "rank": pred["rank"],
        "start": pred["start"],
        "end": pred["end"],
        "duration": pred["duration"],
        "score": pred["score"],
        "iou": max(0.0, best_iou),
        "matched_gt_index": None,
        "gt_start": None,
        "gt_end": None,
        "gt_duration": None,
        "intersection_duration": None,
        "gt_coverage": None,
        "pred_coverage": None,
        "duration_ratio": None,
        "relation": None,
    }
    if best_iou <= EPS or best_index is None:
        return result

    gt_start, gt_end = gts[best_index]
    gt_duration = gt_end - gt_start
    gt_coverage = best_intersection / gt_duration if gt_duration > EPS else None
    pred_coverage = best_intersection / pred["duration"] if pred["duration"] > EPS else None
    if pred["start"] <= gt_start + EPS and pred["end"] >= gt_end - EPS:
        relation = "PRED_CONTAINS_GT"
    elif gt_start <= pred["start"] + EPS and gt_end >= pred["end"] - EPS:
        relation = "GT_CONTAINS_PRED"
    else:
        relation = "PARTIAL_OVERLAP"
    result.update(
        {
            "matched_gt_index": best_index,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "gt_duration": gt_duration,
            "intersection_duration": best_intersection,
            "gt_coverage": gt_coverage,
            "pred_coverage": pred_coverage,
            "duration_ratio": pred["duration"] / gt_duration if gt_duration > EPS else None,
            "relation": relation,
        }
    )
    return result


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in QUANTILES}
    ordered = sorted(values)
    result: dict[str, float | None] = {}
    for name, q in QUANTILES.items():
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            result[name] = ordered[lower]
        else:
            weight = position - lower
            result[name] = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return result


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    positive = [row for row in rows if row["iou"] > EPS]
    high_coverage = [row for row in positive if row["gt_coverage"] is not None and row["gt_coverage"] >= 0.8 - EPS]
    low_iou_05 = [row for row in high_coverage if row["iou"] < 0.5 - EPS]
    low_iou_07 = [row for row in high_coverage if row["iou"] < 0.7 - EPS]
    relation_counts = {name: sum(row["relation"] == name for row in positive) for name in ("PRED_CONTAINS_GT", "GT_CONTAINS_PRED", "PARTIAL_OVERLAP")}
    ratio_high = [row["duration_ratio"] for row in high_coverage if row["duration_ratio"] is not None]
    return {
        "N": n,
        "N_iou_gt_0": len(positive),
        "frac_iou_gt_0": fraction(len(positive), n),
        "median_pred_duration_sec": median([row["duration"] for row in rows]),
        "positive_overlap_medians": {
            "gt_duration_sec": median([row["gt_duration"] for row in positive]),
            "pred_duration_sec": median([row["duration"] for row in positive]),
            "pred_gt_duration_ratio": median([row["duration_ratio"] for row in positive if row["duration_ratio"] is not None]),
            "gt_coverage": median([row["gt_coverage"] for row in positive if row["gt_coverage"] is not None]),
            "pred_coverage": median([row["pred_coverage"] for row in positive if row["pred_coverage"] is not None]),
        },
        "relation_fractions_positive_overlap": {
            name: {"N": relation_counts[name], "fraction": fraction(relation_counts[name], len(positive))}
            for name in relation_counts
        },
        "gt_coverage_ge_0.8": {
            "N": len(high_coverage),
            "fraction_of_all_N": fraction(len(high_coverage), n),
            "median_ratio": median(ratio_high),
            "fraction_ratio_ge_1.5": fraction(sum(value >= 1.5 - EPS for value in ratio_high), len(ratio_high)),
            "fraction_ratio_ge_2": fraction(sum(value >= 2.0 - EPS for value in ratio_high), len(ratio_high)),
            "fraction_ratio_ge_4": fraction(sum(value >= 4.0 - EPS for value in ratio_high), len(ratio_high)),
        },
        "high_coverage_low_iou": {
            "iou_lt_0.5": {
                "N": len(low_iou_05),
                "fraction_of_all_N": fraction(len(low_iou_05), n),
                "median_ratio": median([row["duration_ratio"] for row in low_iou_05 if row["duration_ratio"] is not None]),
            },
            "iou_lt_0.7": {
                "N": len(low_iou_07),
                "fraction_of_all_N": fraction(len(low_iou_07), n),
                "median_ratio": median([row["duration_ratio"] for row in low_iou_07 if row["duration_ratio"] is not None]),
            },
        },
        "prediction_duration_quantiles_sec": quantiles([row["duration"] for row in rows]),
        "matched_gt_duration_quantiles_sec": quantiles([row["gt_duration"] for row in positive]),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def make_report(summary: dict[str, Any]) -> str:
    lines = [
        "# FAILURE AUDIT v0C — TEMPORAL OVER-EXTENSION",
        "",
        "This is a geometric diagnostic of the official QD-DETR predictions. It does not assess semantic correctness and does not use raw audio or manual labels.",
        "",
        "## Definitions",
        "",
        "- `GT length` is the maximum duration among a query's `relevant_windows`; bins are `0 < length ≤ 2`, `2 < length ≤ 5`, `5 < length ≤ 10`, `10 < length ≤ 20`, and `> 20` seconds.",
        "- `Top1` is the first submitted prediction. `Best10` is the prediction among the first 10 with the largest IoU against any GT window; ties keep the lowest prediction rank.",
        "- For each prediction, the matched GT is the GT window with maximum IoU. When maximum IoU is zero, matching-dependent fields are null rather than assigned to an arbitrary GT.",
        "- Coverage is intersection divided by the matched GT duration (`GT coverage`) or prediction duration (`prediction coverage`). Relation fractions use positive-overlap cases; the `GT coverage ≥ 0.8` rate uses all cases in the bin as denominator.",
        "",
        "## Per-bin summary",
        "",
        "| GT-length bin | N | Model | IoU>0 | median pred s | median ratio | median GT cov. | median pred cov. | PRED_CONTAINS_GT | GT cov.≥.8 | high-cov IoU<.5 | high-cov IoU<.7 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in BIN_LABELS:
        for model_key, model_label in (("top1", "Top1"), ("best_top10", "Best10")):
            item = summary["per_bin"][label][model_key]
            pos = item["positive_overlap_medians"]
            relations = item["relation_fractions_positive_overlap"]
            high = item["gt_coverage_ge_0.8"]
            low = item["high_coverage_low_iou"]
            lines.append(
                f"| {label if model_key == 'top1' else ''} | {item['N']} | {model_label} | {pct(item['frac_iou_gt_0'])} | {fmt(item['median_pred_duration_sec'])} | {fmt(pos['pred_gt_duration_ratio'])} | {pct(pos['gt_coverage'])} | {pct(pos['pred_coverage'])} | {pct(relations['PRED_CONTAINS_GT']['fraction'])} | {pct(high['fraction_of_all_N'])} | {pct(low['iou_lt_0.5']['fraction_of_all_N'])} | {pct(low['iou_lt_0.7']['fraction_of_all_N'])} |"
            )
    lines += [
        "",
        "## Positive-overlap medians and relations",
        "",
        "| GT-length bin | Model | N IoU>0 | median GT s | median pred s | median ratio | median GT cov. | median pred cov. | PRED_CONTAINS_GT | GT_CONTAINS_PRED | PARTIAL_OVERLAP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in BIN_LABELS:
        for model_key, model_label in (("top1", "Top1"), ("best_top10", "Best10")):
            item = summary["per_bin"][label][model_key]
            pos = item["positive_overlap_medians"]
            rel = item["relation_fractions_positive_overlap"]
            lines.append(
                f"| {label} | {model_label} | {item['N_iou_gt_0']} | {fmt(pos['gt_duration_sec'])} | {fmt(pos['pred_duration_sec'])} | {fmt(pos['pred_gt_duration_ratio'])} | {pct(pos['gt_coverage'])} | {pct(pos['pred_coverage'])} | {pct(rel['PRED_CONTAINS_GT']['fraction'])} | {pct(rel['GT_CONTAINS_PRED']['fraction'])} | {pct(rel['PARTIAL_OVERLAP']['fraction'])} |"
            )
    lines += [
        "",
        "## High-GT-coverage diagnostics",
        "",
        "`high-cov rate` is the fraction of all queries with GT coverage ≥ 0.8. Low-IoU rates use all queries in the bin as denominator; ratio medians use the corresponding high-coverage subset.",
        "",
        "| GT-length bin | Model | high-cov N/rate | median ratio | ratio≥1.5 | ratio≥2 | ratio≥4 | high-cov IoU<.5 N/rate, median ratio | high-cov IoU<.7 N/rate, median ratio |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for label in BIN_LABELS:
        for model_key, model_label in (("top1", "Top1"), ("best_top10", "Best10")):
            item = summary["per_bin"][label][model_key]
            high = item["gt_coverage_ge_0.8"]
            low = item["high_coverage_low_iou"]
            low05 = low["iou_lt_0.5"]
            low07 = low["iou_lt_0.7"]
            lines.append(
                f"| {label} | {model_label} | {high['N']}/{item['N']} ({pct(high['fraction_of_all_N'])}) | {fmt(high['median_ratio'])} | {pct(high['fraction_ratio_ge_1.5'])} | {pct(high['fraction_ratio_ge_2'])} | {pct(high['fraction_ratio_ge_4'])} | {low05['N']}/{item['N']} ({pct(low05['fraction_of_all_N'])}), {fmt(low05['median_ratio'])} | {low07['N']}/{item['N']} ({pct(low07['fraction_of_all_N'])}), {fmt(low07['median_ratio'])} |"
            )
    lines += [
        "",
        "## Duration quantiles",
        "",
        "| GT-length bin | Model | pred p10 | pred p25 | pred median | pred p75 | pred p90 | matched GT p10 | matched GT p25 | matched GT median | matched GT p75 | matched GT p90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in BIN_LABELS:
        for model_key, model_label in (("top1", "Top1"), ("best_top10", "Best10")):
            item = summary["per_bin"][label][model_key]
            pred_q = item["prediction_duration_quantiles_sec"]
            gt_q = item["matched_gt_duration_quantiles_sec"]
            lines.append(
                f"| {label} | {model_label} | {fmt(pred_q['p10'])} | {fmt(pred_q['p25'])} | {fmt(pred_q['median'])} | {fmt(pred_q['p75'])} | {fmt(pred_q['p90'])} | {fmt(gt_q['p10'])} | {fmt(gt_q['p25'])} | {fmt(gt_q['median'])} | {fmt(gt_q['p75'])} | {fmt(gt_q['p90'])} |"
            )
    lines += ["", "## Very-short focus", ""]
    lines.append("| Focus group | N | Model | IoU>0 | median pred s | median ratio | median GT cov. | GT cov.≥.8 | ratio≥2 within high-cov |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for label in ("<2s", "2≤GT<5s", "10–20s", "20s+"):
        for model_key, model_label in (("top1", "Top1"), ("best_top10", "Best10")):
            item = summary["very_short_focus"][label][model_key]
            pos = item["positive_overlap_medians"]
            high = item["gt_coverage_ge_0.8"]
            lines.append(
                f"| {label if model_key == 'top1' else ''} | {item['N']} | {model_label} | {pct(item['frac_iou_gt_0'])} | {fmt(item['median_pred_duration_sec'])} | {fmt(pos['pred_gt_duration_ratio'])} | {pct(pos['gt_coverage'])} | {pct(high['fraction_of_all_N'])} | {pct(high['fraction_ratio_ge_2'])} |"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "These results can establish temporal over-extension or disproportionate geometry on short GT windows. They cannot establish that a high-coverage prediction is semantically correct; semantic correctness remains outside this diagnostic.",
        "",
        f"Input queries matched by qid: {summary['counts']['matched_qids']}; prediction rows: {summary['counts']['prediction_rows']}; GT rows: {summary['counts']['gt_rows']}.",
        "",
        "No model was run, no baseline source was changed, and no audio was downloaded or read.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    for path in (CSV_PATH, JSON_PATH, MD_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    predictions = load_jsonl(PRED_PATH)
    ground_truth = load_jsonl(GT_PATH)
    pred_by_qid = {row["qid"]: row for row in predictions}
    gt_by_qid = {row["qid"]: row for row in ground_truth}
    if len(pred_by_qid) != len(predictions) or len(gt_by_qid) != len(ground_truth):
        raise ValueError("duplicate qid detected")
    common_qids = sorted(set(pred_by_qid) & set(gt_by_qid))
    missing_predictions = sorted(set(gt_by_qid) - set(pred_by_qid))
    missing_ground_truth = sorted(set(pred_by_qid) - set(gt_by_qid))
    if missing_predictions or missing_ground_truth:
        raise ValueError(f"qid mismatch: missing_predictions={missing_predictions[:5]}, missing_ground_truth={missing_ground_truth[:5]}")

    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    invalid_prediction_count = 0
    for qid in common_qids:
        pred_row = pred_by_qid[qid]
        gt_row = gt_by_qid[qid]
        gt_windows = [interval(value, f"GT window for {qid}") for value in gt_row.get("relevant_windows", [])]
        if not gt_windows:
            raise ValueError(f"no GT windows for {qid}")
        max_gt_length = max(end - start for start, end in gt_windows)
        candidates = [prediction(value, rank) for rank, value in enumerate(pred_row.get("pred_relevant_windows", []), 1)]
        if not candidates:
            raise ValueError(f"no predictions for {qid}")
        if any(candidate["duration"] <= EPS for candidate in candidates[:10]):
            invalid_prediction_count += 1
        top1 = matched_metrics(candidates[0], gt_windows)
        candidate_metrics = [matched_metrics(candidate, gt_windows) for candidate in candidates[:10]]
        best10 = candidate_metrics[0]
        for candidate in candidate_metrics[1:]:
            if candidate["iou"] > best10["iou"] + EPS:
                best10 = candidate
        row = {
            "qid": qid,
            "vid": pred_row.get("vid", gt_row.get("vid")),
            "query": pred_row.get("query", gt_row.get("query")),
            "audio_duration_sec": gt_row.get("duration"),
            "max_gt_length_sec": max_gt_length,
            "gt_duration_bin": bin_for(max_gt_length),
            "top1": top1,
            "best_top10": best10,
        }
        rows.append(row)
        flat: dict[str, Any] = {
            "qid": qid,
            "vid": row["vid"],
            "query": row["query"],
            "audio_duration_sec": row["audio_duration_sec"],
            "max_gt_length_sec": max_gt_length,
            "gt_duration_bin": row["gt_duration_bin"],
            "top10_candidate_count": len(candidate_metrics),
        }
        fields = ("rank", "start", "end", "duration", "score", "iou", "matched_gt_index", "gt_start", "gt_end", "gt_duration", "intersection_duration", "gt_coverage", "pred_coverage", "duration_ratio", "relation")
        for prefix, metrics in (("top1", top1), ("best10", best10)):
            for field in fields:
                flat[f"{prefix}_{field}"] = metrics[field]
        csv_rows.append(flat)

    per_bin: dict[str, Any] = {}
    for label in BIN_LABELS:
        group = [row for row in rows if row["gt_duration_bin"] == label]
        per_bin[label] = {name: metric_summary([row[name] for row in group]) for name in ("top1", "best_top10")}

    focus_groups: dict[str, Any] = {}
    for label in ("<2s", "2≤GT<5s", "10–20s", "20s+"):
        group = [row for row in rows if focus_group(row["max_gt_length_sec"]) == label]
        focus_groups[label] = {name: metric_summary([row[name] for row in group]) for name in ("top1", "best_top10")}

    summary = {
        "status": "FAILURE_AUDIT_V0C_COMPLETE",
        "counts": {
            "prediction_rows": len(predictions),
            "gt_rows": len(ground_truth),
            "matched_qids": len(common_qids),
            "missing_predictions": missing_predictions,
            "missing_ground_truth": missing_ground_truth,
            "invalid_or_zero_duration_prediction_rows": invalid_prediction_count,
        },
        "inputs": {"predictions": str(PRED_PATH), "ground_truth": str(GT_PATH)},
        "definitions": {
            "gt_length": "maximum end-start among relevant_windows for the query",
            "bins": {
                "0–2s": "0 < max_gt_length_sec <= 2",
                "2–5s": "2 < max_gt_length_sec <= 5",
                "5–10s": "5 < max_gt_length_sec <= 10",
                "10–20s": "10 < max_gt_length_sec <= 20",
                "20s+": "max_gt_length_sec > 20",
            },
            "top1": "first submitted prediction",
            "best_top10": "candidate with maximum IoU against any GT window among submitted ranks 1-10; ties keep lowest rank",
            "matched_gt": "GT window with maximum IoU for the selected prediction",
            "zero_iou": "matching-dependent fields are null when maximum IoU <= 1e-9",
            "relation_tolerance_sec": EPS,
            "coverage": "intersection / matched GT duration or prediction duration",
            "relation_denominator": "positive-overlap cases",
            "high_coverage_rate_denominator": "all cases in the bin",
            "quantile_method": "linear interpolation on sorted values at (N-1)*q",
        },
        "per_bin": per_bin,
        "very_short_focus": focus_groups,
        "outputs": {"query_csv": str(CSV_PATH), "summary_json": str(JSON_PATH), "report_md": str(MD_PATH)},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(csv_rows[0].keys())
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MD_PATH.write_text(make_report(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "counts": summary["counts"], "outputs": summary["outputs"]}, ensure_ascii=False, indent=2))
    for label in BIN_LABELS:
        print(label, json.dumps({key: {"N": value["N"], "overlap": value["frac_iou_gt_0"], "median_ratio": value["positive_overlap_medians"]["pred_gt_duration_ratio"], "highcov": value["gt_coverage_ge_0.8"]["fraction_of_all_N"]} for key, value in per_bin[label].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
