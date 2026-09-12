#!/usr/bin/env python3
"""Build the quantitative QD-DETR failure audit without touching the baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EXPECTED_QUERIES = 1347
OFFICIAL_R1_05 = 23.16
OFFICIAL_R1_07 = 10.32
IOU_05 = 0.5
IOU_07 = 0.7
ORACLE_KS = (1, 3, 5, 10)
SEED = 20260820


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                fail("Invalid JSON at {}:{}: {}".format(path, line_number, exc))
            if not isinstance(value, dict):
                fail("Expected an object at {}:{}".format(path, line_number))
            rows.append(value)
    return rows


def require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail("{} must be numeric, got {!r}".format(label, value))
    result = float(value)
    if not math.isfinite(result):
        fail("{} must be finite, got {!r}".format(label, value))
    return result


def validate_interval(value: Any, label: str, allow_zero_length: bool = False) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        fail("{} must contain [start, end], got {!r}".format(label, value))
    start = require_finite_number(value[0], label + ".start")
    end = require_finite_number(value[1], label + ".end")
    if end < start or (end == start and not allow_zero_length):
        fail("{} must have end > start, got {!r}".format(label, value))
    return start, end


def temporal_iou(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0.0 else 0.0


def best_iou(prediction: Tuple[float, float], gt_windows: Sequence[Tuple[float, float]]) -> Tuple[float, int]:
    values = [temporal_iou(prediction, gt) for gt in gt_windows]
    best_index = max(range(len(values)), key=lambda index: values[index])
    return values[best_index], best_index


def first_hit_rank(
    predictions: Sequence[Tuple[float, float, float]],
    gt_windows: Sequence[Tuple[float, float]],
    threshold: float,
) -> Optional[int]:
    for rank, prediction in enumerate(predictions, start=1):
        if best_iou((prediction[0], prediction[1]), gt_windows)[0] >= threshold:
            return rank
    return None


def nearest_disjoint_relation(
    prediction: Tuple[float, float], gt_windows: Sequence[Tuple[float, float]]
) -> Optional[str]:
    candidates: List[Tuple[float, int, str]] = []
    for index, gt in enumerate(gt_windows):
        if prediction[1] <= gt[0]:
            candidates.append((gt[0] - prediction[1], index, "EARLY"))
        elif prediction[0] >= gt[1]:
            candidates.append((prediction[0] - gt[1], index, "LATE"))
    if not candidates:
        return None
    return min(candidates, key=lambda value: (value[0], value[1]))[2]


def rate(rows: Sequence[Dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(1 for row in rows if row[field]) / len(rows)


def optional_mean(rows: Sequence[Dict[str, Any]], field: str) -> Optional[float]:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return mean(values) if values else None


def group_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "N": len(rows),
        "R1@0.5": rate(rows, "top1_pass_05"),
        "R1@0.7": rate(rows, "top1_pass_07"),
        "Oracle10@0.5": rate(rows, "oracle_10_pass_05"),
        "Oracle10@0.7": rate(rows, "oracle_10_pass_07"),
    }


def make_bin(value: float, boundaries: Sequence[float], labels: Sequence[str]) -> str:
    for boundary, label in zip(boundaries, labels):
        if value < boundary:
            return label
    return labels[-1]


def add_confidence_deciles(rows: List[Dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (float(row["top1_score"]), str(row["qid"])))
    for position, row in enumerate(ordered):
        decile = min(10, (position * 10) // len(ordered) + 1)
        row["confidence_decile"] = "decile_{:02d}".format(decile)


def validate_and_join(
    gt_rows: Sequence[Dict[str, Any]], prediction_rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if len(gt_rows) != EXPECTED_QUERIES:
        fail("Expected {} GT queries, found {}".format(EXPECTED_QUERIES, len(gt_rows)))

    def index_rows(rows: Sequence[Dict[str, Any]], name: str) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for row_number, row in enumerate(rows, start=1):
            qid = row.get("qid")
            if not isinstance(qid, str) or not qid:
                fail("{} row {} has invalid qid {!r}".format(name, row_number, qid))
            if qid in indexed:
                fail("Duplicate {} qid: {}".format(name, qid))
            indexed[qid] = row
        return indexed

    gt_by_qid = index_rows(gt_rows, "GT")
    pred_by_qid = index_rows(prediction_rows, "prediction")
    missing = sorted(set(gt_by_qid) - set(pred_by_qid))
    extra = sorted(set(pred_by_qid) - set(gt_by_qid))
    if missing:
        fail("Missing predictions for {} qids: {}".format(len(missing), missing[:5]))
    if extra:
        fail("Predictions contain {} unknown qids: {}".format(len(extra), extra[:5]))
    if len(pred_by_qid) != EXPECTED_QUERIES:
        fail("Expected {} predictions, found {}".format(EXPECTED_QUERIES, len(pred_by_qid)))

    joined: List[Dict[str, Any]] = []
    for gt in gt_rows:
        qid = gt["qid"]
        pred = pred_by_qid[qid]
        if gt.get("query") != pred.get("query"):
            fail("Query mismatch for qid {}".format(qid))
        if gt.get("vid") != pred.get("vid"):
            fail("Video mismatch for qid {}".format(qid))
        duration = require_finite_number(gt.get("duration"), "GT {} duration".format(qid))
        raw_gt_windows = gt.get("relevant_windows")
        if not isinstance(raw_gt_windows, list) or not raw_gt_windows:
            fail("GT {} has no relevant_windows".format(qid))
        gt_windows = [
            validate_interval(window, "GT {} window {}".format(qid, index))
            for index, window in enumerate(raw_gt_windows)
        ]
        raw_predictions = pred.get("pred_relevant_windows")
        if not isinstance(raw_predictions, list) or not raw_predictions:
            fail("Prediction {} has no pred_relevant_windows".format(qid))
        predictions: List[Tuple[float, float, float]] = []
        for index, window in enumerate(raw_predictions):
            if not isinstance(window, (list, tuple)) or len(window) < 3:
                fail("Prediction {} window {} must contain [start, end, score]".format(qid, index))
            start, end = validate_interval(
                window, "Prediction {} window {}".format(qid, index), allow_zero_length=True
            )
            score = require_finite_number(window[2], "Prediction {} score {}".format(qid, index))
            predictions.append((start, end, score))

        top1 = predictions[0]
        top1_interval = (top1[0], top1[1])
        top1_iou, matched_index = best_iou(top1_interval, gt_windows)
        matched_gt = gt_windows[matched_index]
        oracle_best: Dict[int, float] = {}
        for k in ORACLE_KS:
            oracle_best[k] = max(
                best_iou((item[0], item[1]), gt_windows)[0] for item in predictions[:k]
            )
        overlaps_any = top1_iou > 0.0
        gt_length = matched_gt[1] - matched_gt[0]
        pred_length = top1[1] - top1[0]
        category: str
        disjoint_relation: Optional[str] = None
        if top1_iou >= IOU_07:
            category = "PASS_07"
        elif top1_iou >= IOU_05:
            category = "PASS_05_ONLY"
        elif overlaps_any:
            category = "PARTIAL_OVERLAP"
        else:
            disjoint_relation = nearest_disjoint_relation(top1_interval, gt_windows)
            category = "DISJOINT_{}".format(disjoint_relation or "UNRESOLVED")
        joined.append(
            {
                "qid": qid,
                "vid": gt["vid"],
                "query": gt["query"],
                "duration": duration,
                "num_gt_windows": len(gt_windows),
                "gt_windows": raw_gt_windows,
                "num_predictions": len(predictions),
                "top1_start": top1[0],
                "top1_end": top1[1],
                "top1_score": top1[2],
                "top2_score": predictions[1][2] if len(predictions) > 1 else None,
                "score_margin_top1_top2": top1[2] - predictions[1][2] if len(predictions) > 1 else None,
                "top1_best_iou": top1_iou,
                "matched_gt_start": matched_gt[0],
                "matched_gt_end": matched_gt[1],
                "top1_pass_05": top1_iou >= IOU_05,
                "top1_pass_07": top1_iou >= IOU_07,
                "oracle_1_best_iou": oracle_best[1],
                "oracle_3_best_iou": oracle_best[3],
                "oracle_5_best_iou": oracle_best[5],
                "oracle_10_best_iou": oracle_best[10],
                "oracle_1_pass_05": oracle_best[1] >= IOU_05,
                "oracle_1_pass_07": oracle_best[1] >= IOU_07,
                "oracle_3_pass_05": oracle_best[3] >= IOU_05,
                "oracle_3_pass_07": oracle_best[3] >= IOU_07,
                "oracle_5_pass_05": oracle_best[5] >= IOU_05,
                "oracle_5_pass_07": oracle_best[5] >= IOU_07,
                "oracle_10_pass_05": oracle_best[10] >= IOU_05,
                "oracle_10_pass_07": oracle_best[10] >= IOU_07,
                "first_hit_rank_05": first_hit_rank(predictions, gt_windows, IOU_05),
                "first_hit_rank_07": first_hit_rank(predictions, gt_windows, IOU_07),
                "gt_length": gt_length,
                "pred_length": pred_length,
                "start_error": top1[0] - matched_gt[0],
                "end_error": top1[1] - matched_gt[1],
                "center_error": (top1[0] + top1[1]) / 2.0 - (matched_gt[0] + matched_gt[1]) / 2.0,
                "length_ratio": pred_length / gt_length,
                "overlaps_any_gt": overlaps_any,
                "disjoint_relation": disjoint_relation,
                "geometry_label": category,
                "max_gt_length": max(end - start for start, end in gt_windows),
            }
        )
    add_confidence_deciles(joined)
    return joined


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    top1 = {"R@0.5": rate(rows, "top1_pass_05"), "R@0.7": rate(rows, "top1_pass_07")}
    oracle = {
        "R@0.5": {"Top1": rate(rows, "oracle_1_pass_05"), "Top3": rate(rows, "oracle_3_pass_05"), "Top5": rate(rows, "oracle_5_pass_05"), "Top10": rate(rows, "oracle_10_pass_05")},
        "R@0.7": {"Top1": rate(rows, "oracle_1_pass_07"), "Top3": rate(rows, "oracle_3_pass_07"), "Top5": rate(rows, "oracle_5_pass_07"), "Top10": rate(rows, "oracle_10_pass_07")},
    }
    if abs(top1["R@0.5"] - OFFICIAL_R1_05) > 0.02 or abs(top1["R@0.7"] - OFFICIAL_R1_07) > 0.02:
        fail("Metric reproduction failed: {} vs {}, {} vs {}".format(top1["R@0.5"], OFFICIAL_R1_05, top1["R@0.7"], OFFICIAL_R1_07))

    failures_07 = [row for row in rows if not row["top1_pass_07"]]
    iou_buckets = {
        "IoU=0": lambda value: value == 0.0,
        "0<IoU<0.3": lambda value: 0.0 < value < 0.3,
        "0.3<=IoU<0.5": lambda value: 0.3 <= value < 0.5,
        "0.5<=IoU<0.7": lambda value: 0.5 <= value < 0.7,
    }
    failure_iou_distribution: Dict[str, Any] = {"N_failures": len(failures_07)}
    for label, predicate in iou_buckets.items():
        count = sum(1 for row in failures_07 if predicate(float(row["top1_best_iou"])))
        failure_iou_distribution[label] = {"N": count, "fraction": count / len(failures_07) if failures_07 else 0.0}

    geometry_counts: Dict[str, int] = {}
    failure_geometry_counts: Dict[str, int] = {}
    for row in rows:
        label = str(row["geometry_label"])
        geometry_counts[label] = geometry_counts.get(label, 0) + 1
        if not row["top1_pass_07"]:
            failure_geometry_counts[label] = failure_geometry_counts.get(label, 0) + 1

    multi_gt = {
        "single_gt": group_metrics([row for row in rows if row["num_gt_windows"] == 1]),
        "multi_gt": group_metrics([row for row in rows if row["num_gt_windows"] >= 2]),
    }

    duration_boundaries = (60.0, 120.0, 180.0, 240.0, float("inf"))
    duration_labels = ("0-60 sec", "60-120 sec", "120-180 sec", "180-240 sec", "240+ sec")
    duration_bins: Dict[str, Any] = {}
    for row in rows:
        label = make_bin(float(row["duration"]), duration_boundaries, duration_labels)
        duration_bins.setdefault(label, []).append(row)
    duration_stats = {label: {**group_metrics(group), "Oracle10@0.7": rate(group, "oracle_10_pass_07")} for label, group in duration_bins.items()}

    gt_boundaries = (2.0, 5.0, 10.0, 20.0, float("inf"))
    gt_labels = ("0-2 sec", "2-5 sec", "5-10 sec", "10-20 sec", "20+ sec")
    gt_length_bins: Dict[str, Any] = {}
    for row in rows:
        label = make_bin(float(row["max_gt_length"]), gt_boundaries, gt_labels)
        gt_length_bins.setdefault(label, []).append(row)
    gt_length_stats = {label: group_metrics(group) for label, group in gt_length_bins.items()}

    confidence_bins: Dict[str, Any] = {}
    for row in rows:
        confidence_bins.setdefault(str(row["confidence_decile"]), []).append(row)
    confidence_stats = {
        label: {
            "N": len(group),
            "mean_confidence": optional_mean(group, "top1_score"),
            "R1@0.5": rate(group, "top1_pass_05"),
            "R1@0.7": rate(group, "top1_pass_07"),
        }
        for label, group in sorted(confidence_bins.items())
    }

    return {
        "audit_version": "failure_audit_v0A",
        "seed": SEED,
        "n_queries": len(rows),
        "top1_reproduction": {
            "computed_R1@0.5": top1["R@0.5"],
            "official_R1@0.5": OFFICIAL_R1_05,
            "computed_R1@0.7": top1["R@0.7"],
            "official_R1@0.7": OFFICIAL_R1_07,
        },
        "oracle_ranking": oracle,
        "failure_iou_distribution_among_top1_failures_at_0.7": failure_iou_distribution,
        "geometry_counts_all_queries": geometry_counts,
        "geometry_counts_among_top1_failures_at_0.7": failure_geometry_counts,
        "single_vs_multi_gt": multi_gt,
        "duration_bins": duration_stats,
        "gt_length_bins": {
            "policy": "max_gt_window_length_per_query",
            "bins": gt_length_stats,
        },
        "confidence_deciles": {
            "policy": "rank-based deciles, ascending confidence, deterministic qid tie-break",
            "bins": confidence_stats,
        },
    }


def choose_candidates(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: Dict[str, str] = {}
    result: List[Dict[str, Any]] = []

    def pick(group: str, candidates: Iterable[Dict[str, Any]], count: int, randomize: bool = False) -> None:
        available = [row for row in candidates if row["qid"] not in selected]
        if randomize:
            random.Random(SEED + len(result)).shuffle(available)
        else:
            available.sort(key=lambda row: str(row["qid"]))
        for row in available[:count]:
            selected[row["qid"]] = group
            copied = dict(row)
            copied["review_group"] = group
            result.append(copied)

    pick(
        "A_high_confidence_catastrophic",
        sorted(
            (row for row in rows if float(row["top1_best_iou"]) == 0.0),
            key=lambda row: (-float(row["top1_score"]), str(row["qid"])),
        ),
        20,
    )
    pick(
        "B_ranking_gap",
        sorted(
            (row for row in rows if float(row["top1_best_iou"]) < 0.5 and float(row["oracle_10_best_iou"]) >= 0.7),
            key=lambda row: (-float(row["oracle_10_best_iou"]), float(row["top1_best_iou"]), str(row["qid"])),
        ),
        20,
    )
    pick(
        "C_boundary_localization",
        sorted(
            (
                row
                for row in rows
                if 0.3 <= float(row["top1_best_iou"]) < 0.7 and row["overlaps_any_gt"]
            ),
            key=lambda row: (-float(row["top1_best_iou"]), str(row["qid"])),
        ),
        20,
    )
    pick(
        "D_multiple_gt_failure",
        sorted(
            (row for row in rows if row["num_gt_windows"] >= 2 and not row["top1_pass_07"]),
            key=lambda row: (-float(row["top1_score"]), str(row["qid"])),
        ),
        20,
    )
    pick("E_general_failure", (row for row in rows if not row["top1_pass_07"]), 20, randomize=True)
    return result


CSV_FIELDS = [
    "qid", "vid", "query", "duration", "num_gt_windows", "gt_windows", "num_predictions",
    "top1_start", "top1_end", "top1_score", "top2_score", "score_margin_top1_top2",
    "top1_best_iou", "matched_gt_start", "matched_gt_end", "top1_pass_05", "top1_pass_07",
    "oracle_1_best_iou", "oracle_3_best_iou", "oracle_5_best_iou", "oracle_10_best_iou",
    "oracle_1_pass_05", "oracle_1_pass_07", "oracle_3_pass_05", "oracle_3_pass_07",
    "oracle_5_pass_05", "oracle_5_pass_07", "oracle_10_pass_05", "oracle_10_pass_07",
    "first_hit_rank_05", "first_hit_rank_07", "gt_length", "pred_length", "start_error",
    "end_error", "center_error", "length_ratio", "overlaps_any_gt", "disjoint_relation",
    "geometry_label", "max_gt_length", "confidence_decile",
]


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def write_summary(path: Path, summary: Dict[str, Any], rows: Sequence[Dict[str, Any]], candidates: Sequence[Dict[str, Any]]) -> None:
    summary = dict(summary)
    summary["input_validation"] = {
        "gt_queries": len(rows),
        "prediction_queries": len(rows),
        "matched_queries": len(rows),
        "missing": 0,
        "duplicates": 0,
    }
    summary["manual_review"] = {
        "N": len(candidates),
        "group_counts": {
            group: sum(1 for row in candidates if row["review_group"] == group)
            for group in sorted({row["review_group"] for row in candidates})
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def git_commit(baseline_dir: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(baseline_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("/private/research-artifact"))
    parser.add_argument("--audit-dir", type=Path, default=Path("/private/research-artifact"))
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    audit_dir = args.audit_dir.resolve()
    output_dir = audit_dir / "outputs" / "qd_detr"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = baseline_dir / "data" / "castella_test_release.jsonl"
    prediction_path = baseline_dir / "results" / "submission.jsonl"
    gt_rows = load_jsonl(gt_path)
    prediction_rows = load_jsonl(prediction_path)
    rows = validate_and_join(gt_rows, prediction_rows)
    summary = aggregate(rows)
    candidates = choose_candidates(rows)
    summary["baseline_commit"] = git_commit(baseline_dir)
    summary["inputs"] = {
        "ground_truth": str(gt_path),
        "predictions": str(prediction_path),
    }
    write_csv(output_dir / "query_audit.csv", rows, CSV_FIELDS)
    write_summary(output_dir / "summary.json", summary, rows, candidates)
    write_csv(output_dir / "manual_review_candidates.csv", candidates, CSV_FIELDS + ["review_group"])

    print("FAILURE_AUDIT_V0A_COMPLETE")
    print("Input validation: GT={}, predictions={}, matched={}, missing=0, duplicates=0".format(len(gt_rows), len(prediction_rows), len(rows)))
    print("Metric reproduction: R1@0.5={:.2f}, R1@0.7={:.2f}".format(summary["top1_reproduction"]["computed_R1@0.5"], summary["top1_reproduction"]["computed_R1@0.7"]))
    print("Oracle R@0.5: Top1={:.2f}, Top3={:.2f}, Top5={:.2f}, Top10={:.2f}".format(*[summary["oracle_ranking"]["R@0.5"][key] for key in ("Top1", "Top3", "Top5", "Top10")]))
    print("Oracle R@0.7: Top1={:.2f}, Top3={:.2f}, Top5={:.2f}, Top10={:.2f}".format(*[summary["oracle_ranking"]["R@0.7"][key] for key in ("Top1", "Top3", "Top5", "Top10")]))
    candidate_group_counts = {
        group: sum(1 for row in candidates if row["review_group"] == group)
        for group in sorted({row["review_group"] for row in candidates})
    }
    print("Manual review candidates: {} ({})".format(len(candidates), ", ".join("{}={}".format(key, value) for key, value in candidate_group_counts.items())))
    print("Outputs: {}".format(output_dir))


if __name__ == "__main__":
    main()
