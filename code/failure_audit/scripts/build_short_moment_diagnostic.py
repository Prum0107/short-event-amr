#!/usr/bin/env python3
"""Build v0B short-moment and audio-duration confound diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


GT_DURATION_BINS: Sequence[Tuple[str, Callable[[float], bool]]] = (
    ("0-2 sec", lambda value: 0.0 <= value < 2.0),
    ("2-5 sec", lambda value: 2.0 <= value < 5.0),
    ("5-10 sec", lambda value: 5.0 <= value < 10.0),
    ("10-20 sec", lambda value: 10.0 <= value < 20.0),
    ("20+ sec", lambda value: value >= 20.0),
)
AUDIO_DURATION_BINS: Sequence[Tuple[str, Callable[[float], bool]]] = (
    ("0-60 sec", lambda value: 0.0 <= value < 60.0),
    ("60-120 sec", lambda value: 60.0 <= value < 120.0),
    ("120-180 sec", lambda value: 120.0 <= value < 180.0),
    ("180-240 sec", lambda value: 180.0 <= value < 240.0),
    ("240+ sec", lambda value: value >= 240.0),
)
GT_STRATA: Sequence[Tuple[str, Callable[[float], bool]]] = (
    ("GT <= 5 sec", lambda value: 0.0 <= value <= 5.0),
    ("5 sec < GT <= 20 sec", lambda value: 5.0 < value <= 20.0),
    ("GT > 20 sec", lambda value: value > 20.0),
)
IOU_BUCKETS: Sequence[Tuple[str, Callable[[float], bool]]] = (
    ("IoU=0", lambda value: value == 0.0),
    ("0<IoU<0.3", lambda value: 0.0 < value < 0.3),
    ("0.3<=IoU<0.5", lambda value: 0.3 <= value < 0.5),
    ("0.5<=IoU<0.7", lambda value: 0.5 <= value < 0.7),
    ("IoU>=0.7", lambda value: value >= 0.7),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_float(row: Dict[str, str], field: str, row_number: int) -> float:
    raw = row.get(field, "")
    if raw == "":
        fail("Missing {} at CSV row {}".format(field, row_number))
    value = float(raw)
    if not math.isfinite(value):
        fail("Non-finite {} at CSV row {}".format(field, row_number))
    return value


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "qid", "max_gt_length", "duration", "top1_best_iou", "oracle_3_best_iou",
            "oracle_5_best_iou", "oracle_10_best_iou", "start_error", "end_error",
            "center_error", "gt_length",
        }
        missing_fields = sorted(required - set(reader.fieldnames or []))
        if missing_fields:
            fail("query_audit.csv is missing fields: {}".format(missing_fields))
        rows = list(reader)
    if len(rows) != 1347:
        fail("Expected 1347 query rows, found {}".format(len(rows)))
    qids = [row["qid"] for row in rows]
    if len(set(qids)) != len(qids):
        fail("query_audit.csv contains duplicate qids")
    return rows


def find_bin(value: float, bins: Sequence[Tuple[str, Callable[[float], bool]]], label: str) -> str:
    for bin_label, predicate in bins:
        if predicate(value):
            return bin_label
    fail("Value {}={} does not fit {}".format(label, value, [item[0] for item in bins]))


def rate(rows: Sequence[Dict[str, str]], iou_field: str, threshold: float) -> Optional[float]:
    if not rows:
        return None
    values = [parse_float(row, iou_field, -1) for row in rows]
    return 100.0 * sum(value >= threshold for value in values) / len(values)


def median_errors(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    absolute: Dict[str, List[float]] = {"start": [], "end": [], "center": []}
    normalized: Dict[str, List[float]] = {"start": [], "end": [], "center": []}
    for index, row in enumerate(rows, start=1):
        gt_length = parse_float(row, "gt_length", index)
        if gt_length <= 0.0:
            continue
        for name, field in (("start", "start_error"), ("end", "end_error"), ("center", "center_error")):
            error = abs(parse_float(row, field, index))
            absolute[name].append(error)
            normalized[name].append(error / gt_length)

    def med(values: Sequence[float]) -> Optional[float]:
        return median(values) if values else None

    return {
        "matching_definition": "Top-1 prediction matched to the GT window that maximizes Top-1 temporal IoU; errors use the query_audit matched GT fields.",
        "absolute_valid_N": len(absolute["start"]),
        "normalized_valid_N": len(normalized["start"]),
        "median_abs_start_error": med(absolute["start"]),
        "median_abs_end_error": med(absolute["end"]),
        "median_abs_center_error": med(absolute["center"]),
        "median_abs_start_error_over_gt_length": med(normalized["start"]),
        "median_abs_end_error_over_gt_length": med(normalized["end"]),
        "median_abs_center_error_over_gt_length": med(normalized["center"]),
    }


def iou_distribution(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    values = [parse_float(row, "top1_best_iou", -1) for row in rows]
    result: Dict[str, Dict[str, Any]] = {}
    for label, predicate in IOU_BUCKETS:
        count = sum(predicate(value) for value in values)
        result[label] = {"N": count, "fraction": count / len(values) if values else None}
    return result


def gt_duration_stats(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for label, _ in GT_DURATION_BINS:
        group = [row for row in rows if row["gt_duration_bin"] == label]
        result[label] = {
            "N": len(group),
            "Top1 R@0.5": rate(group, "top1_best_iou", 0.5),
            "Top1 R@0.7": rate(group, "top1_best_iou", 0.7),
            "Oracle3@0.5": rate(group, "oracle_3_best_iou", 0.5),
            "Oracle5@0.5": rate(group, "oracle_5_best_iou", 0.5),
            "Oracle10@0.5": rate(group, "oracle_10_best_iou", 0.5),
            "Oracle3@0.7": rate(group, "oracle_3_best_iou", 0.7),
            "Oracle5@0.7": rate(group, "oracle_5_best_iou", 0.7),
            "Oracle10@0.7": rate(group, "oracle_10_best_iou", 0.7),
            "temporal_error": median_errors(group),
            "top1_iou_distribution": iou_distribution(group),
        }
    return result


def audio_duration_by_gt_stratum(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for stratum, _ in GT_STRATA:
        result[stratum] = {}
        stratum_rows = [row for row in rows if row["gt_length_stratum"] == stratum]
        for audio_bin, _ in AUDIO_DURATION_BINS:
            group = [row for row in stratum_rows if row["audio_duration_bin"] == audio_bin]
            result[stratum][audio_bin] = {
                "N": len(group),
                "R1@0.7": rate(group, "top1_best_iou", 0.7),
                "Oracle10@0.7": rate(group, "oracle_10_best_iou", 0.7),
            }
    return result


CSV_FIELDS = [
    "section", "gt_duration_bin", "gt_length_stratum", "audio_duration_bin", "N",
    "Top1_R@0.5", "Top1_R@0.7", "Oracle3@0.5", "Oracle5@0.5", "Oracle10@0.5",
    "Oracle3@0.7", "Oracle5@0.7", "Oracle10@0.7", "median_abs_start_error",
    "median_abs_end_error", "median_abs_center_error", "median_abs_start_error_over_gt_length",
    "median_abs_end_error_over_gt_length", "median_abs_center_error_over_gt_length",
    "absolute_error_valid_N", "normalized_error_valid_N", "IoU=0_N", "IoU=0_fraction",
    "0<IoU<0.3_N", "0<IoU<0.3_fraction", "0.3<=IoU<0.5_N", "0.3<=IoU<0.5_fraction",
    "0.5<=IoU<0.7_N", "0.5<=IoU<0.7_fraction", "IoU>=0.7_N", "IoU>=0.7_fraction",
]


def csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_csv(path: Path, gt_stats: Dict[str, Dict[str, Any]], audio_stats: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for gt_bin, stats in gt_stats.items():
            error = stats["temporal_error"]
            distribution = stats["top1_iou_distribution"]
            row = {
                "section": "gt_duration_bin",
                "gt_duration_bin": gt_bin,
                "N": stats["N"],
                "Top1_R@0.5": stats["Top1 R@0.5"],
                "Top1_R@0.7": stats["Top1 R@0.7"],
                "Oracle3@0.5": stats["Oracle3@0.5"],
                "Oracle5@0.5": stats["Oracle5@0.5"],
                "Oracle10@0.5": stats["Oracle10@0.5"],
                "Oracle3@0.7": stats["Oracle3@0.7"],
                "Oracle5@0.7": stats["Oracle5@0.7"],
                "Oracle10@0.7": stats["Oracle10@0.7"],
                "median_abs_start_error": error["median_abs_start_error"],
                "median_abs_end_error": error["median_abs_end_error"],
                "median_abs_center_error": error["median_abs_center_error"],
                "median_abs_start_error_over_gt_length": error["median_abs_start_error_over_gt_length"],
                "median_abs_end_error_over_gt_length": error["median_abs_end_error_over_gt_length"],
                "median_abs_center_error_over_gt_length": error["median_abs_center_error_over_gt_length"],
                "absolute_error_valid_N": error["absolute_valid_N"],
                "normalized_error_valid_N": error["normalized_valid_N"],
            }
            for label, values in distribution.items():
                row[label + "_N"] = values["N"]
                row[label + "_fraction"] = values["fraction"]
            writer.writerow({field: csv_value(row.get(field)) for field in CSV_FIELDS})
        for stratum, bins in audio_stats.items():
            for audio_bin, stats in bins.items():
                writer.writerow({
                    field: csv_value({
                        "section": "audio_duration_by_gt_stratum",
                        "gt_length_stratum": stratum,
                        "audio_duration_bin": audio_bin,
                        "N": stats["N"],
                        "Top1_R@0.7": stats["R1@0.7"],
                        "Oracle10@0.7": stats["Oracle10@0.7"],
                    }.get(field))
                    for field in CSV_FIELDS
                })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/private/research-artifact"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/research-artifact"),
    )
    args = parser.parse_args()
    output_json = args.output_dir / "short_moment_diagnostic.json"
    output_csv = args.output_dir / "short_moment_diagnostic.csv"
    if output_json.exists() or output_csv.exists():
        fail("Refusing to overwrite existing v0B output")

    rows = load_rows(args.input)
    for index, row in enumerate(rows, start=1):
        max_gt_length = parse_float(row, "max_gt_length", index)
        duration = parse_float(row, "duration", index)
        row["gt_duration_bin"] = find_bin(max_gt_length, GT_DURATION_BINS, "max_gt_length")
        row["audio_duration_bin"] = find_bin(duration, AUDIO_DURATION_BINS, "duration")
        row["gt_length_stratum"] = find_bin(max_gt_length, GT_STRATA, "max_gt_length")

    gt_stats = gt_duration_stats(rows)
    audio_stats = audio_duration_by_gt_stratum(rows)
    summary = {
        "audit_version": "failure_audit_v0B",
        "input": str(args.input),
        "n_queries": len(rows),
        "definitions": {
            "gt_duration": "max_gt_length from query_audit.csv, defined in v0A as the maximum relevant GT window length per query",
            "gt_duration_bins": "[0,2), [2,5), [5,10), [10,20), [20,+inf) seconds",
            "temporal_error_matching": "Top-1 prediction matched to the GT window maximizing Top-1 temporal IoU; absolute errors are |predicted boundary - matched GT boundary|; normalized errors divide each absolute error by matched GT_length",
            "gt_length_strata_for_audio_confound": "GT <= 5 sec; 5 sec < GT <= 20 sec; GT > 20 sec, using max_gt_length per query",
            "audio_duration_bins": "[0,60), [60,120), [120,180), [180,240), [240,+inf) seconds",
            "iou_distribution": "Top-1 best IoU against any GT window",
        },
        "gt_duration_bins": gt_stats,
        "audio_duration_by_gt_length_stratum": audio_stats,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(output_csv, gt_stats, audio_stats)

    print("FAILURE_AUDIT_V0B_COMPLETE")
    print("Input: {} queries from {}".format(len(rows), args.input))
    print("GT-duration primary result:")
    for label, stats in gt_stats.items():
        print(
            "  {}: N={}, R1@0.5={:.2f}, R1@0.7={:.2f}, Oracle10@0.7={:.2f}".format(
                label, stats["N"], stats["Top1 R@0.5"], stats["Top1 R@0.7"], stats["Oracle10@0.7"]
            )
        )
    print("Outputs: {}, {}".format(output_json, output_csv))


if __name__ == "__main__":
    main()
