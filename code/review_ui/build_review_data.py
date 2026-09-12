#!/usr/bin/env python3
"""Create the immutable short-moment probe and review manifest."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


SEED = 20260820
TARGET_PER_GROUP = 15
AUDIT_CSV = Path("/private/research-artifact")
PREDICTIONS_JSONL = Path("/private/research-artifact")
REVIEW_DIR = Path("/private/research-artifact")
AUDIO_DIR = REVIEW_DIR / "audio"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["qid"]
            if qid in result:
                raise RuntimeError("Duplicate prediction qid at line {}: {}".format(line_number, qid))
            result[qid] = row
    return result


def audio_regime(row: Dict[str, str]) -> str:
    duration = float(row["duration"])
    if 60.0 <= duration < 120.0:
        return "60-120 sec"
    if 120.0 <= duration < 240.0:
        return "120-240 sec"
    if duration >= 240.0:
        return "240+ sec"
    return "other"


def diagnostic_regime(row: Dict[str, str]) -> str:
    top1_iou = float(row["top1_best_iou"])
    oracle10_iou = float(row["oracle_10_best_iou"])
    if top1_iou < 0.5 and oracle10_iou >= 0.7:
        return "ranking_gap"
    if top1_iou == 0.0:
        return "iou_zero"
    return "partial_or_boundary_overlap"


def choose_group(rows: Sequence[Dict[str, str]], group_name: str, seed_offset: int) -> List[Dict[str, str]]:
    eligible = [row for row in rows if float(row["max_gt_length"]) < 2.0] if group_name == "short_0_2s" else [row for row in rows if 2.0 <= float(row["max_gt_length"]) < 5.0]
    eligible = [row for row in eligible if float(row["top1_best_iou"]) < 0.7]
    if len(eligible) < TARGET_PER_GROUP:
        raise RuntimeError("{} has only {} eligible rows".format(group_name, len(eligible)))
    rng = random.Random(SEED + seed_offset)
    selected: Dict[str, Dict[str, str]] = {}
    regime_order = ["60-120 sec", "120-240 sec", "240+ sec"]
    target_per_regime = TARGET_PER_GROUP // len(regime_order)
    for regime in regime_order:
        pool = [row for row in eligible if audio_regime(row) == regime]
        rng.shuffle(pool)
        for row in pool[:target_per_regime]:
            selected[row["qid"]] = row

    remaining = [row for row in eligible if row["qid"] not in selected]
    remaining.sort(key=lambda row: (diagnostic_regime(row), row["qid"]))
    rng.shuffle(remaining)
    diagnostic_order = ["ranking_gap", "iou_zero", "partial_or_boundary_overlap"]
    for diagnostic in diagnostic_order:
        pool = [row for row in remaining if diagnostic_regime(row) == diagnostic]
        rng.shuffle(pool)
        for row in pool:
            if len(selected) >= TARGET_PER_GROUP:
                break
            selected[row["qid"]] = row
        if len(selected) >= TARGET_PER_GROUP:
            break

    if len(selected) < TARGET_PER_GROUP:
        raise RuntimeError("Could not fill {}: selected {}".format(group_name, len(selected)))
    output = list(selected.values())[:TARGET_PER_GROUP]
    output.sort(key=lambda row: row["qid"])
    return output


def parse_gt_windows(value: str) -> List[List[float]]:
    windows = json.loads(value)
    return [[float(window[0]), float(window[1])] for window in windows]


def build_manifest(rows: Sequence[Dict[str, str]], predictions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    for row in rows:
        qid = row["qid"]
        pred = predictions.get(qid)
        if pred is None:
            raise RuntimeError("Missing prediction for {}".format(qid))
        if pred.get("query") != row["query"] or pred.get("vid") != row["vid"]:
            raise RuntimeError("Prediction metadata mismatch for {}".format(qid))
        manifest.append(
            {
                "qid": qid,
                "vid": row["vid"],
                "query": row["query"],
                "duration": float(row["duration"]),
                "gt_windows": parse_gt_windows(row["gt_windows"]),
                "num_gt_windows": int(row["num_gt_windows"]),
                "max_gt_length": float(row["max_gt_length"]),
                "top1_start": float(row["top1_start"]),
                "top1_end": float(row["top1_end"]),
                "top1_score": float(row["top1_score"]),
                "top1_best_iou": float(row["top1_best_iou"]),
                "oracle10_best_iou": float(row["oracle_10_best_iou"]),
                "audio_duration_bin": row.get("audio_duration_bin") or audio_regime(row),
                "review_group": row["review_group"],
                "diagnostic_regime": diagnostic_regime(row),
                "audio_path": str(AUDIO_DIR / (row["vid"] + ".wav")),
                "audio_url": "audio/{}.wav".format(row["vid"]),
                "pred_relevant_windows": pred["pred_relevant_windows"],
            }
        )
    return manifest


def main() -> None:
    review_data_dir = REVIEW_DIR / "data"
    review_data_dir.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(AUDIT_CSV)
    predictions = read_jsonl(PREDICTIONS_JSONL)
    if len(rows) != 1347 or len(predictions) != 1347:
        raise RuntimeError("Expected 1347 audit rows and predictions")
    short_0_2 = choose_group(rows, "short_0_2s", 0)
    short_2_5 = choose_group(rows, "short_2_5s", 1)
    selected_rows: List[Dict[str, str]] = []
    for group_name, group_rows in (("short_0_2s", short_0_2), ("short_2_5s", short_2_5)):
        for row in group_rows:
            copied = dict(row)
            copied["review_group"] = group_name
            selected_rows.append(copied)
    if len({row["qid"] for row in selected_rows}) != 30:
        raise RuntimeError("Short probe qids are not unique")
    manifest_cases = build_manifest(selected_rows, predictions)
    manifest = {
        "manifest_version": "manual_review_short_probe_30_v1",
        "seed": SEED,
        "selection": {
            "total": 30,
            "0-2 sec": 15,
            "2-5 sec": 15,
            "eligible_condition": "top1_best_iou < 0.7",
            "audio_regimes": ["60-120 sec", "120-240 sec", "240+ sec"],
            "diagnostic_regimes": ["iou_zero", "partial_or_boundary_overlap", "ranking_gap"],
        },
        "audio_root": str(AUDIO_DIR),
        "cases": manifest_cases,
    }
    with (review_data_dir / "review_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (review_data_dir / "reviews.json").open("w", encoding="utf-8") as handle:
        json.dump({}, handle, indent=2)
        handle.write("\n")
    fields = ["qid", "vid", "query", "duration", "max_gt_length", "top1_best_iou", "oracle10_best_iou", "audio_duration_bin", "review_group", "diagnostic_regime", "audio_path"]
    with (review_data_dir / "short_probe_30.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in manifest_cases:
            writer.writerow({field: case[field] for field in fields})
    counts: Dict[str, int] = {}
    for case in manifest_cases:
        counts[case["review_group"]] = counts.get(case["review_group"], 0) + 1
    print("short_probe_30: {}".format(len(manifest_cases)))
    print("group_counts: {}".format(counts))
    print("audio_regimes: {}".format({label: sum(case["audio_duration_bin"] == label for case in manifest_cases) for label in ["60-120 sec", "120-180 sec", "180-240 sec", "240+ sec"]}))
    print("diagnostic_regimes: {}".format({label: sum(case["diagnostic_regime"] == label for case in manifest_cases) for label in ["iou_zero", "partial_or_boundary_overlap", "ranking_gap"]}))


if __name__ == "__main__":
    main()
