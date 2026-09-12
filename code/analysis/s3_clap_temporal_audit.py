#!/usr/bin/env python3
"""Offline S3 MS-CLAP temporal evidence audit.

This script uses cached CASTELLA features and the already validated official
MS-CLAP projection layers.  It never loads raw audio and never runs an AMR
model.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer
from msclap.models.clap import CLAP


ROOT = Path("/private/research-artifact")
GT_PATH = ROOT / "dcase2026_task6_baseline/data/castella_test_release.jsonl"
QD_PATH = ROOT / "dcase2026_task6_baseline/results/submission.jsonl"
UV_PATH = ROOT / "amr_failure_audit/outputs/uvcom/submission_normalized.jsonl"
AUDIO_DIR = ROOT / "dcase2026_task6_baseline/features/castella/clap"
TEXT_DIR = ROOT / "dcase2026_task6_baseline/features/castella/clap_text"
OUT = ROOT / "amr_failure_audit/outputs/clap_evidence"
CHECKPOINT = OUT / "checkpoints/CLAP_weights_2023.pth"
GPT2_DIR = OUT / "hf_models/gpt2"

BINS = [
    ("0–2s", 0.0, 2.0),
    ("2–5s", 2.0, 5.0),
    ("5–10s", 5.0, 10.0),
    ("10–20s", 10.0, 20.0),
    ("20s+", 20.0, math.inf),
]
BIN_LABELS = [x[0] for x in BINS]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def duration_bin(value: float) -> str:
    for label, lo, hi in BINS:
        if lo <= value < hi:
            return label
    raise ValueError(f"duration outside bins: {value}")


def gt_windows(item: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(s), float(e)) for s, e in item["relevant_windows"]]


def iou_and_match(
    proposal: list[float] | tuple[float, ...],
    windows: list[tuple[float, float]],
) -> tuple[float, int | None, float, float, float]:
    ps, pe = float(proposal[0]), float(proposal[1])
    pdur = max(0.0, pe - ps)
    best = (0.0, None, 0.0, 0.0, pdur)
    for gi, (gs, ge) in enumerate(windows):
        inter = max(0.0, min(pe, ge) - max(ps, gs))
        union = max(pe, ge) - min(ps, gs)
        iou = inter / union if union > 0 else 0.0
        gt_dur = max(0.0, ge - gs)
        coverage = inter / gt_dur if gt_dur > 0 else 0.0
        candidate = (iou, gi, coverage, gt_dur, pdur)
        if iou > best[0] + 1e-12:
            best = candidate
    return best


def evaluate_proposals(
    proposals: list[list[float]],
    windows: list[tuple[float, float]],
    limit: int | None,
) -> dict[str, Any]:
    usable = proposals[:limit] if limit is not None else proposals
    if not usable:
        return {
            "iou": 0.0,
            "matched_gt_index": None,
            "duration_ratio": None,
            "gt_coverage": None,
            "prediction_duration": None,
            "positive_overlap": False,
        }
    best: dict[str, Any] | None = None
    for rank, proposal in enumerate(usable, start=1):
        iou, gi, coverage, gt_dur, pdur = iou_and_match(proposal, windows)
        if best is None or iou > best["iou"] + 1e-12:
            best = {
                "iou": float(iou),
                "matched_gt_index": gi,
                "duration_ratio": float(pdur / gt_dur) if gi is not None and gt_dur > 0 else None,
                "gt_coverage": float(coverage) if gi is not None else None,
                "prediction_duration": float(pdur),
                "positive_overlap": bool(iou > 0.0),
                "selected_rank": rank,
            }
    assert best is not None
    return best


def top1_and_oracle10(
    proposals: list[list[float]], windows: list[tuple[float, float]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return evaluate_proposals(proposals, windows, 1), evaluate_proposals(proposals, windows, 10)


def overlap_mask(
    starts: np.ndarray, ends: np.ndarray, windows: list[tuple[float, float]]
) -> np.ndarray:
    mask = np.zeros(len(starts), dtype=bool)
    for gs, ge in windows:
        mask |= (np.minimum(ends, ge) - np.maximum(starts, gs)) > 0.0
    return mask


def width_from_anchor(
    normalized: np.ndarray, anchor: int, threshold: float
) -> tuple[float, int]:
    if anchor < 0 or anchor >= len(normalized) or normalized[anchor] + 1e-12 < threshold:
        return 0.0, 0
    left = anchor
    right = anchor
    while left > 0 and normalized[left - 1] + 1e-12 >= threshold:
        left -= 1
    while right + 1 < len(normalized) and normalized[right + 1] + 1e-12 >= threshold:
        right += 1
    return float((right + 1) - left), int(right - left + 1)


def percentile_stats(values: Iterable[float], denominator: int | None = None) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    d = int(len(vals) if denominator is None else denominator)
    if len(vals) == 0:
        return {"n": 0, "denominator": d, "p25": None, "median": None, "p75": None}
    return {
        "n": int(len(vals)),
        "denominator": d,
        "p25": float(np.percentile(vals, 25)),
        "median": float(np.median(vals)),
        "p75": float(np.percentile(vals, 75)),
    }


def fraction(count: int, denominator: int) -> dict[str, Any]:
    return {"count": int(count), "denominator": int(denominator), "fraction": float(count / denominator) if denominator else None}


def median_of(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return percentile_stats([row.get(key) for row in rows], len(rows))


def safe_median(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row[key] for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return float(np.median(vals)) if vals else None


def spearman(x: list[float], y: list[float]) -> dict[str, Any]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 2:
        return {"n": len(pairs), "rho": None}

    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and values[order[j]] == values[order[i]]:
                j += 1
            ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
            i = j
        return ranks

    xa = np.asarray([p[0] for p in pairs], dtype=float)
    ya = np.asarray([p[1] for p in pairs], dtype=float)
    xr, yr = rank(xa), rank(ya)
    if np.std(xr) == 0 or np.std(yr) == 0:
        rho = 0.0
    else:
        rho = float(np.corrcoef(xr, yr)[0, 1])
    return {"n": len(pairs), "rho": rho}


def subset_model_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def model(name: str) -> dict[str, Any]:
        top_key = f"{name}_top1_iou"
        oracle_key = f"{name}_oracle10_iou"
        ratio_key = f"{name}_best10_duration_ratio"
        top_success = sum(float(row[top_key]) >= 0.7 for row in rows)
        oracle_success = sum(float(row[oracle_key]) >= 0.7 for row in rows)
        positive = [row for row in rows if float(row[oracle_key]) > 0]
        return {
            "N": len(rows),
            "Top1_R1@0.7": fraction(top_success, len(rows)),
            "Oracle10_R@0.7": fraction(oracle_success, len(rows)),
            "Best10_duration_ratio": {
                "n_positive_overlap": sum(row.get(ratio_key) is not None for row in rows),
                "denominator_positive_overlap": len(positive),
                "median": safe_median(rows, ratio_key),
            },
            "both_model_top1_fail": None,
        }

    out = {"QD-DETR": model("QD"), "UVCOM": model("UV")}
    both_fail = sum(float(row["QD_top1_iou"]) < 0.7 and float(row["UV_top1_iou"]) < 0.7 for row in rows)
    both_oracle_fail = sum(float(row["QD_oracle10_iou"]) < 0.7 and float(row["UV_oracle10_iou"]) < 0.7 for row in rows)
    out["both_model_top1_fail"] = fraction(both_fail, len(rows))
    out["both_model_oracle10_fail"] = fraction(both_oracle_fail, len(rows))
    return out


def count_joint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = Counter(row["joint_clap_qd_uvcom_state"] for row in rows)
    states = {}
    for clap in ("MISS", "HIT"):
        for qd in ("FAIL", "SUCCESS"):
            for uv in ("FAIL", "SUCCESS"):
                key = f"CLAP_{clap}__QD_{qd}__UV_{uv}"
                states[key] = fraction(counts.get(key, 0), total)
    return {"N": total, "states": states}


def model_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "N": total,
        "QD-DETR": {
            "R1@0.7": fraction(sum(float(r["QD_top1_iou"]) >= 0.7 for r in rows), total),
            "Oracle10@0.7": fraction(sum(float(r["QD_oracle10_iou"]) >= 0.7 for r in rows), total),
        },
        "UVCOM": {
            "R1@0.7": fraction(sum(float(r["UV_top1_iou"]) >= 0.7 for r in rows), total),
            "Oracle10@0.7": fraction(sum(float(r["UV_oracle10_iou"]) >= 0.7 for r in rows), total),
        },
    }


def concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"thresholds": {}, "score_distribution": {}}
    all_scores: list[float] = []
    constant = 0
    for row in rows:
        all_scores.extend(row["frame_scores"])
        if row["score_range"] <= 1e-12:
            constant += 1
    arr = np.asarray(all_scores, dtype=float)
    out["score_distribution"] = {
        "n_frames": int(len(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "constant_score_queries": int(constant),
        "denominator_queries": len(rows),
        "normalization": "(s_t - s_min)/(s_max - s_min); constant-query range produces zero normalized scores",
    }
    for threshold in (0.8, 0.9):
        suffix = str(int(threshold * 100))
        out["thresholds"][suffix] = {
            "gt_centered_width_sec": percentile_stats([r[f"gt_peak_width_{suffix}"] for r in rows], len(rows)),
            "global_width_sec": percentile_stats([r[f"global_peak_width_{suffix}"] for r in rows], len(rows)),
            "gt_centered_frames": percentile_stats([r[f"gt_peak_frames_{suffix}"] for r in rows], len(rows)),
            "global_frames": percentile_stats([r[f"global_peak_frames_{suffix}"] for r in rows], len(rows)),
        }
    return out


def per_bin(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for label in BIN_LABELS:
        subset = [r for r in rows if r["duration_bin"] == label]
        result[label] = {
            "N": len(subset),
            "clap": {f"Hit@{k}": fraction(sum(r[f"clap_hit{k}"] for r in subset), len(subset)) for k in (1, 3, 5, 10)},
            "best_gt_frame_rank": percentile_stats([r["best_gt_frame_rank"] for r in subset], len(subset)),
            "normalized_best_gt_frame_rank": percentile_stats([r["normalized_best_gt_frame_rank"] for r in subset], len(subset)),
            "global_peak_overlaps_gt": fraction(sum(r["is_global_peak_gt_overlapping"] for r in subset), len(subset)),
            "gt_peak_width_80": percentile_stats([r["gt_peak_width_80"] for r in subset], len(subset)),
            "gt_peak_width_90": percentile_stats([r["gt_peak_width_90"] for r in subset], len(subset)),
            "global_peak_width_80": percentile_stats([r["global_peak_width_80"] for r in subset], len(subset)),
            "global_peak_width_90": percentile_stats([r["global_peak_width_90"] for r in subset], len(subset)),
            "model_reference": model_reference(subset),
        }
    return result


def short_decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for label in ("0–2s", "2–5s"):
        subset = [r for r in rows if r["duration_bin"] == label]
        hit = [r for r in subset if r["clap_hit5"]]
        qd_overlap_low = [r for r in hit if float(r["QD_oracle10_iou"]) > 0 and float(r["QD_oracle10_iou"]) < 0.5]
        uv_overlap_low = [r for r in hit if float(r["UV_oracle10_iou"]) > 0 and float(r["UV_oracle10_iou"]) < 0.5]
        both_zero = [r for r in hit if float(r["QD_oracle10_iou"]) == 0 and float(r["UV_oracle10_iou"]) == 0]

        def count_metric(values: list[dict[str, Any]], key: str, predicate) -> dict[str, Any]:
            n = sum(predicate(r[key]) for r in values)
            return {"count": int(n), "denominator_all": len(subset), "fraction_all": float(n / len(subset)) if subset else None, "denominator_clap_hit5": len(values), "fraction_clap_hit5": float(n / len(values)) if values else None}

        qd_cov_low = [r for r in hit if r["QD_best10_gt_coverage"] is not None and float(r["QD_best10_gt_coverage"]) >= 0.8 and float(r["QD_oracle10_iou"]) < 0.5]
        uv_cov_low = [r for r in hit if r["UV_best10_gt_coverage"] is not None and float(r["UV_best10_gt_coverage"]) >= 0.8 and float(r["UV_oracle10_iou"]) < 0.5]
        out[label] = {
            "N": len(subset),
            "A_CLAP_Hit5_no": fraction(sum(not r["clap_hit5"] for r in subset), len(subset)),
            "B_Hit5_yes_both_Best10_IoU_zero": fraction(len(both_zero), len(subset)),
            "C_Hit5_yes_at_least_one_model_overlap_but_Best10_IoU_lt_0.5": {
                "union": fraction(len(set(r["qid"] for r in qd_overlap_low + uv_overlap_low)), len(subset)),
                "QD": count_metric(qd_overlap_low, "QD_oracle10_iou", lambda x: True),
                "UVCOM": count_metric(uv_overlap_low, "UV_oracle10_iou", lambda x: True),
            },
            "D_Hit5_yes_Best10_GT_coverage_ge_0.8_and_IoU_lt_0.5": {
                "QD": fraction(len(qd_cov_low), len(subset)),
                "UVCOM": fraction(len(uv_cov_low), len(subset)),
                "QD_denominator_clap_hit5": len(hit),
                "UVCOM_denominator_clap_hit5": len(hit),
            },
            "E_Hit5_yes_model_reaches_IoU_ge_0.7": {
                "QD_Best10": fraction(sum(float(r["QD_oracle10_iou"]) >= 0.7 for r in hit), len(subset)),
                "UVCOM_Best10": fraction(sum(float(r["UV_oracle10_iou"]) >= 0.7 for r in hit), len(subset)),
                "QD_Top1": fraction(sum(float(r["QD_top1_iou"]) >= 0.7 for r in hit), len(subset)),
                "UVCOM_Top1": fraction(sum(float(r["UV_top1_iou"]) >= 0.7 for r in hit), len(subset)),
            },
        }
    return out


def quartiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    short = sorted(
        [r for r in rows if r["duration_bin"] in ("0–2s", "2–5s") and r["gt_peak_width_80"] is not None],
        key=lambda r: (float(r["gt_peak_width_80"]), r["qid"]),
    )
    groups = np.array_split(np.asarray(short, dtype=object), 4)
    result = []
    for idx, group in enumerate(groups, start=1):
        subset = list(group)
        widths = [float(r["gt_peak_width_80"]) for r in subset]
        result.append({
            "quartile": f"Q{idx}",
            "N": len(subset),
            "width_range_sec": [min(widths) if widths else None, max(widths) if widths else None],
            "QD_best10_positive_overlap_N": sum(r["QD_best10_duration_ratio"] is not None for r in subset),
            "QD_median_prediction_duration_sec": safe_median(subset, "QD_best10_prediction_duration"),
            "QD_median_duration_ratio": safe_median(subset, "QD_best10_duration_ratio"),
            "UVCOM_best10_positive_overlap_N": sum(r["UV_best10_duration_ratio"] is not None for r in subset),
            "UVCOM_median_prediction_duration_sec": safe_median(subset, "UV_best10_prediction_duration"),
            "UVCOM_median_duration_ratio": safe_median(subset, "UV_best10_duration_ratio"),
        })
    return {"definition": "short queries split into four deterministic equal-count groups sorted by gt_peak_width_80 then qid; model medians use positive-overlap Best10 cases", "quartiles": result}


def subset_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"N": len(rows), "model_metrics": subset_model_metrics(rows)}
    out["QD_best10_positive_overlap_N"] = sum(r["QD_best10_duration_ratio"] is not None for r in rows)
    out["UVCOM_best10_positive_overlap_N"] = sum(r["UV_best10_duration_ratio"] is not None for r in rows)
    out["both_models_top1_fail"] = fraction(sum(float(r["QD_top1_iou"]) < 0.7 and float(r["UV_top1_iou"]) < 0.7 for r in rows), len(rows))
    out["both_models_oracle10_fail"] = fraction(sum(float(r["QD_oracle10_iou"]) < 0.7 and float(r["UV_oracle10_iou"]) < 0.7 for r in rows), len(rows))
    return out


def choose_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    short = [r for r in rows if r["duration_bin"] in ("0–2s", "2–5s") and r["clap_hit5"]]
    categories = {
        "sharp_hit_both_models_fail": [r for r in short if r["sharp_clap_evidence"] and float(r["QD_top1_iou"]) < 0.7 and float(r["UV_top1_iou"]) < 0.7],
        "diffuse_hit_both_models_fail": [r for r in short if r["diffuse_clap_evidence"] and float(r["QD_top1_iou"]) < 0.7 and float(r["UV_top1_iou"]) < 0.7],
        "clap_hit_qd_fail_uv_success": [r for r in short if float(r["QD_top1_iou"]) < 0.7 and float(r["UV_top1_iou"]) >= 0.7],
    }
    selected = []
    for category, candidates in categories.items():
        for rank, row in enumerate(sorted(candidates, key=lambda r: r["qid"])[:5], start=1):
            selected.append({
                "category": category,
                "category_rank": rank,
                "qid": row["qid"],
                "vid": row["vid"],
                "query": row["query"],
                "duration_bin": row["duration_bin"],
                "max_gt_length": row["max_gt_length"],
                "clap_hit5": row["clap_hit5"],
                "gt_peak_width_80": row["gt_peak_width_80"],
                "qd_top1_iou": row["QD_top1_iou"],
                "qd_oracle10_iou": row["QD_oracle10_iou"],
                "uv_top1_iou": row["UV_top1_iou"],
                "uv_oracle10_iou": row["UV_oracle10_iou"],
                "figure": f"figures/{category}_{rank:02d}_{row['qid'].replace('/', '_')}.png",
            })
    return selected


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def write_query_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "qid", "vid", "query", "max_gt_length", "duration_bin", "audio_duration", "feature_covered_duration", "num_clap_frames",
        "num_gt_overlap_frames", "best_gt_cosine", "best_gt_frame_index", "best_gt_frame_rank", "normalized_best_gt_frame_rank",
        "clap_hit1", "clap_hit3", "clap_hit5", "clap_hit10", "global_peak_frame", "global_peak_start", "global_peak_end",
        "is_global_peak_gt_overlapping", "global_peak_width_80", "global_peak_width_90", "gt_peak_width_80", "gt_peak_width_90",
        "QD_top1_iou", "QD_oracle10_iou", "QD_best10_duration_ratio", "QD_best10_gt_coverage", "UVCOM_top1_iou", "UVCOM_oracle10_iou", "UVCOM_best10_duration_ratio", "UVCOM_best10_gt_coverage",
        "joint_clap_qd_uvcom_state", "sharp_clap_evidence", "diffuse_clap_evidence",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: x["qid"]):
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def markdown_report(summary: dict[str, Any], reps: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# S3 MS-CLAP Temporal Evidence Audit")
    lines.append("")
    lines.append("Status: `CLAP_TEMPORAL_EVIDENCE_AUDIT_COMPLETE`")
    lines.append("")
    lines.append("Decision: `PENDING_MANUAL_INTERPRETATION`")
    lines.append("")
    lines.append("This is a diagnostic audit using cached CASTELLA features and the validated official MS-CLAP 2023 projection layers. It does not establish semantic correctness from temporal overlap.")
    lines.append("")
    lines.append("## Provenance and definitions")
    lines.append("")
    lines.append("- Audio cached features are projected from 768 to 1024 dimensions with `clap.audio_encoder.projection`; text cached GPT-2 final non-padding states are projected with `clap.caption_encoder.projection`; projection precedes 1024-D L2 normalization.")
    lines.append("- Frame support is `[t, t+1)` seconds with 1.0-second hop. The official extractor keeps complete windows; the final partial region is dropped. Cached frame count is therefore the number of complete windows, and `feature_covered_duration` is `N_frames` seconds.")
    lines.append("- GT overlap means positive intersection with any annotated GT window. Ranks are 1-based, descending cosine, with ties resolved by lower frame index.")
    lines.append("- `audio_duration` is the released annotation duration; it is not re-read from raw audio.")
    lines.append("")
    lines.append("## Canonical duration table")
    lines.append("")
    lines.append("Rates are fractions with denominator `N`; model columns are AMR IoU metrics and are not numerically equivalent to CLAP frame Hit@K.")
    lines.append("")
    lines.append("| GT bin | N | CLAP H@1 | H@3 | H@5 | H@10 | peak∩GT | median GT-frame rank | QD R1@.7 | QD Oracle10@.7 | UV R1@.7 | UV Oracle10@.7 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in BIN_LABELS:
        b = summary["view_a"][label]
        mr = b["model_reference"]
        lines.append(f"| {label} | {b['N']} | {b['clap']['Hit@1']['fraction']:.4f} | {b['clap']['Hit@3']['fraction']:.4f} | {b['clap']['Hit@5']['fraction']:.4f} | {b['clap']['Hit@10']['fraction']:.4f} | {b['global_peak_overlaps_gt']['fraction']:.4f} | {b['best_gt_frame_rank']['median']:.2f} | {mr['QD-DETR']['R1@0.7']['fraction']:.4f} | {mr['QD-DETR']['Oracle10@0.7']['fraction']:.4f} | {mr['UVCOM']['R1@0.7']['fraction']:.4f} | {mr['UVCOM']['Oracle10@0.7']['fraction']:.4f} |")
    lines.append("")
    lines.append("## GT-centered concentration by bin")
    lines.append("")
    lines.append("Widths are in seconds; when the best GT-overlapping frame itself is below a threshold after within-query range normalization, width is 0 by definition.")
    lines.append("")
    lines.append("| GT bin | N | GT peak width 80% p25/med/p75 | GT peak width 90% p25/med/p75 | global width 80% median | global peak∩GT |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label in BIN_LABELS:
        b = summary["view_a"][label]
        a, c, d = b["gt_peak_width_80"]["p25"], b["gt_peak_width_80"]["median"], b["gt_peak_width_80"]["p75"]
        e, g, h = b["gt_peak_width_90"]["p25"], b["gt_peak_width_90"]["median"], b["gt_peak_width_90"]["p75"]
        lines.append(f"| {label} | {b['N']} | {a:.2f}/{c:.2f}/{d:.2f} | {e:.2f}/{g:.2f}/{h:.2f} | {b['global_peak_width_80']['median']:.2f} | {b['global_peak_overlaps_gt']['fraction']:.4f} |")
    lines.append("")
    lines.append("## Joint CLAP × QD-DETR × UVCOM")
    lines.append("")
    lines.append("The primary AMR success definition is Top1 IoU ≥ 0.7. All eight states are retained; percentages use the duration-group denominator.")
    lines.append("")
    for label, data in [("overall", summary["joint"]["overall"])] + [(label, summary["joint"][label]) for label in BIN_LABELS]:
        lines.append(f"### {label}")
        lines.append("")
        for state, value in data["states"].items():
            lines.append(f"- `{state}`: {value['count']}/{value['denominator']} ({value['fraction']:.4f})")
        lines.append("")
    lines.append("## Short-event decomposition")
    lines.append("")
    for label, data in summary["short_decomposition"].items():
        lines.append(f"### {label} (N={data['N']})")
        lines.append("")
        lines.append(f"- A CLAP Hit@5 miss: {data['A_CLAP_Hit5_no']['count']}/{data['A_CLAP_Hit5_no']['denominator']} ({data['A_CLAP_Hit5_no']['fraction']:.4f})")
        lines.append(f"- B Hit@5 plus both Best10 IoU=0: {data['B_Hit5_yes_both_Best10_IoU_zero']['count']}/{data['B_Hit5_yes_both_Best10_IoU_zero']['denominator']} ({data['B_Hit5_yes_both_Best10_IoU_zero']['fraction']:.4f})")
        c = data['C_Hit5_yes_at_least_one_model_overlap_but_Best10_IoU_lt_0.5']
        lines.append(f"- C Hit@5 plus positive-overlap Best10 IoU<0.5: union {c['union']['count']}/{c['union']['denominator']} ({c['union']['fraction']:.4f}); QD {c['QD']['count']}/{c['QD']['denominator_all']}; UVCOM {c['UVCOM']['count']}/{c['UVCOM']['denominator_all']}")
        d = data['D_Hit5_yes_Best10_GT_coverage_ge_0.8_and_IoU_lt_0.5']
        lines.append(f"- D high GT coverage plus Best10 IoU<0.5: QD {d['QD']['count']}/{d['QD']['denominator']} ({d['QD']['fraction']:.4f}); UVCOM {d['UVCOM']['count']}/{d['UVCOM']['denominator']} ({d['UVCOM']['fraction']:.4f})")
        e = data['E_Hit5_yes_model_reaches_IoU_ge_0.7']
        lines.append(f"- E Hit@5 plus Best10 IoU≥0.7: QD {e['QD_Best10']['count']}/{e['QD_Best10']['denominator']}; UVCOM {e['UVCOM_Best10']['count']}/{e['UVCOM_Best10']['denominator']}; Top1 QD {e['QD_Top1']['count']}/{e['QD_Top1']['denominator']}; Top1 UVCOM {e['UVCOM_Top1']['count']}/{e['UVCOM_Top1']['denominator']}")
        lines.append("")
    lines.append("## Sharp versus diffuse evidence")
    lines.append("")
    for name, data in summary["sharp_diffuse"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- N={data['N']}; both Top1 failures: {data['both_models_top1_fail']['count']}/{data['both_models_top1_fail']['denominator']} ({data['both_models_top1_fail']['fraction']:.4f}); both Oracle10 failures: {data['both_models_oracle10_fail']['count']}/{data['both_models_oracle10_fail']['denominator']} ({data['both_models_oracle10_fail']['fraction']:.4f})")
        for model in ("QD-DETR", "UVCOM"):
            m = data['model_metrics'][model]
            lines.append(f"- {model}: Top1 R1@.7={m['Top1_R1@0.7']['count']}/{m['Top1_R1@0.7']['denominator']}; Oracle10 R@.7={m['Oracle10_R@0.7']['count']}/{m['Oracle10_R@0.7']['denominator']}; median Best10 ratio={m['Best10_duration_ratio']['median']} (positive-overlap N={m['Best10_duration_ratio']['denominator_positive_overlap']})")
        lines.append("")
    lines.append("## Width association")
    lines.append("")
    for model, values in summary["width_association"]["spearman"].items():
        lines.append(f"- {model}: gt_peak_width_80 vs Best10 prediction duration: n={values['prediction_duration']['n']}, rho={values['prediction_duration']['rho']}; vs duration ratio: n={values['duration_ratio']['n']}, rho={values['duration_ratio']['rho']}. Descriptive association only.")
    lines.append("")
    lines.append("## Representative cases")
    lines.append("")
    lines.append("Plots are generated from cached frame scores and annotation/prediction intervals; no raw audio is used.")
    lines.append("")
    lines.append("| Category | qid | GT bin | GT peak width 80% | QD Top1 IoU | UVCOM Top1 IoU | Figure |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for rep in reps:
        lines.append(f"| {rep['category']} | {rep['qid']} | {rep['duration_bin']} | {rep['gt_peak_width_80']:.2f} | {rep['qd_top1_iou']:.3f} | {rep['uv_top1_iou']:.3f} | `{rep['figure']}` |")
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append("Observed: frame-level CLAP evidence, its temporal concentration, and its joint outcome with two AMR systems are measured on all 1,347 queries.")
    lines.append("Supported: descriptive evidence about whether temporal evidence is absent, ranked away from GT, concentrated, or broad, and whether model failures co-occur with each pattern.")
    lines.append("Not supported: semantic correctness of any frame, causality from evidence width to model prediction width, or a method-level remedy.")
    lines.append("")
    lines.append("The final bottleneck label is set after inspecting the generated statistics, rather than by a fixed numeric rule.")
    lines.append("")
    lines.append("## Final questions")
    lines.append("")
    for question, answer in summary.get("final_questions", {}).items():
        lines.append(f"{question}. {answer}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for path in summary["outputs"]:
        lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt_items = read_jsonl(GT_PATH)
    qd_items = {item["qid"]: item for item in read_jsonl(QD_PATH)}
    uv_items = {item["qid"]: item for item in read_jsonl(UV_PATH)}
    gt_by_qid = {item["qid"]: item for item in gt_items}
    if len(gt_by_qid) != len(gt_items):
        raise RuntimeError("duplicate GT qid")
    if set(gt_by_qid) != set(qd_items) or set(gt_by_qid) != set(uv_items):
        raise RuntimeError("GT/QD/UV qid sets are not identical")
    if len(gt_items) != 1347:
        raise RuntimeError(f"expected 1347 queries, got {len(gt_items)}")

    view_counts = Counter(duration_bin(max(e - s for s, e in gt_windows(item))) for item in gt_items)
    expected_counts = {"0–2s": 90, "2–5s": 376, "5–10s": 273, "10–20s": 251, "20s+": 357}
    if dict(view_counts) != expected_counts:
        raise RuntimeError(f"canonical View A count mismatch: {dict(view_counts)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    clap = CLAP(
        audioenc_name="HTSAT",
        sample_rate=44100,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=527,
        out_emb=768,
        text_model=str(GPT2_DIR),
        transformer_embed_dim=768,
        d_proj=1024,
    )
    state = torch.load(CHECKPOINT, map_location="cpu")["model"]
    missing, unexpected = clap.load_state_dict(state, strict=False)
    clap.eval().to(device)
    audio_projection = clap.audio_encoder.projection
    text_projection = clap.caption_encoder.projection
    tokenizer = AutoTokenizer.from_pretrained(str(GPT2_DIR))
    tokenizer.add_special_tokens({"pad_token": "!"})
    print(f"projection_loaded missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in gt_items:
        grouped[item["vid"]].append(item)
    rows: list[dict[str, Any]] = []

    def project_audio(vid: str) -> np.ndarray:
        path = AUDIO_DIR / f"{vid}.npz"
        with np.load(path) as data:
            raw = np.asarray(data["features"], dtype=np.float32)
        with torch.inference_mode():
            projected = audio_projection(torch.from_numpy(raw).to(device)).detach().float().cpu().numpy()
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.maximum(norms, 1e-12)

    def project_text(qid: str, query: str) -> np.ndarray:
        path = TEXT_DIR / f"qid{qid}.npz"
        with np.load(path) as data:
            hidden = np.asarray(data["last_hidden_state"], dtype=np.float32)
        encoded = tokenizer.encode_plus(
            text=query + " <|endoftext|>",
            add_special_tokens=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].reshape(-1)
        sequence_length = int(torch.ne(input_ids, 0).sum().item() - 1)
        if sequence_length < 0 or sequence_length >= hidden.shape[0]:
            raise RuntimeError(f"invalid text pooling index qid={qid}: {sequence_length} shape={hidden.shape}")
        with torch.inference_mode():
            projected = text_projection(torch.from_numpy(hidden[sequence_length]).to(device).unsqueeze(0)).detach().float().cpu().numpy()[0]
        projected /= max(float(np.linalg.norm(projected)), 1e-12)
        return projected

    for vid_index, (vid, items) in enumerate(sorted(grouped.items()), start=1):
        audio = project_audio(vid)
        for item in sorted(items, key=lambda x: x["qid"]):
            qid = item["qid"]
            windows = gt_windows(item)
            max_gt_length = max(e - s for s, e in windows)
            starts = np.arange(len(audio), dtype=float)
            ends = starts + 1.0
            mask = overlap_mask(starts, ends, windows)
            text = project_text(qid, item["query"])
            scores = np.asarray(audio @ text, dtype=np.float64)
            order = np.lexsort((np.arange(len(scores)), -scores))
            gt_indices = np.flatnonzero(mask)
            rank_positions = np.empty(len(order), dtype=int)
            rank_positions[order] = np.arange(1, len(order) + 1)
            if len(gt_indices):
                best_gt_index: int | None = min(gt_indices.tolist(), key=lambda idx: (-float(scores[idx]), int(idx)))
                best_gt_rank: int | None = int(rank_positions[best_gt_index])
            else:
                # A released GT window can fall in the final partial region
                # that the official extractor does not save as a complete
                # frame. Keep the query, but mark GT-frame metrics invalid.
                best_gt_index = None
                best_gt_rank = None
            score_min, score_max = float(np.min(scores)), float(np.max(scores))
            score_range = score_max - score_min
            normalized = (scores - score_min) / score_range if score_range > 1e-12 else np.zeros_like(scores)
            global_index = int(order[0])
            global_peak_width_80, global_peak_frames_80 = width_from_anchor(normalized, global_index, 0.8)
            global_peak_width_90, global_peak_frames_90 = width_from_anchor(normalized, global_index, 0.9)
            if best_gt_index is None:
                gt_peak_width_80 = gt_peak_width_90 = None
                gt_peak_frames_80 = gt_peak_frames_90 = None
            else:
                gt_peak_width_80, gt_peak_frames_80 = width_from_anchor(normalized, best_gt_index, 0.8)
                gt_peak_width_90, gt_peak_frames_90 = width_from_anchor(normalized, best_gt_index, 0.9)
            qd_top1, qd_best10 = top1_and_oracle10(qd_items[qid]["pred_relevant_windows"], windows)
            uv_top1, uv_best10 = top1_and_oracle10(uv_items[qid]["pred_relevant_windows"], windows)
            hits = {k: bool(np.any(mask[order[:k]])) for k in (1, 3, 5, 10)}
            state_name = f"CLAP_{'HIT' if hits[5] else 'MISS'}__QD_{'SUCCESS' if qd_top1['iou'] >= 0.7 else 'FAIL'}__UV_{'SUCCESS' if uv_top1['iou'] >= 0.7 else 'FAIL'}"
            row = {
                "qid": qid,
                "vid": vid,
                "query": item["query"],
                "max_gt_length": float(max_gt_length),
                "duration_bin": duration_bin(max_gt_length),
                "audio_duration": float(item["duration"]),
                "feature_covered_duration": float(len(audio)),
                "num_clap_frames": int(len(audio)),
                "num_gt_overlap_frames": int(len(gt_indices)),
                "best_gt_cosine": float(scores[best_gt_index]) if best_gt_index is not None else None,
                "best_gt_frame_index": best_gt_index,
                "best_gt_frame_rank": best_gt_rank,
                "normalized_best_gt_frame_rank": float(best_gt_rank / len(scores)) if best_gt_rank is not None else None,
                "clap_hit1": hits[1], "clap_hit3": hits[3], "clap_hit5": hits[5], "clap_hit10": hits[10],
                "global_peak_frame": global_index,
                "global_peak_start": float(global_index),
                "global_peak_end": float(global_index + 1),
                "is_global_peak_gt_overlapping": bool(mask[global_index]),
                "global_peak_width_80": global_peak_width_80, "global_peak_width_90": global_peak_width_90,
                "global_peak_frames_80": global_peak_frames_80, "global_peak_frames_90": global_peak_frames_90,
                "gt_peak_width_80": gt_peak_width_80, "gt_peak_width_90": gt_peak_width_90,
                "gt_peak_frames_80": gt_peak_frames_80, "gt_peak_frames_90": gt_peak_frames_90,
                "score_min": score_min, "score_max": score_max, "score_range": score_range,
                "QD_top1_iou": qd_top1["iou"], "QD_oracle10_iou": qd_best10["iou"],
                "QD_best10_duration_ratio": qd_best10["duration_ratio"], "QD_best10_gt_coverage": qd_best10["gt_coverage"],
                "QD_best10_prediction_duration": qd_best10["prediction_duration"] if qd_best10["duration_ratio"] is not None else None,
                "UV_top1_iou": uv_top1["iou"], "UV_oracle10_iou": uv_best10["iou"],
                "UV_best10_duration_ratio": uv_best10["duration_ratio"], "UV_best10_gt_coverage": uv_best10["gt_coverage"],
                "UV_best10_prediction_duration": uv_best10["prediction_duration"] if uv_best10["duration_ratio"] is not None else None,
                "joint_clap_qd_uvcom_state": state_name,
                "sharp_clap_evidence": bool(max_gt_length < 5.0 and hits[5] and gt_peak_width_80 is not None and gt_peak_width_80 <= 2.0),
                "diffuse_clap_evidence": bool(max_gt_length < 5.0 and hits[5] and gt_peak_width_80 is not None and gt_peak_width_80 >= 5.0),
                "gt_windows": [[float(s), float(e)] for s, e in windows],
                "gt_overlap_frame_indices": gt_indices.astype(int).tolist(),
                "frame_starts": starts.tolist(),
                "frame_ends": ends.tolist(),
                "frame_scores": [float(x) for x in scores.tolist()],
            }
            rows.append(row)
        if vid_index % 50 == 0 or vid_index == len(grouped):
            print(f"processed_vids={vid_index}/{len(grouped)} queries={len(rows)}", flush=True)

    rows.sort(key=lambda x: x["qid"])
    if len(rows) != 1347 or len({r["qid"] for r in rows}) != 1347:
        raise RuntimeError("unexpected S3 row count")

    view = per_bin(rows)
    overall = {
        "N": len(rows),
        "clap": {f"Hit@{k}": fraction(sum(r[f"clap_hit{k}"] for r in rows), len(rows)) for k in (1, 3, 5, 10)},
        "best_gt_frame_rank": percentile_stats([r["best_gt_frame_rank"] for r in rows], len(rows)),
        "normalized_best_gt_frame_rank": percentile_stats([r["normalized_best_gt_frame_rank"] for r in rows], len(rows)),
        "global_peak_overlaps_gt": fraction(sum(r["is_global_peak_gt_overlapping"] for r in rows), len(rows)),
        "model_reference": model_reference(rows),
    }
    joint = {"overall": count_joint(rows)}
    for label in BIN_LABELS:
        joint[label] = count_joint([r for r in rows if r["duration_bin"] == label])

    sharp_rows = [r for r in rows if r["sharp_clap_evidence"]]
    diffuse_rows = [r for r in rows if r["diffuse_clap_evidence"]]
    short_positive = [r for r in rows if r["duration_bin"] in ("0–2s", "2–5s") and r["gt_peak_width_80"] is not None]
    width_association = {"definition": "short (<5s) queries with model Best10 IoU>0; model statistics use the corresponding positive-overlap denominator", "spearman": {}}
    for model in ("QD", "UV"):
        positive = [r for r in short_positive if r[f"{model}_best10_duration_ratio"] is not None]
        width_association["spearman"]["QD-DETR" if model == "QD" else "UVCOM"] = {
            "prediction_duration": spearman([r["gt_peak_width_80"] for r in positive], [r[f"{model}_best10_prediction_duration"] for r in positive]),
            "duration_ratio": spearman([r["gt_peak_width_80"] for r in positive], [r[f"{model}_best10_duration_ratio"] for r in positive]),
        }

    reps = choose_representatives(rows)
    summary: dict[str, Any] = {
        "status": "CLAP_TEMPORAL_EVIDENCE_AUDIT_COMPLETE",
        "decision": "PENDING_MANUAL_INTERPRETATION",
        "provenance": {
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256_expected": "2cef4016d47d00eb28d153d522f397222057f95000e9bad6b9583c631284a1e6",
            "msclap_version": "1.3.4",
            "projection": "official msclap.models.clap.CLAP projection layers; audio 768->1024; text 768->1024",
            "raw_audio_used": False,
            "model_rerun": False,
            "feature_regenerated": False,
        },
        "definitions": {
            "view_a": "max_gt_length = maximum duration among all GT windows for a query; bins are [0,2), [2,5), [5,10), [10,20), [20,+inf)",
            "frame_support": "[frame_index, frame_index+1) sec; window=1.0 sec; hop=1.0 sec; complete windows only; final partial region dropped",
            "no_complete_gt_overlap_frame": "if a GT window has no positive intersection with any cached complete frame, CLAP Hit@K is false and GT-frame rank/GT-centered width are null; rank/width denominators report valid N separately",
            "gt_overlap": "positive temporal intersection with any GT window",
            "rank": "1-based descending cosine; ties by lower frame index",
            "clap_hit": "any top-K frame is GT-overlapping",
            "cosine": "L2-normalized official 1024-D joint embeddings; no logit_scale and no smoothing",
            "score_normalization": "within-query (s-s_min)/(s_max-s_min); constant range uses zero normalized scores",
            "sharp_threshold": "short query, CLAP Hit@5, gt_peak_width_80 <= 2 sec; descriptive only",
            "diffuse_threshold": "short query, CLAP Hit@5, gt_peak_width_80 >= 5 sec; descriptive only",
        },
        "input_counts": {"queries": len(rows), "unique_vids": len(grouped), "view_a_counts": dict(view_counts), "queries_without_complete_gt_overlap_frame": sum(r["num_gt_overlap_frames"] == 0 for r in rows)},
        "expected_view_a_counts": expected_counts,
        "overall": overall,
        "view_a": view,
        "joint": joint,
        "short_decomposition": short_decomposition(rows),
        "concentration": concentration(rows),
        "width_association": width_association,
        "quartiles": quartiles(rows),
        "sharp_diffuse": {"SHARP_CLAP_EVIDENCE": subset_report(sharp_rows), "DIFFUSE_CLAP_EVIDENCE": subset_report(diffuse_rows)},
        "representative_cases": reps,
        "final_questions": {
            "1": "pending manual interpretation from the generated CLAP Hit@K, rank, and width statistics",
            "2": "see canonical duration table for 0–2s CLAP Hit@1/3/5/10",
            "3": "see joint states and short-event decomposition for CLAP Hit@5 with both Top1 AMR failures",
            "4": "see GT-centered width statistics for 0–2s and 2–5s",
            "5": "see SHARP_CLAP_EVIDENCE subset",
            "6": "see DIFFUSE_CLAP_EVIDENCE subset and duration-ratio medians",
            "7": "see descriptive Spearman associations and quartiles; no causal inference",
            "8": "pending manual interpretation; no semantic claim is made",
            "9": "not implemented; this audit stops after S3 as requested",
        },
        "outputs": [
            str(OUT / "clap_temporal_evidence_query.csv"),
            str(OUT / "clap_temporal_evidence_summary.json"),
            str(OUT / "clap_temporal_evidence_report.md"),
            str(OUT / "representative_cases.csv"),
            str(OUT / "figures"),
        ],
        "per_query": rows,
    }

    query_csv = OUT / "clap_temporal_evidence_query.csv"
    summary_json = OUT / "clap_temporal_evidence_summary.json"
    report_md = OUT / "clap_temporal_evidence_report.md"
    reps_csv = OUT / "representative_cases.csv"
    write_query_csv(rows, query_csv)
    with reps_csv.open("w", newline="") as f:
        fields = list(reps[0].keys()) if reps else ["category", "category_rank", "qid"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reps)
    with summary_json.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    report_md.write_text(markdown_report(summary, reps))
    print(json.dumps({"status": summary["status"], "rows": len(rows), "view_a_counts": dict(view_counts), "sharp": len(sharp_rows), "diffuse": len(diffuse_rows), "representatives": len(reps)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
