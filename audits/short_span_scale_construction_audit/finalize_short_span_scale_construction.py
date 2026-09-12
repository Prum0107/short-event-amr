#!/usr/bin/env python3
"""Aggregate the frozen short-span scale construction attribution audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BINS = ("0-2s", "2-5s", "5-10s", "10-20s", "20s+")
SHORT_BINS = ("0-2s", "2-5s")
EPS = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else ["duration_bin"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def quantile(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def fit_linear(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    if len(x) < 3:
        return {"N": len(x), "slope": None, "intercept": None, "correlation": None, "residual_q25": None, "residual_median": None, "residual_q75": None}
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(x_arr)), x_arr])
    intercept, slope = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    residuals = y_arr - design @ np.asarray([intercept, slope])
    correlation = float(np.corrcoef(x_arr, y_arr)[0, 1]) if np.std(x_arr) > 0 and np.std(y_arr) > 0 else None
    return {
        "N": len(x),
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": correlation,
        "residual_q25": float(np.quantile(residuals, 0.25)),
        "residual_median": float(np.median(residuals)),
        "residual_q75": float(np.quantile(residuals, 0.75)),
    }


def ols(rows: Sequence[Mapping[str, Any]], outcome: str, include_gt: bool, include_audio: bool, include_center: bool, include_slots: bool) -> dict[str, Any]:
    if len(rows) < 5:
        return {"N": len(rows), "R2": None, "coefficients": {}}
    y = np.asarray([float(row[outcome]) for row in rows], dtype=np.float64)
    matrix: list[list[float]] = [[1.0] for _ in rows]
    names = ["intercept"]

    def append(name: str, values: Sequence[float]) -> None:
        for current, value in zip(matrix, values):
            current.append(float(value))
        names.append(name)

    if include_gt:
        append("log_gt_duration", [float(row["log_gt_duration"]) for row in rows])
    if include_audio:
        append("log_audio_duration", [float(row["log_audio_duration"]) for row in rows])
    if include_center:
        append("center_error_sec", [float(row["center_error_sec"]) for row in rows])
    if include_slots:
        slots = sorted({int(row["query_slot"]) for row in rows})
        for slot in slots[1:]:
            append(f"slot_{slot}", [1.0 if int(row["query_slot"]) == slot else 0.0 for row in rows])
    x = np.asarray(matrix, dtype=np.float64)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residuals ** 2)) / total if total > 0 else 0.0
    return {"N": len(rows), "R2": r2, "coefficients": {name: float(value) for name, value in zip(names, beta)}}


def cohort_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bin_name in BINS:
        selected = [row for row in rows if row["duration_bin"] == bin_name]
        primary = [row for row in selected if row["center_le_1s"] == "1"]
        secondary = [row for row in selected if row["center_le_2s"] == "1"]
        output[bin_name] = {
            "N_queries": len({row["qid"] for row in selected}),
            "N_proposals": len(selected),
            "N_well_centered_le_1s": len(primary),
            "N_queries_well_centered_le_1s": len({row["qid"] for row in primary}),
            "N_well_centered_le_2s": len(secondary),
            "N_queries_well_centered_le_2s": len({row["qid"] for row in secondary}),
        }
    return output


def prepare_centered_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["center_le_1s"] != "1":
            continue
        gt = max(f(row, "gt_duration_sec"), EPS)
        pred = max(f(row, "pred_width_sec"), EPS)
        current = dict(row)
        current["log_gt_duration"] = math.log(gt)
        current["log_audio_duration"] = math.log(max(f(row, "audio_duration_sec"), EPS))
        current["log_pred_width_sec"] = math.log(pred)
        current["log_pred_width_norm"] = math.log(max(f(row, "pred_width_norm"), EPS))
        current["log_gt_width_norm"] = math.log(max(f(row, "gt_width_norm"), EPS))
        current["log_width_gt_ratio"] = math.log(max(f(row, "width_gt_ratio"), EPS))
        current["token_count"] = len(str(row["query"]).split())
        output.append(current)
    return output


def width_response(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = [(bin_name, [row for row in rows if row["duration_bin"] == bin_name]) for bin_name in BINS]
    groups.append(("GT<5s", [row for row in rows if row["duration_bin"] in SHORT_BINS]))
    for label, selected in groups:
        fit_seconds = fit_linear([row["log_gt_duration"] for row in selected], [row["log_pred_width_sec"] for row in selected])
        fit_norm = fit_linear([row["log_gt_width_norm"] for row in selected], [row["log_pred_width_norm"] for row in selected])
        output.append({
            "duration_bin": label,
            "N_proposals": fit_seconds["N"],
            "N_queries": len({row["qid"] for row in selected}),
            "slope_log_pred_width_sec_vs_log_gt_width_sec": fit_seconds["slope"],
            "intercept_log_pred_width_sec_vs_log_gt_width_sec": fit_seconds["intercept"],
            "correlation_seconds": fit_seconds["correlation"],
            "residual_q25_seconds": fit_seconds["residual_q25"],
            "residual_median_seconds": fit_seconds["residual_median"],
            "residual_q75_seconds": fit_seconds["residual_q75"],
            "slope_log_pred_width_norm_vs_log_gt_width_norm": fit_norm["slope"],
            "intercept_log_pred_width_norm_vs_log_gt_width_norm": fit_norm["intercept"],
            "correlation_normalized": fit_norm["correlation"],
        })
    return output


def layerwise_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bin_name in BINS:
        selected = [row for row in rows if row["duration_bin"] == bin_name]
        for layer_name in ("initial", "layer1", "layer2"):
            widths = [f(row, f"{layer_name}_width_sec") for row in selected]
            ratios = [f(row, f"{layer_name}_width_gt_ratio") for row in selected]
            logs = [f(row, f"{layer_name}_abs_log_width_gt_ratio") for row in selected]
            output[f"{bin_name}_{layer_name}"] = {
                "N": len(selected),
                "median_width_sec": median(widths),
                "median_width_gt_ratio": median(ratios),
                "median_abs_log_width_gt_ratio": median(logs),
                "width_sec_q25": quantile(widths, 0.25),
                "width_sec_q75": quantile(widths, 0.75),
            }
        for transition_name in ("log_width_initial_to_layer1", "log_width_layer1_to_layer2"):
            values = [f(row, transition_name) for row in selected]
            output[f"{bin_name}_{transition_name}"] = {
                "N": len(values),
                "median": median(values),
                "q25": quantile(values, 0.25),
                "q75": quantile(values, 0.75),
            }
    return output


def width_numeric_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bin_name in BINS:
        for layer_name in ("layer1", "layer2"):
            selected = [row for row in rows if row["duration_bin"] == bin_name and row["layer"] == layer_name]
            distances = [f(row, "saturation_distance") for row in selected]
            logits = [f(row, "pre_activation_width_logit") for row in selected]
            widths = [f(row, "post_activation_width_norm") for row in selected]
            output[f"{bin_name}_{layer_name}"] = {
                "N": len(selected),
                "median_saturation_distance": median(distances),
                "q25_saturation_distance": quantile(distances, 0.25),
                "q75_saturation_distance": quantile(distances, 0.75),
                "fraction_saturation_distance_lt_0.05": float(np.mean(np.asarray(distances) < 0.05)) if distances else None,
                "median_abs_width_logit": median([abs(value) for value in logits]),
                "width_norm_min": min(widths) if widths else None,
                "width_norm_max": max(widths) if widths else None,
                "width_norm_median": median(widths),
            }
    return output


def matching_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bin_name in BINS:
        selected = [row for row in rows if row["duration_bin"] == bin_name]
        output[bin_name] = {
            "N": len(selected),
            "median_initial_width_sec": median([f(row, "initial_width_sec") for row in selected]),
            "median_matched_width_gt_ratio": median([f(row, "matched_width_gt_ratio") for row in selected]),
            "median_column_cost_margin": median([f(row, "column_cost_margin") for row in selected]),
            "median_assigned_minus_best_cost": median([f(row, "assigned_minus_best_cost") for row in selected]),
            "median_class_cost": median([f(row, "class_cost") for row in selected]),
            "median_span_l1_cost": median([f(row, "span_l1_cost") for row in selected]),
            "median_giou_cost": median([f(row, "giou_cost") for row in selected]),
            "median_total_cost": median([f(row, "total_cost") for row in selected]),
            "median_center_error_sec": median([f(row, "matched_center_error_sec") for row in selected]),
        }
    return output


def counterfactual_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bin_name in BINS:
        selected = [row for row in rows if row["duration_bin"] == bin_name]
        output[bin_name] = {
            "N": len(selected),
            "median_actual_minus_gt_width_cost": median([f(row, "actual_minus_gt_width_cost") for row in selected]),
            "median_actual_minus_duration_matched_cost": median([f(row, "actual_minus_duration_matched_cost") for row in selected]),
            "median_actual_width_l1_cost": median([f(row, "actual_width_l1_cost") for row in selected]),
            "median_gt_width_l1_cost": median([f(row, "gt_width_l1_cost") for row in selected]),
            "median_duration_matched_width_l1_cost": median([f(row, "duration_matched_width_l1_cost") for row in selected]),
            "median_actual_giou_cost": median([f(row, "actual_giou_cost") for row in selected]),
            "median_gt_width_giou_cost": median([f(row, "gt_width_giou_cost") for row in selected]),
        }
    return output


def gradient_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bin_name in BINS:
        selected = [row for row in rows if row["duration_bin"] == bin_name]
        output[bin_name] = {
            "N": len(selected),
            "median_width_l1_loss": median([f(row, "width_l1_loss_unweighted") for row in selected]),
            "median_giou_loss": median([f(row, "giou_loss_unweighted") for row in selected]),
            "median_width_loss_weighted": median([f(row, "width_loss_weighted") for row in selected]),
            "median_width_coordinate_gradient_abs": median([f(row, "width_coordinate_gradient_abs") for row in selected]),
            "median_width_head_output_gradient_abs": median([f(row, "width_head_output_gradient_abs") for row in selected]),
        }
    return output


def audio_duration_summary(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = {
        "GT_1-2s": [row for row in rows if 1 <= float(row["gt_duration_sec"]) < 2],
        "GT_2-3s": [row for row in rows if 2 <= float(row["gt_duration_sec"]) < 3],
        "GT_3-5s": [row for row in rows if 3 <= float(row["gt_duration_sec"]) < 5],
    }
    output: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for label, selected in groups.items():
        unadjusted = fit_linear([row["log_audio_duration"] for row in selected], [row["log_width_gt_ratio"] for row in selected])
        adjusted = ols(selected, "log_width_gt_ratio", include_gt=True, include_audio=True, include_center=True, include_slots=True)
        record = {
            "duration_slice": label,
            "N": len(selected),
            "unadjusted_audio_log_slope": unadjusted["slope"],
            "unadjusted_audio_log_correlation": unadjusted["correlation"],
            "adjusted_model_R2": adjusted["R2"],
            "adjusted_log_audio_duration_coefficient": adjusted["coefficients"].get("log_audio_duration"),
            "adjusted_center_error_coefficient": adjusted["coefficients"].get("center_error_sec"),
        }
        output.append(record)
        details[label] = {"unadjusted": unadjusted, "adjusted": adjusted}
    return output, details


def query_text_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_qid: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row["qid"])
        current = by_qid.setdefault(qid, {"token_count": int(row["token_count"]), "gt_duration_sec": float(row["gt_duration_sec"]), "centered_log_ratios": []})
        current["centered_log_ratios"].append(float(row["log_width_gt_ratio"]))
    values = list(by_qid.values())
    gt_corr = float(np.corrcoef([item["token_count"] for item in values], [item["gt_duration_sec"] for item in values])[0, 1]) if len(values) >= 3 and np.std([item["token_count"] for item in values]) > 0 else None
    ratio_corr = float(np.corrcoef([item["token_count"] for item in values], [np.median(item["centered_log_ratios"]) for item in values])[0, 1]) if len(values) >= 3 and np.std([item["token_count"] for item in values]) > 0 else None
    return {"N_queries": len(values), "token_count_vs_gt_duration_correlation": gt_corr, "token_count_vs_centered_log_width_gt_ratio_correlation": ratio_corr}


def status_hypotheses(
    response: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, str]],
    numeric: Mapping[str, Any],
    matching: Mapping[str, Any],
    gradients: Mapping[str, Any],
    audio_rows: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    models: Mapping[str, Any],
) -> dict[str, str]:
    response_short = next((row for row in response if row["duration_bin"] == "GT<5s"), {})
    slope = response_short.get("slope_log_pred_width_sec_vs_log_gt_width_sec")
    if response_short.get("N_proposals", 0) < 50 or slope is None:
        h1 = "INCONCLUSIVE"
    elif slope < 0.5:
        h1 = "SUPPORTED"
    elif slope < 0.8:
        h1 = "PARTIALLY_SUPPORTED"
    else:
        h1 = "NOT_SUPPORTED"

    slot_model = models.get("slot_only", {}).get("R2")
    gt_model = models.get("gt_only", {}).get("R2")
    if slot_model is None or gt_model is None:
        h2 = "INCONCLUSIVE"
    elif slot_model > gt_model + 0.1:
        h2 = "SUPPORTED"
    elif slot_model > gt_model + 0.03:
        h2 = "PARTIALLY_SUPPORTED"
    else:
        h2 = "NOT_SUPPORTED"

    short_numeric = [value for key, value in numeric.items() if key.startswith(("0-2s_", "2-5s_"))]
    long_numeric = [value for key, value in numeric.items() if key.startswith(("5-10s_", "10-20s_", "20s+_"))]
    short_sat = [value["median_saturation_distance"] for value in short_numeric if value["median_saturation_distance"] is not None]
    long_sat = [value["median_saturation_distance"] for value in long_numeric if value["median_saturation_distance"] is not None]
    short_edge = [value["fraction_saturation_distance_lt_0.05"] for value in short_numeric if value["fraction_saturation_distance_lt_0.05"] is not None]
    long_edge = [value["fraction_saturation_distance_lt_0.05"] for value in long_numeric if value["fraction_saturation_distance_lt_0.05"] is not None]
    if not short_sat or not long_sat:
        h3 = "INCONCLUSIVE"
    elif np.median(short_sat) < np.median(long_sat) and (not long_edge or np.median(short_edge) > np.median(long_edge) + 0.1):
        h3 = "SUPPORTED"
    elif np.median(short_sat) < np.median(long_sat) or (long_edge and np.median(short_edge) > np.median(long_edge)):
        h3 = "PARTIALLY_SUPPORTED"
    else:
        h3 = "NOT_SUPPORTED"

    short_match = [matching[key] for key in SHORT_BINS if key in matching and matching[key]["N"]]
    long_match = [matching[key] for key in BINS[2:] if key in matching and matching[key]["N"]]
    if not short_match or not long_match:
        h4 = "INCONCLUSIVE"
    else:
        short_margin = np.median([item["median_column_cost_margin"] for item in short_match])
        long_margin = np.median([item["median_column_cost_margin"] for item in long_match])
        short_ratio = np.median([item["median_matched_width_gt_ratio"] for item in short_match])
        long_ratio = np.median([item["median_matched_width_gt_ratio"] for item in long_match])
        if short_margin < long_margin and short_ratio > long_ratio:
            h4 = "SUPPORTED"
        elif short_margin < long_margin or short_ratio > long_ratio:
            h4 = "PARTIALLY_SUPPORTED"
        else:
            h4 = "NOT_SUPPORTED"

    short_grad = [gradients[key] for key in SHORT_BINS if key in gradients and gradients[key]["N"]]
    long_grad = [gradients[key] for key in BINS[2:] if key in gradients and gradients[key]["N"]]
    if not short_grad or not long_grad:
        h5 = "INCONCLUSIVE"
    else:
        short_g = np.median([item["median_width_head_output_gradient_abs"] for item in short_grad])
        long_g = np.median([item["median_width_head_output_gradient_abs"] for item in long_grad])
        if short_g < 0.8 * long_g:
            h5 = "SUPPORTED"
        elif short_g < long_g:
            h5 = "PARTIALLY_SUPPORTED"
        else:
            h5 = "NOT_SUPPORTED"

    audio_coeffs = [row.get("adjusted_log_audio_duration_coefficient") for row in audio_rows if row.get("adjusted_log_audio_duration_coefficient") is not None]
    if not audio_coeffs:
        h6 = "INCONCLUSIVE"
    elif all(value > 0 for value in audio_coeffs):
        h6 = "SUPPORTED"
    elif any(value > 0 for value in audio_coeffs):
        h6 = "PARTIALLY_SUPPORTED"
    else:
        h6 = "NOT_SUPPORTED"

    short_centered = sum(cohort.get(bin_name, {}).get("N_well_centered_le_1s", 0) for bin_name in SHORT_BINS)
    short_ratios = []
    for bin_name in SHORT_BINS:
        key = f"{bin_name}_layer2"
        # The final layer summary is added by the caller as a separate field.
        if bin_name in cohort:
            short_ratios.append(cohort[bin_name].get("median_centered_width_gt_ratio"))
    final_ratios = models.get("centered_scale", {})
    ratio_values = [value for value in final_ratios.values() if value is not None]
    if short_centered < 50 or not ratio_values:
        h7 = "INCONCLUSIVE"
    elif np.median(ratio_values) > 2:
        h7 = "SUPPORTED"
    else:
        h7 = "NOT_SUPPORTED"

    return {
        "H1_SHORT_SCALE_INSENSITIVITY": h1,
        "H2_QUERY_SLOT_SCALE_PRIOR": h2,
        "H3_OUTPUT_COORDINATE_COMPRESSION": h3,
        "H4_MATCHING_SCALE_BIAS": h4,
        "H5_OPTIMIZATION_GEOMETRY_IMBALANCE": h5,
        "H6_AUDIO_DURATION_SCALE_EFFECT": h6,
        "H7_LOCATION_SCALE_DISSOCIATION": h7,
    }


def choose_branch(statuses: Mapping[str, str]) -> str:
    supported = [key for key, value in statuses.items() if value in {"SUPPORTED", "PARTIALLY_SUPPORTED"}]
    if statuses["H2_QUERY_SLOT_SCALE_PRIOR"] == "SUPPORTED" and len(supported) == 1:
        return "QUERY_SLOT_SCALE_PRIOR"
    if statuses["H4_MATCHING_SCALE_BIAS"] == "SUPPORTED" and len(supported) == 1:
        return "MATCHING_OPTIMIZATION_GEOMETRY"
    if statuses["H3_OUTPUT_COORDINATE_COMPRESSION"] == "SUPPORTED" and len(supported) == 1:
        return "OUTPUT_WIDTH_PARAMETERIZATION"
    if statuses["H6_AUDIO_DURATION_SCALE_EFFECT"] == "SUPPORTED" and len(supported) == 1:
        return "AUDIO_LENGTH_SCALE_GEOMETRY"
    if len(supported) >= 2:
        return "MULTIPLE_COUPLED_SCALE_FACTORS"
    if not supported:
        return "SCALE_FAILURE_NOT_EXPLAINED"
    return "INCONCLUSIVE"


def write_reports(
    output: Path,
    validation: Mapping[str, Any],
    cohort: Mapping[str, Any],
    response: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, str]],
    layer_summary_data: Mapping[str, Any],
    numeric_summary: Mapping[str, Any],
    matching: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    gradients: Mapping[str, Any],
    audio_rows: Sequence[Mapping[str, Any]],
    audio_details: Mapping[str, Any],
    text_summary: Mapping[str, Any],
    models: Mapping[str, Any],
    statuses: Mapping[str, str],
    branch: str,
    final_scale: Mapping[str, float],
) -> None:
    experiment_card = f"""# Short-span scale construction attribution audit

Current gate: frozen baseline attribution (inference, matching, offline cost, and final-checkpoint gradient readout).

Primary decision: determine which measured factors explain final width error conditional on final center error ≤1 s.

Primary treatment/controls: no model treatment. The analysis conditions on well-centered frozen baseline proposals and uses descriptive comparisons across GT duration, query slot, audio duration, decoder layer, official matching costs, and final-checkpoint width gradients.

Fixed contract: no training, no parameter update, no decoder/loss/matcher/width change, no threshold tuning after results, and no method design.

Primary conclusion ceiling: descriptive attribution only. No factor is called causal without a future controlled intervention.
"""
    (output / "experiment_card.md").write_text(experiment_card, encoding="utf-8")

    response_lines = [
        "# GT-to-predicted width response",
        "",
        "The primary cohort contains all frozen final raw proposals with center error ≤1 s. Rows are proposal-level and may share a query; regressions are descriptive, not independent-sample confirmation.",
        "",
        "| Group | N proposals | N queries | slope seconds | corr seconds | slope normalized | corr normalized |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in response:
        response_lines.append(f"| {row['duration_bin']} | {row['N_proposals']} | {row['N_queries']} | {row['slope_log_pred_width_sec_vs_log_gt_width_sec']} | {row['correlation_seconds']} | {row['slope_log_pred_width_norm_vs_log_gt_width_norm']} | {row['correlation_normalized']} |")
    (output / "gt_to_pred_width_response.md").write_text("\n".join(response_lines) + "\n", encoding="utf-8")

    model_lines = [
        "# Explanatory model",
        "",
        "Outcome: `log(predicted_width / GT_width)` among final proposals with center error ≤1 s. Models are bounded descriptive OLS summaries; coefficients are associations, not causal effects.",
        "",
        "| Model | N | R² |",
        "|---|---:|---:|",
    ]
    for name, value in models.items():
        if isinstance(value, Mapping) and "R2" in value:
            model_lines.append(f"| {name} | {value['N']} | {value['R2']} |")
    model_lines += [
        "",
        f"Query-text token-count check: `{text_summary}`. This uses only the official query string and no external duration labels or uncontrolled clustering.",
        "",
        f"Audio-duration conditional summaries: `{audio_details}`.",
        "",
        "The strongest attribution is the factor combination shown by the largest descriptive R² together with the layerwise and width-head numerical summaries. This is not a trained predictive model and is not evidence that the factor causes the scale error.",
    ]
    (output / "explanatory_model.md").write_text("\n".join(model_lines) + "\n", encoding="utf-8")

    hypothesis_lines = [
        "# Hypothesis assessment",
        "",
        "Predeclared interpretation rules: slopes below 0.5 are evidence of weak short-duration response, 0.5–0.8 is partial, and ≥0.8 is not weak; slot/matching/output/gradient claims require the corresponding descriptive contrast and are not causal; H7 requires substantial centered short-event rows with median width/GT >2.",
        "",
        "| Hypothesis | Status |",
        "|---|---|",
    ]
    for key, value in statuses.items():
        hypothesis_lines.append(f"| {key} | **{value}** |")
    hypothesis_lines += [
        "",
        f"Cohort summary: `{cohort}`.",
        f"Layerwise summary: `{layer_summary_data}`.",
        f"Width-head summary: `{numeric_summary}`.",
        f"Matching summary: `{matching}`.",
        f"Matching-cost counterfactual summary: `{counterfactual}`.",
        f"Gradient summary: `{gradients}`.",
    ]
    (output / "hypothesis_assessment.md").write_text("\n".join(hypothesis_lines) + "\n", encoding="utf-8")

    (output / "next_scientific_branch.md").write_text(
        f"# Next scientific branch\n\n`{branch}`\n\nThis is the least-commitment branch selected from the descriptive attribution results. It is not a method proposal and must receive a separate experiment card before any intervention.\n",
        encoding="utf-8",
    )
    (output / "method_design_gate.md").write_text(
        "# Method-design gate\n\nDecision: **NO**\n\nThe audit is descriptive and uses a single frozen checkpoint. It does not identify a causal intervention variable, and competing explanations among query-slot priors, output geometry, matching, gradients, and audio-duration conditioning remain unresolved at the required level. A future controlled intervention plus an independent confirmation would be required before method design.\n",
        encoding="utf-8",
    )

    summary = {
        "status": "COMPLETE" if validation.get("test_official_same_run", {}).get("official_same_run_pass") else "INVALID",
        "validation": validation,
        "well_centered_cohort": cohort,
        "width_response": list(response),
        "layerwise_summary": layer_summary_data,
        "width_head_summary": numeric_summary,
        "matching_summary": matching,
        "matching_cost_counterfactual_summary": counterfactual,
        "gradient_summary": gradients,
        "audio_duration_summary": list(audio_rows),
        "audio_duration_details": audio_details,
        "query_text_summary": text_summary,
        "explanatory_models": models,
        "hypothesis_status": dict(statuses),
        "next_scientific_branch": branch,
        "method_design_gate": "NO",
        "centered_scale_medians": dict(final_scale),
        "claim_boundary": "frozen-baseline descriptive attribution; no causal mechanism or method claim",
        "training_performed": False,
    }
    write_json(output / "summary.json", summary)


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    validation = json.loads((output / "run_validation.json").read_text(encoding="utf-8"))
    well_rows = read_csv(output / "well_centered_cohort.csv")
    slot_rows = read_csv(output / "query_slot_scale_prior.csv")
    layer_rows = read_csv(output / "layerwise_width_construction.csv")
    numeric_rows = read_csv(output / "width_head_numerics.csv")
    match_rows = read_csv(output / "hungarian_scale_audit.csv")
    counterfactual_rows = read_csv(output / "matching_cost_counterfactual.csv")
    gradient_rows = read_csv(output / "gradient_geometry.csv")
    centered_rows = prepare_centered_rows(well_rows)
    cohort = cohort_summary(well_rows)
    response = width_response(centered_rows)
    layer_data = layerwise_summary(layer_rows)
    numeric_data = width_numeric_summary(numeric_rows)
    matching_data = matching_summary(match_rows)
    counterfactual_data = counterfactual_summary(counterfactual_rows)
    gradient_data = gradient_summary(gradient_rows)
    audio_rows, audio_details = audio_duration_summary(centered_rows)
    text_data = query_text_summary(centered_rows)
    models = {
        "gt_only": ols(centered_rows, "log_width_gt_ratio", include_gt=True, include_audio=False, include_center=False, include_slots=False),
        "audio_only": ols(centered_rows, "log_width_gt_ratio", include_gt=False, include_audio=True, include_center=False, include_slots=False),
        "slot_only": ols(centered_rows, "log_width_gt_ratio", include_gt=False, include_audio=False, include_center=False, include_slots=True),
        "gt_audio": ols(centered_rows, "log_width_gt_ratio", include_gt=True, include_audio=True, include_center=False, include_slots=False),
        "gt_audio_slot_center": ols(centered_rows, "log_width_gt_ratio", include_gt=True, include_audio=True, include_center=True, include_slots=True),
    }
    centered_scale = {
        bin_name: median([f(row, "width_gt_ratio") for row in centered_rows if row["duration_bin"] == bin_name])
        for bin_name in SHORT_BINS
    }
    models["centered_scale"] = centered_scale
    statuses = status_hypotheses(response, slot_rows, numeric_data, matching_data, gradient_data, audio_rows, cohort, models)
    branch = choose_branch(statuses)
    write_csv(output / "gt_to_pred_width_response.csv", response)
    write_csv(output / "audio_duration_conditional_scale.csv", audio_rows)
    write_reports(output, validation, cohort, response, slot_rows, layer_data, numeric_data, matching_data, counterfactual_data, gradient_data, audio_rows, audio_details, text_data, models, statuses, branch, centered_scale)
    print(json.dumps({"status": "COMPLETE" if validation.get("test_official_same_run", {}).get("official_same_run_pass") else "INVALID", "cohort": cohort, "response": response, "hypotheses": statuses, "next_branch": branch}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
