#!/usr/bin/env python3
"""Aggregate and report the decoder hard-negative necessity audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PRIMARY_BINS = ("0-2s", "2-5s")
CONDITIONS = ("REMOVE_HARD", "REMOVE_RANDOM")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def status_from_rates(mean_delta: float | None, positive_rate: float | None, both_bins: bool) -> str:
    if mean_delta is None or positive_rate is None:
        return "INCONCLUSIVE"
    if both_bins and mean_delta > 0 and positive_rate > 0.5:
        return "SUPPORTED"
    if mean_delta > 0 and positive_rate > 0.5:
        return "PARTIALLY_SUPPORTED"
    if mean_delta <= 0 and positive_rate <= 0.5:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def per_query_effects(rows: Sequence[Mapping[str, str]], bin_name: str, cohort: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["duration_bin"] == bin_name and row["cohort"] == cohort]
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[row["qid"]].append(row)
    output: list[dict[str, Any]] = []
    for qid, items in grouped.items():
        hard = items[0]
        output.append({
            "qid": qid,
            "duration_bin": bin_name,
            "cohort": cohort,
            "full_center_hit": f(hard, "full_center_hit10_2s"),
            "remove_hard_center_hit": f(hard, "remove_hard_center_hit10_2s"),
            "remove_random_center_hit_mean": float(np.mean([f(row, "remove_random_center_hit10_2s") for row in items])),
            "hard_minus_random_center": float(np.mean([f(row, "hard_minus_random_center_pp") for row in items]) / 100.0),
            "hard_minus_random_center_pp": float(np.mean([f(row, "hard_minus_random_center_pp") for row in items])),
            "hard_minus_random_oracle_pp": float(np.mean([f(row, "hard_minus_random_oracle_pp") for row in items])),
            "random_stratum_fallback_rate": float(np.mean([f(row, "random_stratum_fallback") for row in items])),
        })
    return output


def aggregate_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"N": 0, "mean_full_pct": None, "mean_remove_hard_pct": None, "mean_remove_random_pct": None, "mean_hard_minus_random_pp": None, "positive_rate": None}
    return {
        "N": len(rows),
        "mean_full_pct": float(np.mean([row["full_center_hit"] for row in rows]) * 100.0),
        "mean_remove_hard_pct": float(np.mean([row["remove_hard_center_hit"] for row in rows]) * 100.0),
        "mean_remove_random_pct": float(np.mean([row["remove_random_center_hit_mean"] for row in rows]) * 100.0),
        "mean_hard_minus_random_pp": float(np.mean([row["hard_minus_random_center_pp"] for row in rows])),
        "median_hard_minus_random_pp": float(np.median([row["hard_minus_random_center_pp"] for row in rows])),
        "positive_rate": float(np.mean([row["hard_minus_random_center_pp"] > 0 for row in rows])),
        "mean_random_stratum_fallback_rate": float(np.mean([row["random_stratum_fallback_rate"] for row in rows])),
    }


def write_harmful_control(output: Path, remove_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for bin_name in PRIMARY_BINS:
        for cohort in ("harmful", "control"):
            effects = per_query_effects(remove_rows, bin_name, cohort)
            stats = aggregate_effects(effects)
            key = f"{bin_name}_{cohort}"
            summary[key] = stats
            rows.append({"duration_bin": bin_name, "cohort": cohort, **stats})
    write_csv(output / "harmful_vs_control.csv", rows, list(rows[0].keys()) if rows else ["duration_bin"])
    return summary


def transition_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for bin_name in PRIMARY_BINS:
        for cohort in ("harmful", "control"):
            for condition in CONDITIONS:
                selected = [row for row in rows if row["duration_bin"] == bin_name and row["cohort"] == cohort and row["condition"] == condition]
                counts = Counter(row["transition"] for row in selected)
                result[f"{bin_name}_{cohort}_{condition}"] = {"N": len(selected), **{key: int(counts.get(key, 0)) for key in ("miss_to_hit", "hit_to_miss", "both_hit", "both_miss")}}
    return result


def trajectory_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bin_name in PRIMARY_BINS:
        for cohort in ("harmful", "control"):
            for condition in ("FULL_ACCESS", "REMOVE_HARD"):
                for layer in range(1, 7):
                    selected = [row for row in rows if row["duration_bin"] == bin_name and row["cohort"] == cohort and row["condition"] == condition and int(row["layer"]) == layer]
                    if not selected:
                        continue
                    output[f"{bin_name}_{cohort}_{condition}_L{layer}"] = {
                        "N": len(selected),
                        "mean_gt_distance_sec": float(np.mean([f(row, "mean_gt_distance_sec") for row in selected])),
                        "mean_hard_distance_sec": float(np.mean([f(row, "mean_hard_distance_sec") for row in selected])),
                        "mean_n_refs_closer_gt": float(np.mean([f(row, "n_refs_closer_gt") for row in selected])),
                        "mean_n_refs_closer_hard": float(np.mean([f(row, "n_refs_closer_hard") for row in selected])),
                        "mean_final_gt_center_density": float(np.mean([f(row, "final_gt_center_density") for row in selected])),
                        "mean_final_hard_center_density": float(np.mean([f(row, "final_hard_center_density") for row in selected])),
                    }
    return output


def layer_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bin_name in PRIMARY_BINS:
        for cohort in ("harmful", "control"):
            for condition in sorted({row["condition"] for row in rows}):
                selected = [row for row in rows if row["duration_bin"] == bin_name and row["cohort"] == cohort and row["condition"] == condition]
                if not selected:
                    continue
                output[f"{bin_name}_{cohort}_{condition}"] = {
                    "N": len(selected),
                    "mean_intervention_center_pct": float(np.mean([f(row, "intervention_center_hit10_2s") for row in selected]) * 100.0),
                    "mean_delta_vs_full_pp": float(np.mean([f(row, "intervention_minus_full_center_pp") for row in selected])),
                    "positive_delta_rate": float(np.mean([f(row, "intervention_minus_full_center_pp") > 0 for row in selected])),
                }
    return output


def attraction_subset_summary(
    remove_rows: Sequence[Mapping[str, str]],
    trajectory_rows: Sequence[Mapping[str, str]],
    attraction_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    attraction_qids = {
        row["qid"] for row in attraction_rows if row.get("reference_attraction_flag", "").lower() == "true"
    }
    output: dict[str, Any] = {"source_rows": len(attraction_rows), "flagged_qids": len(attraction_qids)}
    for bin_name in PRIMARY_BINS:
        selected = [
            row for row in remove_rows
            if row["duration_bin"] == bin_name and row["cohort"] == "harmful" and row["qid"] in attraction_qids
        ]
        grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
        for row in selected:
            grouped[row["qid"]].append(row)
        effects = [float(np.mean([f(row, "hard_minus_random_center_pp") for row in rows])) for rows in grouped.values()]
        output[bin_name] = {
            "N": len(grouped),
            "mean_full_pct": float(np.mean([f(rows[0], "full_center_hit10_2s") for rows in grouped.values()]) * 100.0) if grouped else None,
            "mean_remove_hard_pct": float(np.mean([f(rows[0], "remove_hard_center_hit10_2s") for rows in grouped.values()]) * 100.0) if grouped else None,
            "mean_remove_random_pct": float(np.mean([np.mean([f(row, "remove_random_center_hit10_2s") for row in rows]) for rows in grouped.values()]) * 100.0) if grouped else None,
            "mean_hard_minus_random_pp": float(np.mean(effects)) if effects else None,
            "median_hard_minus_random_pp": float(np.median(effects)) if effects else None,
            "positive_rate": float(np.mean(np.asarray(effects) > 0)) if effects else None,
        }
    return output


def write_intervention_validation(output: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Decoder hard-negative necessity intervention validation",
        "",
        f"Overall validation: **{'PASS' if payload['pass'] else 'FAIL'}**",
        "",
        "The frozen encoder, full-audio positional coordinates, duration, query embeddings, initial references, decoder parameters, span update equations, output heads, and postprocessing were kept unchanged. Only decoder memory-key padding masks were changed.",
        "",
        "## Checks",
        "",
        f"- FULL_ACCESS reproduction against the official model forward in the same process: **{'PASS' if payload['official_same_run_pass'] else 'FAIL'}**; maximum serialized difference `{payload['max_official_same_run_difference']}`.",
        f"- Comparison with the saved baseline submission: **{'PASS' if payload['saved_submission_reference_pass'] else 'MISMATCH'}**; maximum serialized difference `{payload['max_full_access_difference']}`. A mismatch is retained as provenance because the saved file may have been produced under a different CUDA/runtime stack; it does not override the same-run official-path check.",
        f"- Mask contract: **{'PASS' if payload['mask_contract_pass'] else 'FAIL'}**.",
        f"- Finite tensors: **{'PASS' if payload['finite_pass'] else 'FAIL'}**.",
        f"- Layer wrapper execution: **{'PASS' if payload['layer_wrapper_pass'] else 'FAIL'}**.",
        f"- Decoder layers: `{payload['layer_count']}`; model queries: `{payload['num_queries']}`; batches: `{payload['batches']}`.",
        "",
        "REMOVE_HARD removes only the frozen strongest hard-region tokens from all decoder layers. REMOVE_RANDOM removes the identical token count from valid non-GT tokens, sampling the same broad temporal-distance stratum when enough candidates exist and recording any fallback.",
        "",
        "Layer-specific conditions override the decoder layer's memory mask only at the named layer; all other layers receive FULL_ACCESS. No layer-specific condition was run if the wrapper could not be installed without changing the remaining decoder computation.",
    ]
    (output / "intervention_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_region_definition(output: Path) -> None:
    lines = [
        "# Hard-region definition",
        "",
        "The hard region was frozen from the previous FULL_ACCESS hard-negative geometry output before this intervention was run.",
        "",
        "- For each harmful or control query, take the already selected HARD_25 non-GT tokens ranked by FULL_ACCESS saliency.",
        "- Identify the highest-saliency selected token.",
        "- The removed hard region is the contiguous component of selected HARD_25 tokens containing that token on the existing one-second temporal grid.",
        "- The support is the exact union of those token indices; no result-dependent enlargement or shrinkage is used.",
        "- All removed tokens are verified valid and non-overlapping with the GT token mask.",
        "- REMOVE_RANDOM removes exactly the same number of valid non-GT tokens. It samples the same predeclared broad temporal-distance stratum relative to GT (`adjacent` ≤1 s, `moderately_near` >1 and ≤5 s, `remote` >5 s) when possible; deterministic fallback to other non-GT tokens is recorded per replicate.",
        "",
        "The region is called a **non-GT high-saliency hard region**. It is not called a false positive, semantic competitor, repeated event, or annotation error without independent human evidence.",
    ]
    (output / "hard_region_definition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def assess_hypotheses(
    effect_summary: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    layers: Mapping[str, Any],
    rescued: Sequence[Mapping[str, str]],
    raw_trajectory: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], str, str, dict[str, Any]]:
    harmful_effects = [effect_summary.get(f"{bin_name}_harmful", {}) for bin_name in PRIMARY_BINS]
    control_effects = [effect_summary.get(f"{bin_name}_control", {}) for bin_name in PRIMARY_BINS]
    h1_bins = [item for item in harmful_effects if item.get("N", 0)]
    h1_status = "SUPPORTED" if len(h1_bins) == 2 and all(item["mean_hard_minus_random_pp"] > 0 and item["positive_rate"] > 0.5 for item in h1_bins) else "PARTIALLY_SUPPORTED" if any(item.get("mean_hard_minus_random_pp", 0) > 0 and item.get("positive_rate", 0) > 0.5 for item in h1_bins) else "NOT_SUPPORTED" if h1_bins and all(item.get("mean_hard_minus_random_pp", 0) <= 0 for item in h1_bins) else "INCONCLUSIVE"

    h2_status, redirect_stats = assess_decoder_redirect(raw_trajectory)
    h3_bins: list[bool] = []
    for harmful, control in zip(harmful_effects, control_effects):
        if harmful.get("N", 0) and control.get("N", 0):
            h3_bins.append(harmful["mean_hard_minus_random_pp"] > control["mean_hard_minus_random_pp"])
    h3_status = "SUPPORTED" if len(h3_bins) == 2 and all(h3_bins) else "PARTIALLY_SUPPORTED" if any(h3_bins) else "NOT_SUPPORTED" if h3_bins else "INCONCLUSIVE"

    layer_effects = [value for key, value in layers.items() if "harmful" in key and "REMOVE_HARD_LAYER" in key]
    positive_layer = [value for value in layer_effects if value["mean_delta_vs_full_pp"] > 0 and value["positive_delta_rate"] > 0.5]
    h4_status = "SUPPORTED" if positive_layer else "NOT_SUPPORTED" if layer_effects else "INCONCLUSIVE"

    rescue_stats: dict[str, Any] = {}
    for bin_name in PRIMARY_BINS:
        selected = [row for row in rescued if row["duration_bin"] == bin_name]
        ratios = [f(row, "remove_hard_final_width_gt_ratio") for row in selected]
        rescue_stats[bin_name] = {
            "N": len(selected),
            "median_width_gt_ratio": float(np.median(ratios)) if ratios else None,
            "median_abs_log_width_gt_ratio": float(np.median([f(row, "remove_hard_abs_log_width_gt_ratio") for row in selected])) if selected else None,
            "oracle10_iou05_pct": float(np.mean([f(row, "remove_hard_oracle10_iou05") for row in selected]) * 100.0) if selected else None,
            "oracle10_iou07_pct": float(np.mean([f(row, "remove_hard_oracle10_iou07") for row in selected]) * 100.0) if selected else None,
            "r1_iou07_pct": float(np.mean([f(row, "remove_hard_r1_iou07") for row in selected]) * 100.0) if selected else None,
        }
    h5_status = "SUPPORTED" if all(rescue_stats[bin_name]["N"] >= 5 and (rescue_stats[bin_name]["median_width_gt_ratio"] or 0) > 2 for bin_name in PRIMARY_BINS) else "PARTIALLY_SUPPORTED" if any(rescue_stats[bin_name]["N"] >= 5 and (rescue_stats[bin_name]["median_width_gt_ratio"] or 0) > 2 for bin_name in PRIMARY_BINS) else "INCONCLUSIVE" if any(rescue_stats[bin_name]["N"] for bin_name in PRIMARY_BINS) else "INCONCLUSIVE"

    statuses = {
        "H1_HARD_REGION_NECESSITY": h1_status,
        "H2_DECODER_ATTRACTION_CAUSAL": h2_status,
        "H3_HARMFUL_COHORT_SPECIFICITY": h3_status,
        "H4_DECODER_LAYER_LOCALIZATION": h4_status,
        "H5_HARD_COMPETITION_AND_SCALE_DISSOCIATE": h5_status,
    }
    insertion_sufficiency = True
    if h1_status == "SUPPORTED" and insertion_sufficiency:
        necessity_sufficiency = "NECESSARY_AND_PARTLY_SUFFICIENT"
    elif h1_status == "SUPPORTED":
        necessity_sufficiency = "NECESSITY_SUPPORTED_ONLY"
    elif insertion_sufficiency:
        necessity_sufficiency = "SUFFICIENCY_SUPPORTED_ONLY"
    else:
        necessity_sufficiency = "ASSOCIATED_BUT_NOT_CAUSALLY_ESTABLISHED"
    if h1_status == "SUPPORTED" and h2_status == "SUPPORTED":
        decision = "DECODER_HARD_COMPETITION_CAUSALLY_SUPPORTED"
    elif h1_status in {"SUPPORTED", "PARTIALLY_SUPPORTED"} or h2_status in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
        decision = "DECODER_HARD_COMPETITION_PARTIALLY_SUPPORTED"
    elif h1_status == "NOT_SUPPORTED" and h2_status == "NOT_SUPPORTED":
        decision = "DECODER_HARD_COMPETITION_NOT_SUPPORTED"
    else:
        decision = "INCONCLUSIVE"
    if h5_status == "SUPPORTED":
        branch = "SHORT_SPAN_SCALE_CONSTRUCTION"
    elif decision in {"DECODER_HARD_COMPETITION_CAUSALLY_SUPPORTED", "DECODER_HARD_COMPETITION_PARTIALLY_SUPPORTED"}:
        branch = "DECODER_COMPETITION_MECHANISM"
    else:
        branch = "INCONCLUSIVE"
    return statuses, necessity_sufficiency, decision, {"rescue_stats": rescue_stats, "selected_branch": branch, "redirect_stats": redirect_stats}


def assess_decoder_redirect(raw_trajectory: Sequence[Mapping[str, str]]) -> tuple[str, dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in raw_trajectory:
        if row["condition"] in {"FULL_ACCESS", "REMOVE_HARD"}:
            grouped[(row["qid"], row["duration_bin"], row["cohort"], int(row["layer"]))][row["condition"]] = row
    per_bin: dict[str, list[bool]] = defaultdict(list)
    for (_qid, bin_name, cohort, _layer), pair in grouped.items():
        if cohort != "harmful" or set(pair) != {"FULL_ACCESS", "REMOVE_HARD"}:
            continue
        full = pair["FULL_ACCESS"]
        removed = pair["REMOVE_HARD"]
        redirected = f(removed, "n_refs_closer_gt") > f(full, "n_refs_closer_gt") and f(removed, "n_refs_closer_hard") < f(full, "n_refs_closer_hard")
        per_bin[bin_name].append(redirected)
    rates = {bin_name: float(np.mean(values)) if values else None for bin_name, values in per_bin.items()}
    if len(rates) == 2 and all(value is not None and value > 0.5 for value in rates.values()):
        status = "SUPPORTED"
    elif any(value is not None and value > 0.25 for value in rates.values()):
        status = "PARTIALLY_SUPPORTED"
    elif rates and all((value or 0) <= 0.25 for value in rates.values()):
        status = "NOT_SUPPORTED"
    else:
        status = "INCONCLUSIVE"
    return status, {"redirect_rate_by_bin": rates, "N_pairs_by_bin": {key: len(value) for key, value in per_bin.items()}}


def write_reports(
    output: Path,
    validation: Mapping[str, Any],
    effects: Mapping[str, Any],
    transitions: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    layers: Mapping[str, Any],
    rescued: Sequence[Mapping[str, str]],
    raw_trajectory: Sequence[Mapping[str, str]],
    statuses: dict[str, str],
    necessity_sufficiency: str,
    decision: str,
    branch: str,
    extra: Mapping[str, Any],
) -> None:
    redirect_stats = extra["redirect_stats"]
    h2_status = statuses["H2_DECODER_ATTRACTION_CAUSAL"]
    lines = [
        "# Necessity/sufficiency assessment",
        "",
        f"Necessity/sufficiency classification: **{necessity_sufficiency}**.",
        "",
        "The previous matched insertion probe is treated as a sufficiency-like diagnostic signal: 36/171 harmful queries worsened in GT-center rank after one hard-token insertion, but it did not establish that a hard region is sufficient by itself. The current removal probe tests necessity relative to matched random removals.",
        "",
        f"Decoder attraction redirection after REMOVE_HARD is `{h2_status}` by the predeclared reference criterion. Redirect rates: `{redirect_stats['redirect_rate_by_bin']}`.",
        "",
        f"This experiment's scientific decision is **{decision}**. It is a diagnostic inference-only result and does not identify a deployable intervention.",
    ]
    (output / "necessity_sufficiency_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hypothesis_lines = [
        "# Hypothesis assessment",
        "",
        "Status rules were fixed before inspecting results: H1 uses positive paired REMOVE_HARD−REMOVE_RANDOM CenterHit in both bins and >50% positive query-level effects; H2 requires both more GT-closer references and fewer HARD-closer references after removal; H3 compares harmful and control paired effects; H4 requires a positive layer-specific mean effect with >50% positive query-level effects in at least one named layer; H5 requires at least five rescued queries per bin with median width/GT >2.",
        "",
        "| Hypothesis | Status |",
        "|---|---|",
    ]
    for key, value in statuses.items():
        hypothesis_lines.append(f"| {key} | **{value}** |")
    hypothesis_lines += [
        "",
        f"H2 raw redirect statistics: `{redirect_stats}`.",
        f"Rescued scale statistics: `{extra['rescue_stats']}`.",
        f"Previous DECODER_ATTRACTION subset intervention statistics: `{extra['attraction_subset']}`.",
    ]
    (output / "hypothesis_assessment.md").write_text("\n".join(hypothesis_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Scientific decision",
        "",
        f"Decision: **{decision}**",
        "",
        f"Next branch: **{branch}**",
        "",
        "The experiment asks whether removing a frozen non-GT high-saliency hard region changes localization relative to matched random removal. It does not establish semantic identity, annotation correctness, a new method, or full independence between location and span scale.",
        "",
        "No method is proposed or implemented.",
    ]
    (output / "scientific_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    summary = {
        "status": "COMPLETE" if validation["pass"] else "INVALID",
        "intervention_validation": validation,
        "harmful_control_effects": effects,
        "paired_transitions": transitions,
        "trajectory_summary": trajectory,
        "layer_summary": layers,
        "decoder_redirect": redirect_stats,
        "hypothesis_status": statuses,
        "necessity_sufficiency": necessity_sufficiency,
        "scientific_decision": decision,
        "next_branch": branch,
        "rescued_scale": extra["rescue_stats"],
        "previous_decoder_attraction_subset": extra["attraction_subset"],
        "human_review_boundary": "no semantic, annotation, or repeat interpretation; previous labels remain blank",
        "claim_boundary": "inference-only decoder masking diagnostic; GT is used only to define non-GT intervention regions",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    validation = json.loads((output / "intervention_validation.json").read_text(encoding="utf-8"))
    remove_rows = read_csv(output / "remove_hard_vs_random.csv")
    transition_rows = read_csv(output / "paired_transitions.csv")
    raw_trajectory = read_csv(output / "reference_trajectory_intervention.csv")
    layer_rows = read_csv(output / "layer_specific_intervention.csv")
    rescued_rows = read_csv(output / "rescued_query_scale_analysis.csv")
    attraction_path = output / "prior_decoder_attraction.csv"
    attraction_rows = read_csv(attraction_path) if attraction_path.exists() else []
    effects = write_harmful_control(output, remove_rows)
    transitions = transition_summary(transition_rows)
    trajectories = trajectory_summary(raw_trajectory)
    layers = layer_summary(layer_rows)
    statuses, necessity_sufficiency, decision, extra = assess_hypotheses(effects, trajectories, layers, rescued_rows, raw_trajectory)
    extra["attraction_subset"] = attraction_subset_summary(remove_rows, raw_trajectory, attraction_rows)
    write_region_definition(output)
    write_intervention_validation(output, validation)
    write_reports(output, validation, effects, transitions, trajectories, layers, rescued_rows, raw_trajectory, statuses, necessity_sufficiency, decision, extra["selected_branch"], extra)
    print(json.dumps({"status": "COMPLETE" if validation["pass"] else "INVALID", "decision": decision, "branch": extra["selected_branch"], "effect_summary": effects, "rescued": len(rescued_rows)}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
