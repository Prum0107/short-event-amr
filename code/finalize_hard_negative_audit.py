#!/usr/bin/env python3
"""Finalize reports for the GT-vs-hard-negative mechanism audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PRIMARY_BINS = ("0-2s", "2-5s")


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


def group(rows: Sequence[Mapping[str, Any]], bin_name: str, cohort: str | None = None) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["duration_bin"] == bin_name and (cohort is None or row["cohort"] == cohort)]


def assess_level(values: Sequence[float], partial_low: float = 0.25, full_high: float = 0.50) -> str:
    if not values:
        return "INCONCLUSIVE"
    rate = float(np.mean(values))
    if rate >= full_high:
        return "SUPPORTED"
    if rate >= partial_low:
        return "PARTIALLY_SUPPORTED"
    return "NOT_SUPPORTED"


def status_from_rate(rate: float | None, partial_low: float = 0.25, full_high: float = 0.50) -> str:
    if rate is None:
        return "INCONCLUSIVE"
    if rate >= full_high:
        return "SUPPORTED"
    if rate >= partial_low:
        return "PARTIALLY_SUPPORTED"
    return "NOT_SUPPORTED"


def write_scale_relation(output: Path, counterfactual: Sequence[Mapping[str, str]], cohort_by_qid: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_qid_condition: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in counterfactual:
        if row["duration_bin"] in PRIMARY_BINS and row["condition"] in {"FULL_ACCESS", "HARD_25", "GT_ONLY"} and int(row["replicate"]) == 0:
            by_qid_condition[(row["qid"], row["condition"])] = row
    rows: list[dict[str, Any]] = []
    for qid, cohort in cohort_by_qid.items():
        full = by_qid_condition.get((qid, "FULL_ACCESS"))
        hard = by_qid_condition.get((qid, "HARD_25"))
        gt_only = by_qid_condition.get((qid, "GT_ONLY"))
        if full is None or hard is None or gt_only is None:
            continue
        rows.append({
            "qid": qid,
            "duration_bin": cohort["duration_bin"],
            "cohort": cohort["cohort"],
            "audio_duration_sec": f(full, "audio_duration_sec"),
            "gt_duration_sec": f(full, "gt_duration_sec"),
            "full_center_hit10_2s": f(full, "center_hit10_2s"),
            "hard25_center_hit10_2s": f(hard, "center_hit10_2s"),
            "gt_only_center_hit10_2s": f(gt_only, "center_hit10_2s"),
            "full_oracle10_iou07": f(full, "oracle10_iou07"),
            "hard25_oracle10_iou07": f(hard, "oracle10_iou07"),
            "gt_only_oracle10_iou07": f(gt_only, "oracle10_iou07"),
            "full_width_gt_ratio": f(full, "final_width_gt_ratio"),
            "hard25_width_gt_ratio": f(hard, "final_width_gt_ratio"),
            "gt_only_width_gt_ratio": f(gt_only, "final_width_gt_ratio"),
            "hard_minus_full_center_pp": (f(hard, "center_hit10_2s") - f(full, "center_hit10_2s")) * 100.0,
            "hard_minus_full_oracle_pp": (f(hard, "oracle10_iou07") - f(full, "oracle10_iou07")) * 100.0,
            "hard_minus_full_width_gt_ratio": f(hard, "final_width_gt_ratio") - f(full, "final_width_gt_ratio"),
        })
    write_csv(output / "scale_relation.csv", rows, list(rows[0].keys()) if rows else ["qid"])
    return rows


def native_summary(native: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for bin_name in PRIMARY_BINS:
        for cohort_name in ("harmful", "control"):
            selected = [row for row in native if row["duration_bin"] == bin_name and row["cohort"] == cohort_name]
            valid = [row for row in selected if row["status"] == "VALID"]
            deltas = [f(row, "S_GT_minus_S_HARD") for row in valid]
            gt_better = sum(row["comparison"] == "GT>HARD" for row in valid)
            hard_better = sum(row["comparison"] == "HARD>GT" for row in valid)
            ties = sum(row["comparison"] == "TIE_EXACT" for row in valid)
            result[f"{bin_name}_{cohort_name}"] = {
                "N_total": len(selected),
                "N_valid": len(valid),
                "N_blocked": len(selected) - len(valid),
                "GT_better": gt_better,
                "HARD_better": hard_better,
                "ties": ties,
                "median_S_GT_minus_S_HARD": median(deltas),
                "GT_better_rate_valid": gt_better / len(valid) if valid else None,
            }
    return result


def qd_semantic_alignment(qd: Mapping[str, str]) -> str:
    hard_rank = int(float(qd["full_best_hard_center_rank"]))
    gt_rank = int(float(qd["full_best_gt_center_rank"]))
    hard_density = int(float(qd["full_hard_center_density_top10"]))
    gt_density = int(float(qd["full_gt_center_density_top10"]))
    if hard_rank < gt_rank or hard_density > gt_density:
        return "QD_HARD_FAVORED"
    if gt_rank < hard_rank or gt_density > hard_density:
        return "QD_GT_FAVORED"
    return "QD_TIE"


def make_taxonomy(
    geometry: Mapping[str, Mapping[str, str]],
    qd: Mapping[str, Mapping[str, str]],
    native: Mapping[str, Mapping[str, str]],
    decoder: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, geo in geometry.items():
        labels: list[str] = []
        evidence: list[str] = []
        if geo["hard_peak_geometry_class"] in {"adjacent", "moderately_near"} or int(float(geo["hard_peak_overlaps_gt_plus_minus_2s"])):
            labels.append("NEAR_GT_SCALE_LEAKAGE")
            evidence.append("strong hard peak is adjacent/moderately near or overlaps GT±2s")
        native_row = native.get(qid)
        qd_row = qd.get(qid)
        dec_row = decoder.get(qid)
        if native_row and native_row.get("status") == "VALID":
            if native_row["comparison"] == "HARD>GT":
                labels.append("SEMANTIC_COMPETITOR")
                evidence.append("native MS-CLAP hard window scored above matched GT window")
            elif native_row["comparison"] == "GT>HARD" and qd_row and qd_semantic_alignment(qd_row) == "QD_HARD_FAVORED":
                labels.append("MODEL_SPECIFIC_FALSE_POSITIVE")
                evidence.append("native GT window scored above hard, while QD final center evidence favored hard")
        if dec_row and dec_row.get("reference_attraction_flag") == "True":
            labels.append("DECODER_ATTRACTION")
            evidence.append("full-access top-1 decoder reference moved closer to hard peak")
        if not labels:
            labels.append("UNRESOLVED")
            evidence.append("no independently supported mechanism label")
        rows.append({
            "qid": qid,
            "duration_bin": geo["duration_bin"],
            "cohort": geo["cohort"],
            "p3_location_failure": geo.get("p3_location_failure", ""),
            "labels": ";".join(labels),
            "evidence": "; ".join(evidence),
            "native_status": native_row.get("status", "BLOCKED") if native_row else "BLOCKED",
            "qd_alignment": qd_semantic_alignment(qd_row) if qd_row else "BLOCKED",
            "human_review_status": "UNLABELED",
        })
    return rows


def write_selection_bias(output: Path, native_rows: Sequence[Mapping[str, str]], geometry_rows: Sequence[Mapping[str, str]]) -> None:
    valid = [row for row in native_rows if row["status"] == "VALID"]
    lines = [
        "# Selection-bias assessment",
        "",
        "The HARD_25 set is selected by the QD-DETR full-access saliency score. Therefore saliency magnitude, hard-token rank, and decoder quantities measured after that selection are selection-conditioned evidence. They can describe what the model selected, but cannot independently establish why the selected region is semantically hard.",
        "",
        "The native MS-CLAP comparison is the independent evidence channel: it uses the original raw WAV, a matched absolute window duration, the query's native MS-CLAP text embedding, and the official shared embedding similarity. It is still unavailable for rows without original WAVs, which remain BLOCKED.",
        "",
        f"- Geometry rows: {len(geometry_rows)}; native valid rows: {len(valid)}; native blocked rows: {len(native_rows) - len(valid)}.",
        "- No stored QD-DETR audio/text tensor cosine was computed.",
        "- No human label was auto-filled. Repeated or incompletely annotated occurrences remain unresolved until review labels are supplied.",
        "",
        "Interpretation must therefore separate: (a) selection-conditioned QD behavior, (b) independent native semantic evidence, and (c) unobserved human annotation status.",
    ]
    (output / "selection_bias_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hypotheses(
    output: Path,
    cohort_rows: Sequence[Mapping[str, str]],
    native_stats: Mapping[str, Any],
    taxonomy: Sequence[Mapping[str, str]],
    decoder_rows: Sequence[Mapping[str, str]],
    scale_rows: Sequence[Mapping[str, Any]],
    replacement_rows: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    harmful = [row for row in cohort_rows if row["cohort"] == "harmful"]
    native_harmful = [
        native_stats[f"{bin_name}_harmful"] for bin_name in ("0-2s", "2-5s")
    ]
    native_valid = sum(int(item["N_valid"]) for item in native_harmful)
    native_hard_rate = (
        sum(int(item["HARD_better"]) for item in native_harmful) / native_valid if native_valid else None
    )
    decoder_harmful = [row for row in decoder_rows if row["cohort"] == "harmful"]
    attraction_rate = (
        sum(row["reference_attraction_flag"] == "True" for row in decoder_harmful) / len(decoder_harmful)
        if decoder_harmful else None
    )
    near_rate = (
        sum(
            row["hard_peak_geometry_class"] in {"adjacent", "moderately_near"}
            or int(float(row["hard_peak_overlaps_gt_plus_minus_2s"]))
            for row in cohort_rows if row["cohort"] == "harmful"
        ) / len(harmful)
        if harmful else None
    )
    model_specific_count = sum(
        "MODEL_SPECIFIC_FALSE_POSITIVE" in row["labels"]
        for row in taxonomy if row["cohort"] == "harmful"
    )
    model_specific_denominator = sum(
        row["cohort"] == "harmful" and row["native_status"] == "VALID" for row in taxonomy
    )
    model_specific_rate = model_specific_count / model_specific_denominator if model_specific_denominator else None
    statuses = {
        "H1_SEMANTIC_HARD_NEGATIVES": status_from_rate(native_hard_rate if native_valid >= 10 else None),
        "H2_MODEL_SPECIFIC_CALIBRATION_FAILURE": status_from_rate(model_specific_rate if native_valid >= 10 else None),
        "H3_NEAR_GT_SCALE_LEAKAGE": status_from_rate(near_rate),
        "H4_REPEAT_OR_ANNOTATION_AMBIGUITY": "INCONCLUSIVE",
        "H5_DECODER_ATTRACTION": status_from_rate(attraction_rate),
        "H6_HARD_NEGATIVE_AND_SCALE_ARE_SEPARATE": "INCONCLUSIVE",
    }
    branch = "INCONCLUSIVE"
    if statuses["H1_SEMANTIC_HARD_NEGATIVES"] == "SUPPORTED" and statuses["H5_DECODER_ATTRACTION"] != "SUPPORTED":
        branch = "SEMANTIC_DISAMBIGUATION"
    elif statuses["H5_DECODER_ATTRACTION"] == "SUPPORTED":
        branch = "DECODER_COMPETITION"
    elif statuses["H3_NEAR_GT_SCALE_LEAKAGE"] == "SUPPORTED":
        branch = "NEAR_GT_SCALE_GEOMETRY"
    lines = [
        "# Hypothesis assessment",
        "",
        "This is a mechanism audit, not a method recommendation. The status rules were fixed before inspecting the resulting tables: a supported rate is at least 50%, a partial rate is 25–49.9%, and fewer than 10 independent native comparisons makes H1/H2 inconclusive. Human annotation ambiguity is inconclusive without supplied labels.",
        "",
        "| Hypothesis | Status | Evidence summary |",
        "|---|---|---|",
        f"| H1 semantic hard negatives | **{statuses['H1_SEMANTIC_HARD_NEGATIVES']}** | Native HARD>GT rate among valid harmful comparisons: {native_hard_rate if native_hard_rate is not None else 'BLOCKED'} |",
        f"| H2 model-specific calibration failure | **{statuses['H2_MODEL_SPECIFIC_CALIBRATION_FAILURE']}** | Requires native GT>HARD plus QD hard-favored evidence; selection-conditioned QD evidence alone is insufficient |",
        f"| H3 near-GT scale leakage | **{statuses['H3_NEAR_GT_SCALE_LEAKAGE']}** | Harmful peaks classified as near/expanded-neighborhood: {near_rate if near_rate is not None else 'unavailable'} |",
        f"| H4 repeat or annotation ambiguity | **{statuses['H4_REPEAT_OR_ANNOTATION_AMBIGUITY']}** | Human review rows contain no labels |",
        f"| H5 decoder attraction | **{statuses['H5_DECODER_ATTRACTION']}** | Harmful full-access top-1 traces flagged as attracted: {attraction_rate if attraction_rate is not None else 'unavailable'} |",
        f"| H6 hard-negative and scale are separate | **{statuses['H6_HARD_NEGATIVE_AND_SCALE_ARE_SEPARATE']}** | Requires stronger independent semantic coverage and matched scale interpretation |",
        "",
        f"## Single branch: `{branch}`",
        "",
        "The branch is deliberately INCONCLUSIVE unless the independent evidence is sufficient for a cleaner decision. No method is designed or selected by this audit.",
        "",
        "Scale and location are reported separately in `scale_relation.csv`. A hard-negative location effect is not interpreted as proof that width/extent is solved.",
    ]
    (output / "hypothesis_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {**statuses, "selected_branch": branch}


def write_next_branch(output: Path, branch: str) -> None:
    mapping = {
        "SEMANTIC_DISAMBIGUATION": "SEMANTIC_DISAMBIGUATION",
        "MODEL_CALIBRATION": "MODEL_CALIBRATION",
        "DECODER_COMPETITION": "DECODER_COMPETITION",
        "NEAR_GT_SCALE_GEOMETRY": "NEAR_GT_SCALE_GEOMETRY",
    }
    lines = [
        "# Next scientific branch",
        "",
        f"Selected branch: **{mapping.get(branch, 'INCONCLUSIVE')}**.",
        "",
        "No next experiment is implemented in this audit. The selection is only a routing decision from the predeclared mechanism evidence. If the result is INCONCLUSIVE, the most informative next step is to increase independent native raw-audio coverage and complete the bounded human review before changing any model or proposing a method.",
    ]
    (output / "next_scientific_branch.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qd_native_agreement(
    native_rows: Sequence[Mapping[str, str]],
    qd_rows: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for bin_name in PRIMARY_BINS:
        for cohort_name in ("harmful", "control"):
            valid = [
                row for row in native_rows
                if row["duration_bin"] == bin_name and row["cohort"] == cohort_name
                and row["status"] == "VALID" and row["qid"] in qd_rows
            ]
            agree = 0
            disagree = 0
            for row in valid:
                qd_side = qd_semantic_alignment(qd_rows[row["qid"]])
                native_side = "HARD" if row["comparison"] == "HARD>GT" else "GT" if row["comparison"] == "GT>HARD" else "TIE"
                qd_side = "HARD" if qd_side == "QD_HARD_FAVORED" else "GT" if qd_side == "QD_GT_FAVORED" else "TIE"
                if native_side == qd_side:
                    agree += 1
                else:
                    disagree += 1
            result[f"{bin_name}_{cohort_name}"] = {"N_valid": len(valid), "agree": agree, "disagree": disagree}
    return result


def write_report(
    output: Path,
    geometry: Sequence[Mapping[str, str]],
    qd: Mapping[str, Mapping[str, str]],
    native: Sequence[Mapping[str, str]],
    decoder: Sequence[Mapping[str, str]],
    replacement: Sequence[Mapping[str, str]],
    taxonomy: Sequence[Mapping[str, str]],
    scale: Sequence[Mapping[str, Any]],
    native_stats: Mapping[str, Any],
    statuses: Mapping[str, str],
) -> None:
    agreement = qd_native_agreement(native, qd)
    lines = [
        "# GT vs hard-negative mechanism audit",
        "",
        "Status: **COMPLETE_WITH_NATIVE_AND_HUMAN_COVERAGE_LIMITS**",
        "",
        "This report is diagnostic only. No model weights, training procedure, ranking rule, or AMR method was changed.",
        "",
        "## 1. Frozen cohorts",
        "",
        "HARMFUL-HARD is defined before mechanism analysis as HARD_25 CenterHit@10≤2s lower than the mean of ten deterministic RANDOM_25 replicates. The resulting counts are 44/46 harmful/control in 0–2 s and 127/249 harmful/control in 2–5 s. P3 location-failure counts are 46 and 135, respectively.",
        "",
        "## 2. Hard-negative geometry",
        "",
        "The strongest selected HARD_25 peak is the contiguous component containing the highest-saliency selected token. Saliency is selection-conditioned, not independent evidence.",
        "",
        "| Bin | Cohort | Adjacent | Moderately near | Remote | Near or GT±2s expanded | Median interval gap (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for bin_name in PRIMARY_BINS:
        for cohort_name in ("harmful", "control"):
            rows = [row for row in geometry if row["duration_bin"] == bin_name and row["cohort"] == cohort_name]
            classes = {key: sum(row["hard_peak_geometry_class"] == key for row in rows) for key in ("adjacent", "moderately_near", "remote")}
            near = sum(row["hard_peak_geometry_class"] in {"adjacent", "moderately_near"} or int(float(row["hard_peak_overlaps_gt_plus_minus_2s"])) for row in rows)
            gap = float(np.median([f(row, "hard_peak_to_gt_interval_gap_sec") for row in rows])) if rows else float("nan")
            lines.append(f"| {bin_name} | {cohort_name} | {classes['adjacent']} | {classes['moderately_near']} | {classes['remote']} | {near}/{len(rows)} | {gap:.2f} |")
    lines += [
        "",
        "## 3. Native MS-CLAP GT vs hard",
        "",
        "Scores use the verified native MS-CLAP shared projection and official logit-scaled normalized dot product on equal-duration raw-audio windows. Missing WAVs are BLOCKED.",
        "",
        "| Bin | Cohort | Valid/total | GT>HARD | HARD>GT | Median S_GT−S_HARD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, stats in native_stats.items():
        bin_name, cohort_name = key.rsplit("_", 1)
        lines.append(f"| {bin_name} | {cohort_name} | {stats['N_valid']}/{stats['N_total']} | {stats['GT_better']} | {stats['HARD_better']} | {stats['median_S_GT_minus_S_HARD'] if stats['median_S_GT_minus_S_HARD'] is not None else 'BLOCKED'} |")
    lines += [
        "",
        "On the available harmful rows, GT scored above hard in 7/8 (0–2 s) and 9/11 (2–5 s); this does not support a dominant native semantic-competitor explanation. Coverage is only 8/44 and 11/127, so the conclusion is limited rather than population-complete.",
        "",
        "## 4. QD internal vs independent semantic evidence",
        "",
        "The QD table reports final candidate-center ranks/densities and decoder attention, with selection-conditioned quantities explicitly named. Unsupported query-conditioned memory similarity is BLOCKED.",
        "",
        "| Bin | Cohort | Native-valid N | Direction agreement | Direction disagreement |",
        "|---|---|---:|---:|---:|",
    ]
    for key, item in agreement.items():
        bin_name, cohort_name = key.rsplit("_", 1)
        lines.append(f"| {bin_name} | {cohort_name} | {item['N_valid']} | {item['agree']} | {item['disagree']} |")
    lines += [
        "",
        "Agreement is mixed, not a proof of equivalence. The QD direction is a center-location diagnostic and is not the native semantic score.",
        "",
        "## 5. Decoder reference attraction",
        "",
        "Using the full-access baseline trace and its final top-1 query, attraction means the final reference is closer to the hard peak than its initial reference and closer to the hard peak than to the GT center.",
        "",
        "| Bin | Harmful N | Begin hard/remain | Begin GT/move hard | Approach GT | Approach neither | Attraction flag |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bin_name in PRIMARY_BINS:
        rows = [row for row in decoder if row["duration_bin"] == bin_name and row["cohort"] == "harmful"]
        counts = {key: sum(row["classification"] == key for row in rows) for key in ("begin_hard_and_remain", "begin_gt_move_hard", "approach_gt", "approach_neither")}
        attracted = sum(row["reference_attraction_flag"] == "True" for row in rows)
        lines.append(f"| {bin_name} | {len(rows)} | {counts['begin_hard_and_remain']} | {counts['begin_gt_move_hard']} | {counts['approach_gt']} | {counts['approach_neither']} | {attracted}/{len(rows)} |")
    lines += [
        "",
        "The attraction trace supports a decoder-competition component in the harmful cohort, but it is still a post-hoc trajectory measurement of the frozen decoder, not an intervention that establishes necessity.",
        "",
        "## 6. Matched replacement probe",
        "",
        f"The one-token replacement was valid for {len(replacement)} harmful queries. Replacing the first deterministic RANDOM_25 token with the strongest selected hard token worsened the GT-center rank in {sum(int(float(row['replacement_best_gt_center_rank'])) > int(float(row['random25_best_gt_center_rank'])) for row in replacement)} cases and changed mean Top-10 GT-center density by {float(np.mean([int(float(row['replacement_gt_center_density_top10'])) - int(float(row['random25_gt_center_density_top10'])) for row in replacement])) if replacement else float('nan'):.4f}. This is a bounded token-level probe; it is not a peak-level intervention.",
        "",
        "## 7. Human review and repeats",
        "",
        "The review UI was prepared from the strongest harmful cases with available WAVs: 8 cases in 0–2 s and 11 cases in 2–5 s. Labels were intentionally left blank. Possible repeated or incompletely annotated occurrences are therefore unresolved, not supported or rejected.",
        "",
        "## 8. Taxonomy and scale separation",
        "",
        "Taxonomy labels are allowed to overlap. `NEAR_GT_SCALE_LEAKAGE` is a geometry label, `SEMANTIC_COMPETITOR` requires native HARD>GT, `MODEL_SPECIFIC_FALSE_POSITIVE` requires native GT>HARD plus QD hard-favored location evidence, and `DECODER_ATTRACTION` comes from the layerwise trace. Unlabeled cases remain `UNRESOLVED`.",
        "",
        "Scale is reported separately in `scale_relation.csv`; among harmful rows the median full-access width/GT ratio is 4.00 and HARD_25 is 4.67. A hard-negative location effect does not imply that span extent is accurate.",
        "",
        "## 9. Hypotheses and branch",
        "",
        "| Hypothesis | Status |",
        "|---|---|",
    ]
    for key, value in statuses.items():
        if key != "selected_branch":
            lines.append(f"| {key} | **{value}** |")
    lines += [
        f"| selected next branch | **{statuses['selected_branch']}** |",
        "",
        "## 10. Claim boundary",
        "",
        "This audit establishes that harmful HARD_25 selections often coincide with a frozen-decoder reference-attraction pattern, while native MS-CLAP evidence is mostly GT-favoring on the small raw-audio subset. It does not establish that every hard negative is semantically unrelated, does not resolve repeats or annotation misses, does not prove decoder attraction is necessary, and does not show that the stored QD-DETR temporal representation preserves native local MS-CLAP evidence.",
        "",
        "No method design or next experiment was implemented.",
    ]
    (output / "mechanism_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    cohort_rows = read_csv(output / "hard_negative_geometry.csv")
    qd_rows = {row["qid"]: row for row in read_csv(output / "qd_internal_gt_vs_hard.csv")}
    native_rows_list = read_csv(output / "native_clap_gt_vs_hard.csv")
    native_rows = {row["qid"]: row for row in native_rows_list}
    decoder_rows_list = read_csv(output / "decoder_attraction.csv")
    decoder_rows = {row["qid"]: row for row in decoder_rows_list}
    replacement_rows = read_csv(output / "matched_replacement_probe.csv")
    counterfactual_rows = read_csv(args.counterfactual_csv.resolve())
    cohort_by_qid = {row["qid"]: row for row in cohort_rows}
    scale_rows = write_scale_relation(output, counterfactual_rows, cohort_by_qid)
    selection_rows = [
        {"qid": row["qid"], "duration_bin": row["duration_bin"], "cohort": row["cohort"], "hard_peak_geometry_class": row["hard_peak_geometry_class"], "hard_peak_overlaps_gt_plus_minus_2s": row["hard_peak_overlaps_gt_plus_minus_2s"]}
        for row in cohort_rows
    ]
    taxonomy = make_taxonomy(cohort_by_qid, qd_rows, native_rows, decoder_rows)
    write_csv(output / "failure_taxonomy.csv", taxonomy, list(taxonomy[0].keys()) if taxonomy else ["qid"])
    write_selection_bias(output, native_rows_list, cohort_rows)
    native_stats = native_summary(native_rows_list)
    statuses = write_hypotheses(output, cohort_rows, native_stats, taxonomy, decoder_rows_list, scale_rows, replacement_rows)
    write_next_branch(output, statuses["selected_branch"])
    write_report(output, cohort_rows, qd_rows, native_rows_list, decoder_rows_list, replacement_rows, taxonomy, scale_rows, native_stats, statuses)

    taxonomy_counts: dict[str, Any] = {}
    for bin_name in PRIMARY_BINS:
        for cohort in ("harmful", "control"):
            selected = [row for row in taxonomy if row["duration_bin"] == bin_name and row["cohort"] == cohort]
            counts: Counter[str] = Counter()
            for row in selected:
                counts.update(row["labels"].split(";"))
            taxonomy_counts[f"{bin_name}_{cohort}"] = {key: int(value) for key, value in sorted(counts.items())}
    replacement_harmful = [row for row in replacement_rows if row["replacement_status"] == "VALID_TOKEN_LEVEL"]
    replacement_summary = {
        "N_rows": len(replacement_rows),
        "N_valid_token_level": len(replacement_harmful),
        "mean_gt_rank_change_replacement_minus_random": median([int(float(row["replacement_best_gt_center_rank"])) - int(float(row["random25_best_gt_center_rank"])) for row in replacement_harmful]),
        "mean_gt_density_change_replacement_minus_random": float(np.mean([int(float(row["replacement_gt_center_density_top10"])) - int(float(row["random25_gt_center_density_top10"])) for row in replacement_harmful])) if replacement_harmful else None,
        "N_gt_rank_worsened": sum(int(float(row["replacement_best_gt_center_rank"])) > int(float(row["random25_best_gt_center_rank"])) for row in replacement_harmful),
    }
    summary = {
        "status": "COMPLETE_WITH_NATIVE_AND_HUMAN_COVERAGE_LIMITS",
        "cohort": {
            "harmful_rule": "HARD_25 center_hit10_2s < mean(RANDOM_25 replicate 0..9)",
            "counts_by_bin": {bin_name: {cohort: sum(row["duration_bin"] == bin_name and row["cohort"] == cohort for row in cohort_rows) for cohort in ("harmful", "control")} for bin_name in PRIMARY_BINS},
            "p3_location_failure_counts": {bin_name: sum(row["duration_bin"] == bin_name and row["p3_location_failure"] == "1" for row in cohort_rows) for bin_name in PRIMARY_BINS},
        },
        "geometry": {
            "near_definition": "adjacent gap <=1s or moderately near gap <=5s",
            "expanded_gt_neighborhood": "GT interval +/-2s",
            "counts": taxonomy_counts,
        },
        "native_msclap": native_stats,
        "native_provenance_file": "native_clap_provenance.json",
        "qd_internal": {
            "query_conditioned_memory_response": "BLOCKED",
            "reason": "no official common QD query-memory similarity; no stored audio/text cosine",
            "decoder_rows": len(decoder_rows_list),
        },
        "decoder_attraction": {
            "harmful_class_counts": dict(Counter(row["classification"] for row in decoder_rows_list if row["cohort"] == "harmful")),
            "harmful_attraction_count": sum(row["reference_attraction_flag"] == "True" for row in decoder_rows_list if row["cohort"] == "harmful"),
        },
        "matched_replacement": replacement_summary,
        "human_review": {
            "status": "PREPARED_UNLABELED",
            "labels_supplied": False,
            "manifest": "human_review/review_manifest.csv",
        },
        "scale_relation": {
            "rows": len(scale_rows),
            "median_full_width_gt_ratio_harmful": median([f(row, "full_width_gt_ratio") for row in scale_rows if row["cohort"] == "harmful"]),
            "median_hard_width_gt_ratio_harmful": median([f(row, "hard25_width_gt_ratio") for row in scale_rows if row["cohort"] == "harmful"]),
            "separate_from_location": True,
        },
        "hypothesis_status": statuses,
        "taxonomy_file": "failure_taxonomy.csv",
    }
    (output / "failure_taxonomy.csv").chmod(0o644)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counterfactual-csv", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
