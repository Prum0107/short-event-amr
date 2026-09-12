#!/usr/bin/env python3
"""Recompute learning-dynamics interpretation reports from completed CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import baseline_scale_learning_dynamics_audit as audit

    width = read_rows(output / "gt_pred_width_response_over_time.csv")
    numeric = read_rows(output / "width_head_numerics_over_time.csv")
    gradient = read_rows(output / "gradient_history_by_duration.csv")
    matched = read_rows(output / "hungarian_assignment_dynamics.csv")
    specialization = read_rows(output / "slot_specialization_over_time.csv")
    audio = read_rows(output / "audio_duration_effect_over_time.csv")
    for row in width:
        row["N_centered_proposals"] = int(float(row["N_centered_proposals"]))
        for key in ("median_pred_gt_ratio", "slope"):
            row[key] = None if row[key] in ("", "None") else float(row[key])
    for row in numeric:
        row["N"] = int(float(row["N"]))
        row["saturation_fraction_distance_lt_0.05"] = None if row["saturation_fraction_distance_lt_0.05"] in ("", "None") else float(row["saturation_fraction_distance_lt_0.05"])
    for row in gradient:
        row["epoch"] = int(float(row["epoch"]))
        row["matched_targets"] = int(float(row["matched_targets"]))
        row["span_embed_width_grad_norm_median"] = float(row["span_embed_width_grad_norm_median"])
    for row in matched:
        row["epoch"] = int(float(row["epoch"]))
        row["assigned_query_slot"] = int(float(row["assigned_query_slot"]))
        row["column_cost_margin"] = None if row["column_cost_margin"] in ("", "None") else float(row["column_cost_margin"])
    for row in specialization:
        row["epoch"] = int(float(row["epoch"]))
        row["assignment_count"] = int(float(row["assignment_count"]))
        row["query_slot"] = int(float(row["query_slot"]))
        row["bin_assignment_fraction"] = None if row["bin_assignment_fraction"] in ("", "None") else float(row["bin_assignment_fraction"])
    for row in audio:
        row["epoch"] = int(float(row["epoch"])) if row["epoch"] else None
        row["adjusted_N"] = int(float(row["adjusted_N"]))
        row["adjusted_log_audio_duration_coefficient"] = None if row["adjusted_log_audio_duration_coefficient"] in ("", "None") else float(row["adjusted_log_audio_duration_coefficient"])

    events = audit.temporal_events(width, numeric, gradient, matched, specialization, audio)
    hypotheses, candidate = audit.assess_hypotheses(events)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["temporal_events"] = events
    summary["hypotheses"] = hypotheses
    summary["causal_priority_candidate"] = candidate
    summary["method_design_gate"] = "NO"
    summary["causal_ambiguity_remaining"] = "The run is observational. Same-epoch onset of initial scale error, matching ambiguity, and slot effects does not establish precedence; missing short-vs-long gradient separation and the absence of interventions leave output-coordinate, matching, slot-prior, audio-duration, and optimization mechanisms coupled."
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Temporal ordering", "",
        "This is an observational ordering audit. Strict precedence requires an earlier checkpoint; events in the same checkpoint are treated as inconclusive for precedence.", "",
        "| event | first_epoch |", "|---|---:|",
    ]
    lines.extend(f"| {key} | {'' if value is None else value} |" for key, value in events.items())
    lines.extend(["", "No event ordering here is a causal intervention."])
    (output / "temporal_ordering.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["# Hypothesis assessment", "", "| hypothesis | description | status | supporting_event |", "|---|---|---|---:|"]
    lines.extend(f"| {row['hypothesis']} | {row['description']} | {row['status']} | {'' if row['supporting_event'] is None else row['supporting_event']} |" for row in hypotheses)
    lines.extend(["", "SUPPORTED means only that the predeclared temporal criterion was met; it does not prove causation."])
    (output / "hypothesis_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "causal_priority.md").write_text(f"# Causal priority\n\nCandidate: **{candidate}**\n\nThe priority is based on co-occurring early events and is not a causal conclusion.\n", encoding="utf-8")
    (output / "method_design_gate.md").write_text("# Method-design gate\n\n**NO.** This observational audit does not authorize a new method. The remaining mechanisms are not causally separated.\n", encoding="utf-8")
    print(json.dumps({"events": events, "hypotheses": hypotheses, "candidate": candidate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
