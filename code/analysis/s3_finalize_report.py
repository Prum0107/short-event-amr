#!/usr/bin/env python3
"""Finalize the S3 report after manual inspection of generated statistics."""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path("/private/research-artifact")


def main() -> None:
    summary_path = OUT / "clap_temporal_evidence_summary.json"
    report_path = OUT / "clap_temporal_evidence_report.md"
    summary = json.loads(summary_path.read_text())
    summary["decision"] = "MIXED_BOTTLENECK"
    summary["decision_rationale"] = {
        "evidence_label": "exploratory diagnostic evidence",
        "observed": [
            "MS-CLAP frame retrieval collapses strongly for very short GT bins: 0–2s Hit@5 is 12/90 while 20s+ is 263/357.",
            "Among short queries with CLAP Hit@5, both AMR systems still fail Top1 IoU@0.7 frequently, including 111/128 SHARP_CLAP_EVIDENCE cases.",
            "GT-centered normalized-score widths are usually 0–1 seconds; only 7 short Hit@5 queries meet the descriptive diffuse threshold of >=5 seconds.",
        ],
        "supported": "The results support a mixed diagnostic: weak or poorly ranked short-event CLAP evidence is present, and a separate localization failure remains on many short queries even when Top5 evidence reaches GT.",
        "not_supported": [
            "No semantic correctness follows from frame overlap.",
            "The width correlations are descriptive and do not establish that CLAP width causes AMR prediction width.",
            "The audit does not select or test a method.",
        ],
    }
    summary["final_questions"] = {
        "1": "Yes. There is a strong short-duration CLAP frame-retrieval collapse: 0–2s Hit@1/3/5/10 are 6/90 (6.7%), 11/90 (12.2%), 12/90 (13.3%), and 17/90 (18.9%), versus 20s+ values 200/357 (56.0%), 244/357 (68.3%), 263/357 (73.7%), and 297/357 (83.2%).",
        "2": "For 0–2s: Hit@1=6/90 (6.7%), Hit@3=11/90 (12.2%), Hit@5=12/90 (13.3%), Hit@10=17/90 (18.9%).",
        "3": "CLAP Hit@5 with both AMR systems failing Top1 IoU@0.7 occurs in 11/90 (12.2%) of 0–2s queries, or 11/12 (91.7%) of the CLAP-Hit@5 subset; for 2–5s it is 120/376 (31.9%), or 120/140 (85.7%) of CLAP hits.",
        "4": "Very-short evidence is usually sharp or weak rather than diffuse under the defined threshold-width measure: GT-centered width80 has median 0s and p75 0s for 0–2s, and median 0s and p75 1s for 2–5s. Only 7/466 short queries are diffuse (width80 >=5s), including 7/152 short CLAP-Hit@5 queries.",
        "5": "Within SHARP_CLAP_EVIDENCE (N=128), both models fail Top1 IoU@0.7 in 111/128 (86.7%); both fail Oracle10@0.7 in 89/128 (69.5%).",
        "6": "DIFFUSE_CLAP_EVIDENCE is small (N=7). QD-DETR Best10 duration-ratio median is 3.33 and UVCOM is 1.67; both Top1 models fail in 6/7 (85.7%). This is descriptive and low-powered.",
        "7": "No positive width association is observed among short positive-overlap cases: QD rho=-0.099 for prediction duration and -0.149 for duration ratio; UVCOM rho=-0.069 and -0.145. Quartile patterns are descriptive only.",
        "8": "The evidence is more consistent with MIXED_BOTTLENECK: short-event CLAP evidence/ranking is weak, but substantial Top1 AMR failure remains even when Top5 CLAP evidence reaches GT.",
        "9": "The single next diagnostic would be a fixed, untrained CLAP-evidence interval control: convert the top-ranked CLAP frame evidence into a minimal temporal interval and compare its IoU@0.7 with AMR Top1/Best10 on the same short queries. This was not implemented here.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    report = report_path.read_text()
    report = report.replace("Decision: `PENDING_MANUAL_INTERPRETATION`", "Decision: `MIXED_BOTTLENECK`")
    table_start = report.index("| GT bin | N | CLAP H@1")
    table_end = report.index("## GT-centered concentration by bin", table_start)
    table_lines = [
        "| GT bin | N | CLAP H@1 | H@3 | H@5 | H@10 | peak∩GT | GT rank p25/med/p75 | median normalized rank | QD R1@.7 | QD Oracle10@.7 | UV R1@.7 | UV Oracle10@.7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["0–2s", "2–5s", "5–10s", "10–20s", "20s+"]:
        b = summary["view_a"][label]
        mr = b["model_reference"]
        rank = b["best_gt_frame_rank"]
        norm = b["normalized_best_gt_frame_rank"]
        table_lines.append(
            f"| {label} | {b['N']} | {b['clap']['Hit@1']['fraction']:.4f} | {b['clap']['Hit@3']['fraction']:.4f} | {b['clap']['Hit@5']['fraction']:.4f} | {b['clap']['Hit@10']['fraction']:.4f} | {b['global_peak_overlaps_gt']['fraction']:.4f} | {rank['p25']:.2f}/{rank['median']:.2f}/{rank['p75']:.2f} (valid {rank['n']}/{rank['denominator']}) | {norm['median']:.4f} (valid {norm['n']}/{norm['denominator']}) | {mr['QD-DETR']['R1@0.7']['fraction']:.4f} | {mr['QD-DETR']['Oracle10@0.7']['fraction']:.4f} | {mr['UVCOM']['R1@0.7']['fraction']:.4f} | {mr['UVCOM']['Oracle10@0.7']['fraction']:.4f} |"
        )
    report = report[:table_start] + "\n".join(table_lines) + "\n\n" + report[table_end:]
    report = report.replace(
        "- `audio_duration` is the released annotation duration; it is not re-read from raw audio.",
        "- `audio_duration` is the released annotation duration; it is not re-read from raw audio. Two queries have no positive intersection with any cached complete frame because their GT lies in the unrecovered final partial region; their Hit@K is false and GT-frame rank/GT-centered width are null, with valid denominators shown above.",
    )
    quartile_start = report.index("## Sharp versus diffuse evidence")
    quartile_lines = ["## Short positive-overlap width quartiles", "", "Short queries were split into deterministic equal-count groups by GT-centered width80; model medians use positive-overlap Best10 cases.", "", "| Quartile | N | width range (s) | QD positive N | QD median duration (s) | QD median ratio | UV positive N | UV median duration (s) | UV median ratio |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for q in summary["quartiles"]["quartiles"]:
        quartile_lines.append(f"| {q['quartile']} | {q['N']} | {q['width_range_sec'][0]:.2f}–{q['width_range_sec'][1]:.2f} | {q['QD_best10_positive_overlap_N']} | {q['QD_median_prediction_duration_sec']:.2f} | {q['QD_median_duration_ratio']:.3f} | {q['UVCOM_best10_positive_overlap_N']} | {q['UVCOM_median_prediction_duration_sec']:.2f} | {q['UVCOM_median_duration_ratio']:.3f} |")
    report = report[:quartile_start] + "\n".join(quartile_lines) + "\n\n" + report[quartile_start:]
    start = report.index("## Final questions")
    end = report.index("## Outputs", start)
    q = summary["final_questions"]
    lines = ["## Final questions", ""]
    for idx in range(1, 10):
        lines.append(f"{idx}. {q[str(idx)]}")
        lines.append("")
    interpretation = summary["decision_rationale"]
    lines.extend([
        "## Decision rationale",
        "",
        "Evidence label: `exploratory diagnostic evidence`.",
        "",
        f"Supported: {interpretation['supported']}",
        "",
        "Not supported:",
        "",
    ])
    for item in interpretation["not_supported"]:
        lines.append(f"- {item}")
    lines.append("")
    report = report[:start] + "\n".join(lines) + "\n" + report[end:]
    report_path.write_text(report)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "report": str(report_path)}))


if __name__ == "__main__":
    main()
