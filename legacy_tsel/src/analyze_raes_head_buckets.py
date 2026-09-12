import argparse
import json
import os
from collections import Counter

from analyze_semantic_temporal_evidence import evidence_role, outcome


HEADS = [
    ("semantic_gain", "pred_semantic_gain"),
    ("temporal_gain", "pred_temporal_gain"),
    ("anchor_reliability", "pred_anchor_reliability"),
    ("anchor_risk", "pred_anchor_risk"),
    ("semantic_risk", "pred_semantic_risk"),
    ("temporal_risk", "pred_temporal_risk"),
]


BUCKETS = [
    ("[0.00,0.25)", 0.00, 0.25),
    ("[0.25,0.50)", 0.25, 0.50),
    ("[0.50,0.75)", 0.50, 0.75),
    ("[0.75,1.00]", 0.75, 1.01),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def top_row(row):
    rows = row.get("top_rows") or []
    return rows[0] if rows else {}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def in_bucket(value, low, high):
    return low <= float(value) < high


def summarize_bucket(rows):
    roles = Counter(evidence_role(row) for row in rows)
    outcomes = Counter(outcome(row) for row in rows)
    deltas = [float(row.get("top1_iou", 0.0)) - float(row.get("ms_top1_iou", 0.0)) for row in rows]
    return {
        "count": len(rows),
        "mean_iou": mean([float(row.get("top1_iou", 0.0)) for row in rows]),
        "mean_delta": mean(deltas),
        "strict_hit_rate": mean([1.0 if float(row.get("top1_iou", 0.0)) >= 0.7 else 0.0 for row in rows]),
        "improved_rate": outcomes.get("improved", 0) / max(len(rows), 1),
        "regression_rate": outcomes.get("regressed", 0) / max(len(rows), 1),
        "semantic_recovery_rate": roles.get("semantic_recovery", 0) / max(len(rows), 1),
        "temporal_positive_rate": (
            roles.get("temporal_correction_to_good", 0) + roles.get("temporal_refinement", 0)
        )
        / max(len(rows), 1),
        "anchor_failure_rate": (
            roles.get("anchor_regression", 0) + roles.get("anchor_weakening", 0)
        )
        / max(len(rows), 1),
    }


def summarize(rows):
    changed = [row for row in rows if row.get("changed_from_ms")]
    report = {
        "source_rows": len(rows),
        "changed_rows": len(changed),
        "heads": {},
    }
    for short_name, key in HEADS:
        rows_with_head = [row for row in changed if key in top_row(row)]
        bucket_rows = {}
        for label, low, high in BUCKETS:
            selected = [row for row in rows_with_head if in_bucket(top_row(row).get(key, 0.0), low, high)]
            bucket_rows[label] = summarize_bucket(selected)
        report["heads"][short_name] = {
            "key": key,
            "buckets": bucket_rows,
        }
    return report


def fmt_pct(value):
    return f"{100.0 * float(value):.1f}%"


def fmt_num(value):
    return f"{float(value):.3f}"


def markdown_table(head_name, head_report):
    lines = [
        f"### {head_name}",
        "",
        "| Bucket | Count | Mean IoU | Delta | Strict hit | Improved | Regressed | Semantic recovery | Temporal positive | Anchor failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in head_report["buckets"].items():
        lines.append(
            "| {bucket} | {count} | {mean_iou} | {delta} | {strict} | {improved} | {regressed} | {semantic} | {temporal} | {anchor} |".format(
                bucket=label,
                count=metrics["count"],
                mean_iou=fmt_num(metrics["mean_iou"]),
                delta=fmt_num(metrics["mean_delta"]),
                strict=fmt_pct(metrics["strict_hit_rate"]),
                improved=fmt_pct(metrics["improved_rate"]),
                regressed=fmt_pct(metrics["regression_rate"]),
                semantic=fmt_pct(metrics["semantic_recovery_rate"]),
                temporal=fmt_pct(metrics["temporal_positive_rate"]),
                anchor=fmt_pct(metrics["anchor_failure_rate"]),
            )
        )
    return "\n".join(lines)


def render_markdown(report, rows_path):
    semantic_hi = report["heads"]["semantic_gain"]["buckets"]["[0.75,1.00]"]
    semantic_lo = report["heads"]["semantic_gain"]["buckets"]["[0.00,0.25)"]
    temporal_hi = report["heads"]["temporal_gain"]["buckets"]["[0.75,1.00]"]
    temporal_lo = report["heads"]["temporal_gain"]["buckets"]["[0.00,0.25)"]
    anchor_hi = report["heads"]["anchor_reliability"]["buckets"]["[0.75,1.00]"]
    anchor_lo = report["heads"]["anchor_reliability"]["buckets"]["[0.00,0.25)"]
    lines = [
        "# RAES Head Bucket Analysis",
        "",
        "This lightweight analysis groups changed TSEL-RAES decisions by evidence-head score buckets.",
        "It is meant for paper analysis and sanity checking, not for model selection.",
        "",
        f"- Source rows: `{rows_path}`",
        f"- Total rows: {report['source_rows']}",
        f"- Changed-from-MS rows: {report['changed_rows']}",
        "",
        "## Reading Guide",
        "",
        "- `Delta` is mean `TSEL-RAES top1_iou - MS top1_iou`.",
        "- `Strict hit` is the fraction with top-1 IoU >= 0.7.",
        "- `Semantic recovery` indicates MS semantic miss recovered by the selected candidate.",
        "- `Temporal positive` groups temporal correction/refinement roles.",
        "- `Anchor failure` groups anchor regression and anchor weakening.",
        "",
        "## Current Seed Observations",
        "",
        "- High `semantic_gain` is strongly aligned with semantic recovery: "
        f"{fmt_pct(semantic_hi['semantic_recovery_rate'])} in the high bucket vs "
        f"{fmt_pct(semantic_lo['semantic_recovery_rate'])} in the low bucket. "
        f"It also has lower regression rate ({fmt_pct(semantic_hi['regression_rate'])} vs "
        f"{fmt_pct(semantic_lo['regression_rate'])}).",
        "- High `temporal_gain` is aligned with temporal-positive behavior: "
        f"{fmt_pct(temporal_hi['temporal_positive_rate'])} in the high bucket vs "
        f"{fmt_pct(temporal_lo['temporal_positive_rate'])} in the low bucket. "
        f"However, the high bucket still contains regressions ({fmt_pct(temporal_hi['regression_rate'])}), "
        "which supports the limitation that temporal evidence can be over-trusted.",
        "- High `anchor_reliability` corresponds to higher mean IoU and strict-hit rate "
        f"({fmt_num(anchor_hi['mean_iou'])} / {fmt_pct(anchor_hi['strict_hit_rate'])}) than the low bucket "
        f"({fmt_num(anchor_lo['mean_iou'])} / {fmt_pct(anchor_lo['strict_hit_rate'])}). "
        "For anchor protection itself, use the pairwise ECF-vs-RAES analysis in `main_tables.md`; "
        "bucket analysis alone is not enough to claim calibrated anchor protection.",
        "",
        "## Bucket Tables",
        "",
    ]
    for head_name, head_report in report["heads"].items():
        lines.append(markdown_table(head_name, head_report))
        lines.append("")
    lines.extend(
        [
            "## Paper Interpretation",
            "",
            "The bucket analysis should not be over-claimed as causal proof. Its role is to show whether RAES heads are behaviorally aligned with the case taxonomy.",
            "",
            "Useful patterns to cite if stable:",
            "",
            "- Higher `semantic_gain` should concentrate semantic recovery behavior.",
            "- Higher `temporal_gain` should concentrate temporal-positive decisions, but may also expose over-trusted temporal evidence.",
            "- Higher `anchor_reliability` is useful supporting evidence, but pairwise anchor-protection analysis is the stronger claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows_path",
        default="results/test_semantic_temporal_candidate_scorer_v2_quality_guard_seed2027_cases/seed2027/case_rows.json",
    )
    parser.add_argument("--output_md", default="docs/paper_assets/raes_head_bucket_analysis.md")
    parser.add_argument("--output_json", default="docs/paper_assets/raes_head_bucket_analysis.json")
    args = parser.parse_args()

    rows = load_json(args.rows_path)
    report = summarize(rows)
    save_json(report, args.output_json)
    save_text(render_markdown(report, args.rows_path), args.output_md)
    print(f"saved {args.output_md}")
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
