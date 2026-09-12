import argparse
import json
import os
from collections import Counter, defaultdict


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def quality_key(row, prefix=""):
    value = float(row.get(f"{prefix}top1_iou" if prefix else "top1_iou", 0.0))
    return (value >= 0.7, value >= 0.5, value)


def outcome(row):
    ms_key = (float(row.get("ms_top1_iou", 0.0)) >= 0.7, float(row.get("ms_top1_iou", 0.0)) >= 0.5, float(row.get("ms_top1_iou", 0.0)))
    new_key = quality_key(row)
    if new_key > ms_key:
        return "improved"
    if new_key < ms_key:
        return "regressed"
    return "same"


def evidence_role(row):
    ms_cat = row.get("ms_category", "")
    new_cat = row.get("category", "")
    result = outcome(row)
    delta = float(row.get("top1_iou", 0.0)) - float(row.get("ms_top1_iou", 0.0))

    if ms_cat == "semantic_miss" and new_cat != "semantic_miss":
        return "semantic_recovery"
    if ms_cat == "good" and new_cat != "good":
        return "anchor_regression"
    if ms_cat == "good" and new_cat == "good":
        if result == "improved":
            return "anchor_refinement"
        if result == "regressed":
            return "anchor_weakening"
        return "anchor_preserved"
    if new_cat == "semantic_miss" and ms_cat != "semantic_miss":
        return "semantic_regression"
    if new_cat == "good" and ms_cat != "good":
        return "temporal_correction_to_good"
    if ms_cat in {"boundary_error", "candidate_exists", "evidence_good_decode_bad"} and result == "improved":
        return "temporal_refinement"
    if ms_cat in {"boundary_error", "candidate_exists", "evidence_good_decode_bad"} and result == "regressed":
        return "temporal_regression"
    if abs(delta) >= 0.1:
        return "iou_shift"
    return "neutral"


def compact_case(row):
    top = row.get("top_rows", [{}])[0] if row.get("top_rows") else {}
    return {
        "qid": row.get("qid"),
        "query": row.get("query", ""),
        "vid": row.get("vid", ""),
        "role": evidence_role(row),
        "outcome": outcome(row),
        "ms_category": row.get("ms_category", ""),
        "new_category": row.get("category", ""),
        "chosen_representation": row.get("chosen_representation", ""),
        "candidate_source": row.get("candidate_source", ""),
        "ms_top1_iou": float(row.get("ms_top1_iou", 0.0)),
        "new_top1_iou": float(row.get("top1_iou", 0.0)),
        "delta_top1_iou": float(row.get("top1_iou", 0.0)) - float(row.get("ms_top1_iou", 0.0)),
        "pred_quality": float(top.get("pred_quality", 0.0)),
        "target_raw_iou": float(top.get("target_raw_iou", 0.0)),
        "gt_windows": row.get("gt_windows", []),
        "windows": row.get("windows", [])[:3],
    }


def summarize(rows, key_fn):
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    output = {}
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        outcomes = Counter(outcome(row) for row in group)
        roles = Counter(evidence_role(row) for row in group)
        output[str(key)] = {
            "count": len(group),
            "improved": outcomes.get("improved", 0),
            "regressed": outcomes.get("regressed", 0),
            "same": outcomes.get("same", 0),
            "avg_delta_top1_iou": sum(float(row.get("top1_iou", 0.0)) - float(row.get("ms_top1_iou", 0.0)) for row in group)
            / max(len(group), 1),
            "roles": dict(roles),
        }
    return output


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt_float(value, digits=4):
    return f"{float(value):.{digits}f}"


def analyze(rows):
    changed = [row for row in rows if row.get("changed_from_ms")]
    roles = Counter(evidence_role(row) for row in changed)
    outcomes = Counter(outcome(row) for row in changed)

    role_summary = {}
    for role in sorted(roles):
        group = [row for row in changed if evidence_role(row) == role]
        role_summary[role] = {
            "count": len(group),
            "improved": sum(1 for row in group if outcome(row) == "improved"),
            "regressed": sum(1 for row in group if outcome(row) == "regressed"),
            "same": sum(1 for row in group if outcome(row) == "same"),
            "avg_delta_top1_iou": sum(float(row.get("top1_iou", 0.0)) - float(row.get("ms_top1_iou", 0.0)) for row in group)
            / max(len(group), 1),
            "adapter_count": sum(1 for row in group if row.get("chosen_representation") == "adapter"),
            "ms_count": sum(1 for row in group if row.get("chosen_representation") == "ms_clap"),
        }

    semantic_rows = [row for row in changed if evidence_role(row) == "semantic_recovery"]
    temporal_rows = [
        row
        for row in changed
        if evidence_role(row) in {"temporal_correction_to_good", "temporal_refinement"}
    ]
    risk_rows = [
        row
        for row in changed
        if evidence_role(row) in {"anchor_regression", "semantic_regression", "temporal_regression"}
    ]

    return {
        "overall": {
            "total": len(rows),
            "changed": len(changed),
            "improved": outcomes.get("improved", 0),
            "regressed": outcomes.get("regressed", 0),
            "same": outcomes.get("same", 0),
            "semantic_recovery": len(semantic_rows),
            "temporal_positive": len(temporal_rows),
            "risk_negative": len(risk_rows),
            "intervention_precision": outcomes.get("improved", 0) / max(len(changed), 1),
        },
        "role_summary": role_summary,
        "by_role_and_source": summarize(changed, lambda row: f"{evidence_role(row)}::{row.get('candidate_source', '')}"),
        "by_role_and_representation": summarize(changed, lambda row: f"{evidence_role(row)}::{row.get('chosen_representation', '')}"),
        "by_transition": summarize(changed, lambda row: f"{row.get('ms_category', '')}->{row.get('category', '')}"),
        "examples": {
            "semantic_recovery": [
                compact_case(row)
                for row in sorted(semantic_rows, key=lambda item: float(item.get("top1_iou", 0.0)) - float(item.get("ms_top1_iou", 0.0)), reverse=True)[:12]
            ],
            "temporal_positive": [
                compact_case(row)
                for row in sorted(temporal_rows, key=lambda item: float(item.get("top1_iou", 0.0)) - float(item.get("ms_top1_iou", 0.0)), reverse=True)[:12]
            ],
            "risk_negative": [
                compact_case(row)
                for row in sorted(risk_rows, key=lambda item: float(item.get("top1_iou", 0.0)) - float(item.get("ms_top1_iou", 0.0)))[:12]
            ],
        },
    }


def render_markdown(analysis):
    role_rows = []
    for role, item in sorted(analysis["role_summary"].items(), key=lambda pair: (-pair[1]["count"], pair[0])):
        role_rows.append(
            [
                role,
                item["count"],
                item["improved"],
                item["regressed"],
                item["same"],
                fmt_float(item["avg_delta_top1_iou"]),
                item["adapter_count"],
                item["ms_count"],
            ]
        )

    transition_rows = []
    for transition, item in sorted(analysis["by_transition"].items(), key=lambda pair: (-pair[1]["count"], pair[0]))[:16]:
        transition_rows.append(
            [
                transition,
                item["count"],
                item["improved"],
                item["regressed"],
                item["same"],
                fmt_float(item["avg_delta_top1_iou"]),
            ]
        )

    source_rows = []
    for key, item in sorted(analysis["by_role_and_source"].items(), key=lambda pair: (-abs(pair[1]["avg_delta_top1_iou"]) * pair[1]["count"], pair[0]))[:18]:
        source_rows.append(
            [
                key,
                item["count"],
                item["improved"],
                item["regressed"],
                fmt_float(item["avg_delta_top1_iou"]),
            ]
        )

    text = [
        "# Semantic-Temporal Evidence Analysis",
        "",
        "This report decomposes candidate-level fusion interventions into semantic and temporal roles.",
        "",
        "## Overall",
        "",
        "```json",
        json.dumps(analysis["overall"], indent=2),
        "```",
        "",
        "## Role Summary",
        "",
        table(
            ["Role", "Count", "Improved", "Regressed", "Same", "Avg Delta", "Adapter", "MS"],
            role_rows,
        ),
        "",
        "## Category Transitions",
        "",
        table(["Transition", "Count", "Improved", "Regressed", "Same", "Avg Delta"], transition_rows),
        "",
        "## Role And Candidate Source",
        "",
        table(["Role::Source", "Count", "Improved", "Regressed", "Avg Delta"], source_rows),
        "",
        "## Interpretation",
        "",
        "- `semantic_recovery` measures cases where fusion moves an MS semantic miss into a temporally plausible candidate.",
        "- `temporal_correction_to_good` and `temporal_refinement` measure boundary/candidate improvements once semantic evidence is already present.",
        "- `anchor_regression` is the main failure mode for a robust selector: the fused candidate overrides an already-good MS anchor.",
        "- `semantic_regression` and `temporal_regression` identify cases where the intervention loses temporal grounding.",
    ]
    return "\n".join(text) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_rows_path", required=True)
    parser.add_argument("--output_json_path", required=True)
    parser.add_argument("--output_md_path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_json(args.case_rows_path)
    analysis = analyze(rows)
    save_json(analysis, args.output_json_path)
    save_text(render_markdown(analysis), args.output_md_path)
    print(json.dumps(analysis["overall"], indent=2))
    print("saved", args.output_json_path)
    print("saved", args.output_md_path)


if __name__ == "__main__":
    main()
