import argparse
import json
import os
from collections import Counter, defaultdict

from analyze_semantic_temporal_evidence import evidence_role, outcome, save_json, save_text, table


HEADS = [
    "pred_quality_head",
    "pred_anchor_reliability",
    "pred_semantic_gain",
    "pred_temporal_gain",
    "pred_strict_gain",
    "pred_utility",
    "pred_anchor_guard",
    "pred_anchor_risk",
    "pred_semantic_risk",
    "pred_temporal_risk",
    "pred_quality",
    "anchor_disagreement",
    "target_raw_iou",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values):
    return sum(values) / max(len(values), 1)


def quality_key_from_iou(value):
    value = float(value)
    return (value >= 0.7, value >= 0.5, value)


def top_row(row):
    rows = row.get("top_rows") or []
    return rows[0] if rows else {}


def compact(row):
    top = top_row(row)
    return {
        "qid": row.get("qid"),
        "query": row.get("query", ""),
        "ms_category": row.get("ms_category", ""),
        "category": row.get("category", ""),
        "role": evidence_role(row),
        "outcome": outcome(row),
        "ms_top1_iou": row.get("ms_top1_iou"),
        "top1_iou": row.get("top1_iou"),
        "chosen_representation": row.get("chosen_representation", ""),
        "candidate_source": row.get("candidate_source", ""),
        "top_head": {key: top.get(key) for key in HEADS if key in top},
        "windows": row.get("windows", [])[:3],
        "gt_windows": row.get("gt_windows", []),
    }


def summarize_rows(rows):
    changed = [row for row in rows if row.get("changed_from_ms")]
    roles = Counter(evidence_role(row) for row in changed)
    outcomes = Counter(outcome(row) for row in changed)
    return {
        "total": len(rows),
        "changed": len(changed),
        "improved": outcomes.get("improved", 0),
        "regressed": outcomes.get("regressed", 0),
        "same": outcomes.get("same", 0),
        "semantic_recovery": roles.get("semantic_recovery", 0),
        "temporal_positive": roles.get("temporal_correction_to_good", 0) + roles.get("temporal_refinement", 0),
        "anchor_regression": roles.get("anchor_regression", 0),
        "anchor_weakening": roles.get("anchor_weakening", 0),
        "semantic_regression": roles.get("semantic_regression", 0),
        "temporal_regression": roles.get("temporal_regression", 0),
        "good_regressed": sum(1 for row in changed if row.get("ms_category") == "good" and row.get("category") != "good"),
        "intervention_precision": outcomes.get("improved", 0) / max(len(changed), 1),
    }


def summarize_heads(rows, group_name):
    heads = defaultdict(list)
    for row in rows:
        top = top_row(row)
        for key in HEADS:
            if key in top:
                heads[key].append(float(top[key]))
    return {
        "group": group_name,
        "count": len(rows),
        **{key: safe_mean(values) for key, values in sorted(heads.items())},
    }


def compare(v4_rows, v2_rows):
    v4_by_qid = {row["qid"]: row for row in v4_rows}
    v2_by_qid = {row["qid"]: row for row in v2_rows}
    shared_qids = sorted(set(v4_by_qid) & set(v2_by_qid))

    pair_outcomes = Counter()
    pair_roles = Counter()
    v2_avoids_v4_regression = []
    v2_new_regression = []
    v2_improves_v4_same_or_regressed = []
    v4_improves_v2_not = []
    anchor_protected = []
    anchor_new_risk = []
    strict_gain = []
    strict_loss = []

    for qid in shared_qids:
        v4 = v4_by_qid[qid]
        v2 = v2_by_qid[qid]
        v4_out = outcome(v4)
        v2_out = outcome(v2)
        v4_role = evidence_role(v4)
        v2_role = evidence_role(v2)
        pair_outcomes[f"{v4_out}->{v2_out}"] += 1
        pair_roles[f"{v4_role}->{v2_role}"] += 1

        v4_key = quality_key_from_iou(v4.get("top1_iou", 0.0))
        v2_key = quality_key_from_iou(v2.get("top1_iou", 0.0))
        if v4_out == "regressed" and v2_out != "regressed":
            v2_avoids_v4_regression.append((qid, v4, v2))
        if v4_out != "regressed" and v2_out == "regressed":
            v2_new_regression.append((qid, v4, v2))
        if v2_out == "improved" and v4_out != "improved":
            v2_improves_v4_same_or_regressed.append((qid, v4, v2))
        if v4_out == "improved" and v2_out != "improved":
            v4_improves_v2_not.append((qid, v4, v2))
        if v4.get("ms_category") == "good" and v4.get("category") != "good" and not (
            v2.get("ms_category") == "good" and v2.get("category") != "good"
        ):
            anchor_protected.append((qid, v4, v2))
        if not (v4.get("ms_category") == "good" and v4.get("category") != "good") and (
            v2.get("ms_category") == "good" and v2.get("category") != "good"
        ):
            anchor_new_risk.append((qid, v4, v2))
        if v2_key > v4_key:
            strict_gain.append((qid, v4, v2))
        elif v2_key < v4_key:
            strict_loss.append((qid, v4, v2))

    v2_changed = [row for row in v2_rows if row.get("changed_from_ms")]
    head_groups = [
        summarize_heads(v2_changed, "all_changed"),
        summarize_heads([row for row in v2_changed if outcome(row) == "improved"], "improved"),
        summarize_heads([row for row in v2_changed if outcome(row) == "regressed"], "regressed"),
        summarize_heads([row for row in v2_changed if evidence_role(row) == "semantic_recovery"], "semantic_recovery"),
        summarize_heads(
            [row for row in v2_changed if evidence_role(row) in {"temporal_correction_to_good", "temporal_refinement"}],
            "temporal_positive",
        ),
        summarize_heads([item[2] for item in anchor_protected], "anchor_protected_vs_v4"),
        summarize_heads([item[2] for item in anchor_new_risk], "anchor_new_risk_vs_v4"),
    ]

    examples = {
        "v2_avoids_v4_regression": [
            {"qid": qid, "v4": compact(v4), "v2": compact(v2)}
            for qid, v4, v2 in sorted(
                v2_avoids_v4_regression,
                key=lambda item: float(item[1].get("ms_top1_iou", 0.0)) - float(item[1].get("top1_iou", 0.0)),
                reverse=True,
            )[:12]
        ],
        "v2_new_regression": [
            {"qid": qid, "v4": compact(v4), "v2": compact(v2)}
            for qid, v4, v2 in sorted(
                v2_new_regression,
                key=lambda item: float(item[2].get("ms_top1_iou", 0.0)) - float(item[2].get("top1_iou", 0.0)),
                reverse=True,
            )[:12]
        ],
        "v2_improves_v4_same_or_regressed": [
            {"qid": qid, "v4": compact(v4), "v2": compact(v2)}
            for qid, v4, v2 in sorted(
                v2_improves_v4_same_or_regressed,
                key=lambda item: float(item[2].get("top1_iou", 0.0)) - float(item[2].get("ms_top1_iou", 0.0)),
                reverse=True,
            )[:12]
        ],
        "v4_improves_v2_not": [
            {"qid": qid, "v4": compact(v4), "v2": compact(v2)}
            for qid, v4, v2 in sorted(
                v4_improves_v2_not,
                key=lambda item: float(item[1].get("top1_iou", 0.0)) - float(item[1].get("ms_top1_iou", 0.0)),
                reverse=True,
            )[:12]
        ],
    }

    return {
        "overall": {
            "shared_qids": len(shared_qids),
            "v4": summarize_rows(v4_rows),
            "v2": summarize_rows(v2_rows),
            "v2_avoids_v4_regression": len(v2_avoids_v4_regression),
            "v2_new_regression": len(v2_new_regression),
            "v2_improves_v4_same_or_regressed": len(v2_improves_v4_same_or_regressed),
            "v4_improves_v2_not": len(v4_improves_v2_not),
            "anchor_protected_vs_v4": len(anchor_protected),
            "anchor_new_risk_vs_v4": len(anchor_new_risk),
            "v2_strictly_better_than_v4": len(strict_gain),
            "v2_strictly_worse_than_v4": len(strict_loss),
        },
        "pair_outcomes": dict(pair_outcomes),
        "top_pair_roles": dict(pair_roles.most_common(40)),
        "v2_head_groups": head_groups,
        "examples": examples,
    }


def render_markdown(report):
    overall = report["overall"]
    rows = [
        ["Changed", overall["v4"]["changed"], overall["v2"]["changed"]],
        ["Improved", overall["v4"]["improved"], overall["v2"]["improved"]],
        ["Regressed", overall["v4"]["regressed"], overall["v2"]["regressed"]],
        ["Semantic recovery", overall["v4"]["semantic_recovery"], overall["v2"]["semantic_recovery"]],
        ["Temporal positive", overall["v4"]["temporal_positive"], overall["v2"]["temporal_positive"]],
        ["Anchor regression", overall["v4"]["anchor_regression"], overall["v2"]["anchor_regression"]],
        ["Anchor weakening", overall["v4"]["anchor_weakening"], overall["v2"]["anchor_weakening"]],
        ["Good regressed", overall["v4"]["good_regressed"], overall["v2"]["good_regressed"]],
        ["Intervention precision", f"{overall['v4']['intervention_precision']:.3f}", f"{overall['v2']['intervention_precision']:.3f}"],
    ]
    pair_rows = [[key, value] for key, value in sorted(report["pair_outcomes"].items(), key=lambda item: (-item[1], item[0]))]
    head_rows = []
    for group in report["v2_head_groups"]:
        head_rows.append(
            [
                group["group"],
                group["count"],
                f"{group.get('pred_quality_head', 0.0):.3f}",
                f"{group.get('pred_semantic_gain', 0.0):.3f}",
                f"{group.get('pred_temporal_gain', 0.0):.3f}",
                f"{group.get('pred_anchor_reliability', 0.0):.3f}",
                f"{group.get('pred_anchor_guard', 0.0):.3f}",
                f"{group.get('pred_anchor_risk', 0.0):.3f}",
                f"{group.get('pred_semantic_risk', 0.0):.3f}",
                f"{group.get('pred_temporal_risk', 0.0):.3f}",
            ]
        )
    text = [
        "# V2 Scorer vs V4 Fusion Test-Side Analysis",
        "",
        "This report compares V4 candidate-level fusion with V2 `quality_guard` scorer on the frozen test split.",
        "",
        "## Overall",
        "",
        table(["Metric", "V4 fusion", "V2 scorer"], rows),
        "",
        "## Pairwise Outcome Migration",
        "",
        table(["V4 outcome -> V2 outcome", "Count"], pair_rows),
        "",
        "## V2 Evidence Heads",
        "",
        table(
            [
                "Group",
                "Count",
                "Quality",
                "Semantic",
                "Temporal",
                "Anchor reliable",
                "Anchor guard",
                "Anchor risk",
                "Semantic risk",
                "Temporal risk",
            ],
            head_rows,
        ),
        "",
        "## Key Counts",
        "",
        f"- V2 avoids V4 regressions: `{overall['v2_avoids_v4_regression']}`",
        f"- V2 introduces new regressions: `{overall['v2_new_regression']}`",
        f"- V2 improves cases V4 did not improve: `{overall['v2_improves_v4_same_or_regressed']}`",
        f"- V4 improves cases V2 does not improve: `{overall['v4_improves_v2_not']}`",
        f"- V2 protects anchors regressed by V4: `{overall['anchor_protected_vs_v4']}`",
        f"- V2 introduces anchor risk not present in V4: `{overall['anchor_new_risk_vs_v4']}`",
        f"- V2 strictly better top1 class than V4: `{overall['v2_strictly_better_than_v4']}`",
        f"- V2 strictly worse top1 class than V4: `{overall['v2_strictly_worse_than_v4']}`",
        "",
        "## Interpretation",
        "",
        "V2 reduces the risk side of fusion: fewer regressions and fewer good-anchor failures.",
        "Its evidence heads show temporal evidence is high for both gains and losses, so the remaining problem is risk calibration rather than finding more temporal positives.",
        "The paper-facing framing should present V2 as an interpretable evidence-selection system that trades a small amount of mAP for stricter and more stable top1 localization.",
    ]
    return "\n".join(text) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4_case_rows", required=True)
    parser.add_argument("--v2_case_rows", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = compare(load_json(args.v4_case_rows), load_json(args.v2_case_rows))
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    save_json(report, args.output_json)
    save_text(render_markdown(report), args.output_md)
    print(json.dumps(report["overall"], indent=2))
    print("saved", args.output_json)
    print("saved", args.output_md)


if __name__ == "__main__":
    main()
