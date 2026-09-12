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


def quality_key(row, prefix=""):
    value = float(row.get(f"{prefix}top1_iou" if prefix else "top1_iou", 0.0))
    return (value >= 0.7, value >= 0.5, value)


def outcome(row):
    ms_key = (row.get("ms_top1_iou", 0.0) >= 0.7, row.get("ms_top1_iou", 0.0) >= 0.5, row.get("ms_top1_iou", 0.0))
    new_key = quality_key(row)
    if new_key > ms_key:
        return "improved"
    if new_key < ms_key:
        return "regressed"
    return "same"


def score_bin(value):
    value = float(value)
    if value < 0.2:
        return "<0.20"
    if value < 0.3:
        return "0.20-0.30"
    if value < 0.4:
        return "0.30-0.40"
    if value < 0.5:
        return "0.40-0.50"
    if value < 0.6:
        return "0.50-0.60"
    if value < 0.7:
        return "0.60-0.70"
    return ">=0.70"


def compact_case(row):
    top = row.get("top_rows", [{}])[0] if row.get("top_rows") else {}
    return {
        "qid": row.get("qid"),
        "query": row.get("query", ""),
        "vid": row.get("vid", ""),
        "ms_category": row.get("ms_category", ""),
        "new_category": row.get("category", ""),
        "chosen_representation": row.get("chosen_representation", ""),
        "candidate_source": row.get("candidate_source", ""),
        "ms_top1_iou": row.get("ms_top1_iou", 0.0),
        "new_top1_iou": row.get("top1_iou", 0.0),
        "delta_top1_iou": float(row.get("top1_iou", 0.0)) - float(row.get("ms_top1_iou", 0.0)),
        "top_pred_quality": top.get("pred_quality", 0.0),
        "top_target_raw_iou": top.get("target_raw_iou", 0.0),
        "windows": row.get("windows", [])[:3],
        "gt_windows": row.get("gt_windows", []),
    }


def summarize_group(rows, key_fn):
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    out = {}
    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        outcomes = Counter(outcome(row) for row in group_rows)
        out[str(key)] = {
            "count": len(group_rows),
            "improved": outcomes.get("improved", 0),
            "regressed": outcomes.get("regressed", 0),
            "same": outcomes.get("same", 0),
            "good_regressed": sum(1 for row in group_rows if row.get("ms_category") == "good" and row.get("category") != "good"),
            "semantic_recovered": sum(
                1 for row in group_rows if row.get("ms_category") == "semantic_miss" and row.get("category") != "semantic_miss"
            ),
            "avg_delta_top1_iou": sum(float(row.get("top1_iou", 0.0)) - float(row.get("ms_top1_iou", 0.0)) for row in group_rows)
            / max(len(group_rows), 1),
        }
    return out


def analyze(rows):
    changed = [row for row in rows if row.get("changed_from_ms")]
    regressed = [row for row in changed if outcome(row) == "regressed"]
    improved = [row for row in changed if outcome(row) == "improved"]
    same = [row for row in changed if outcome(row) == "same"]
    good_regressed = [row for row in changed if row.get("ms_category") == "good" and row.get("category") != "good"]
    semantic_recovered = [row for row in changed if row.get("ms_category") == "semantic_miss" and row.get("category") != "semantic_miss"]

    def top_score(row):
        top = row.get("top_rows", [{}])[0] if row.get("top_rows") else {}
        return float(top.get("pred_quality", 0.0))

    return {
        "overall": {
            "total": len(rows),
            "changed": len(changed),
            "improved": len(improved),
            "regressed": len(regressed),
            "same": len(same),
            "good_regressed": len(good_regressed),
            "semantic_recovered": len(semantic_recovered),
            "intervention_precision": len(improved) / max(len(changed), 1),
        },
        "by_ms_category": summarize_group(changed, lambda row: row.get("ms_category", "")),
        "by_new_category": summarize_group(changed, lambda row: row.get("category", "")),
        "by_category_transition": summarize_group(changed, lambda row: f"{row.get('ms_category', '')}->{row.get('category', '')}"),
        "by_representation": summarize_group(changed, lambda row: row.get("chosen_representation", "")),
        "by_candidate_source": summarize_group(changed, lambda row: row.get("candidate_source", "")),
        "by_pred_quality_bin": summarize_group(changed, lambda row: score_bin(top_score(row))),
        "top_regressions": [
            compact_case(row)
            for row in sorted(regressed, key=lambda item: float(item.get("top1_iou", 0.0)) - float(item.get("ms_top1_iou", 0.0)))[:20]
        ],
        "top_good_regressions": [
            compact_case(row)
            for row in sorted(good_regressed, key=lambda item: float(item.get("top1_iou", 0.0)) - float(item.get("ms_top1_iou", 0.0)))[:20]
        ],
        "top_semantic_recoveries": [
            compact_case(row)
            for row in sorted(
                semantic_recovered,
                key=lambda item: float(item.get("top1_iou", 0.0)) - float(item.get("ms_top1_iou", 0.0)),
                reverse=True,
            )[:20]
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_json(args.case_rows_path)
    summary = analyze(rows)
    save_json(summary, args.output_path)
    print(json.dumps(summary["overall"], indent=2))
    print("saved analysis to", args.output_path)


if __name__ == "__main__":
    main()
