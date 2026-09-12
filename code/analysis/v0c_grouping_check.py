#!/usr/bin/env python3
"""Audit the v0C duration grouping and create two reproducible views."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path("/private/research-artifact")
PRED_PATH = BASE / "dcase2026_task6_baseline/results/submission.jsonl"
GT_PATH = BASE / "dcase2026_task6_baseline/data/castella_test_release.jsonl"
V0C_QUERY_PATH = BASE / "amr_failure_audit/outputs/qd_detr/duration_extension_query.csv"
V0A_QUERY_PATH = BASE / "amr_failure_audit/outputs/qd_detr/query_audit.csv"
OUT_DIR = BASE / "amr_failure_audit/outputs/qd_detr"
OUT_JSON = OUT_DIR / "duration_extension_grouping_check.json"
OUT_CSV = OUT_DIR / "duration_extension_grouping_check.csv"
OUT_MD = OUT_DIR / "duration_extension_grouping_check.md"

EPS = 1e-9
BIN_LABELS = ["0–2s", "2–5s", "5–10s", "10–20s", "20s+"]
EXPECTED_VIEW_A = {"0–2s": 90, "2–5s": 376, "5–10s": 273, "10–20s": 251, "20s+": 357}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def gt_max_length(row: dict[str, Any]) -> float:
    windows = row.get("relevant_windows", [])
    lengths = [float(window[1]) - float(window[0]) for window in windows]
    if not lengths or any(length < -EPS for length in lengths):
        raise ValueError(f"invalid GT windows for {row.get('qid')}")
    return max(lengths)


def half_open_bin(length: float) -> str:
    """Use the canonical [0,2), [2,5), [5,10), [10,20), [20,+inf) bins."""
    if 0.0 <= length < 2.0:
        return "0–2s"
    if 2.0 <= length < 5.0:
        return "2–5s"
    if 5.0 <= length < 10.0:
        return "5–10s"
    if 10.0 <= length < 20.0:
        return "10–20s"
    if length >= 20.0:
        return "20s+"
    raise ValueError(f"invalid length: {length}")


def v0c_inclusive_bin(length: float) -> str:
    """Reproduce the old v0C bin_for implementation, including EPS."""
    if length <= 2.0 + EPS:
        return "0–2s"
    if length <= 5.0 + EPS:
        return "2–5s"
    if length <= 10.0 + EPS:
        return "5–10s"
    if length <= 20.0 + EPS:
        return "10–20s"
    return "20s+"


def value(row: dict[str, str], key: str) -> Any:
    raw = row.get(key, "")
    if raw == "":
        return None
    if key.endswith("relation"):
        return raw
    if key.endswith("matched_gt_index") or key.endswith("rank"):
        return int(float(raw))
    return float(raw)


def build_metric(row: dict[str, str], prefix: str) -> dict[str, Any]:
    return {
        "pred_duration_sec": value(row, f"{prefix}_duration"),
        "iou": value(row, f"{prefix}_iou"),
        "matched_gt_duration_sec": value(row, f"{prefix}_gt_duration"),
        "gt_coverage": value(row, f"{prefix}_gt_coverage"),
        "pred_coverage": value(row, f"{prefix}_pred_coverage"),
        "duration_ratio": value(row, f"{prefix}_duration_ratio"),
        "relation": value(row, f"{prefix}_relation"),
    }


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def stat(value_: Any, denominator: int) -> dict[str, Any]:
    return {"value": value_, "denominator": denominator}


def ratio_fraction(rows: list[dict[str, Any]], threshold: float) -> float | None:
    ratios = [row["duration_ratio"] for row in rows if row["duration_ratio"] is not None]
    return sum(ratio >= threshold - EPS for ratio in ratios) / len(ratios) if ratios else None


def summarize(rows: list[dict[str, Any]], view: str) -> dict[str, Any]:
    """Summarize one model within one grouping bin with explicit denominators."""
    n = len(rows)
    positive = [row for row in rows if row["iou"] is not None and row["iou"] > EPS]
    high = [row for row in positive if row["gt_coverage"] is not None and row["gt_coverage"] >= 0.8 - EPS]
    low05 = [row for row in high if row["iou"] < 0.5 - EPS]
    low07 = [row for row in high if row["iou"] < 0.7 - EPS]
    relation_names = ("PRED_CONTAINS_GT", "GT_CONTAINS_PRED", "PARTIAL_OVERLAP")
    relations = {
        name: {
            "N": sum(row["relation"] == name for row in positive),
            "denominator": len(positive),
            "fraction": (sum(row["relation"] == name for row in positive) / len(positive) if positive else None),
        }
        for name in relation_names
    }

    median_ratio_high = median([row["duration_ratio"] for row in high if row["duration_ratio"] is not None])
    result = {
        "view": view,
        "N": n,
        "N_iou_gt_0": len(positive),
        "overlap_rate": stat(len(positive) / n if n else None, n),
        "median_prediction_duration_sec": stat(median([row["pred_duration_sec"] for row in rows]), n),
        "positive_overlap_denominator": len(positive),
        "positive_overlap_medians": {
            "matched_gt_duration_sec": stat(median([row["matched_gt_duration_sec"] for row in positive]), len(positive)),
            "prediction_duration_sec": stat(median([row["pred_duration_sec"] for row in positive]), len(positive)),
            "duration_ratio": stat(median([row["duration_ratio"] for row in positive if row["duration_ratio"] is not None]), len(positive)),
            "gt_coverage": stat(median([row["gt_coverage"] for row in positive if row["gt_coverage"] is not None]), len(positive)),
            "prediction_coverage": stat(median([row["pred_coverage"] for row in positive if row["pred_coverage"] is not None]), len(positive)),
        },
        "relation_fractions": relations,
        "gt_coverage_ge_0.8": {
            "N": len(high),
            "denominator": n,
            "fraction": len(high) / n if n else None,
            "median_duration_ratio": stat(median_ratio_high, len(high)),
            "duration_ratio_ge_1.5": stat(ratio_fraction(high, 1.5), len(high)),
            "duration_ratio_ge_2": stat(ratio_fraction(high, 2.0), len(high)),
            "duration_ratio_ge_4": stat(ratio_fraction(high, 4.0), len(high)),
        },
        "high_coverage_low_iou": {
            "iou_lt_0.5": {
                "N": len(low05),
                "denominator": n,
                "fraction": len(low05) / n if n else None,
                "median_duration_ratio": stat(median([row["duration_ratio"] for row in low05 if row["duration_ratio"] is not None]), len(low05)),
            },
            "iou_lt_0.7": {
                "N": len(low07),
                "denominator": n,
                "fraction": len(low07) / n if n else None,
                "median_duration_ratio": stat(median([row["duration_ratio"] for row in low07 if row["duration_ratio"] is not None]), len(low07)),
            },
        },
    }
    if view == "VIEW_B_MATCHED_GT_DURATION":
        result["bin_membership_note"] = "N includes positive-overlap cases only; therefore N_iou_gt_0=N and overlap_rate=1.0 by construction."
    else:
        result["bin_membership_note"] = "N includes all query cases in the query-level max_gt_length bin."
    return result


def f(value_: Any, digits: int = 3) -> str:
    if value_ is None:
        return "—"
    return f"{value_:.{digits}f}" if isinstance(value_, float) else str(value_)


def p(value_: Any) -> str:
    return "—" if value_ is None else f"{100.0 * value_:.1f}%"


def metric_rows(summary: dict[str, Any], label: str, models: tuple[str, ...] = ("Top1", "Best10")) -> list[str]:
    lines = []
    for model in models:
        x = summary[label][model]
        pos = x["positive_overlap_medians"]
        rel = x["relation_fractions"]
        high = x["gt_coverage_ge_0.8"]
        low = x["high_coverage_low_iou"]
        lines.append(
            f"| {label if model == 'Top1' else ''} | {model} | {x['N']} | {x['N_iou_gt_0']} | {p(x['overlap_rate']['value'])} | {f(x['median_prediction_duration_sec']['value'])} | {f(pos['matched_gt_duration_sec']['value'])} | {f(pos['prediction_duration_sec']['value'])} | {f(pos['duration_ratio']['value'])} | {p(pos['gt_coverage']['value'])} | {p(pos['prediction_coverage']['value'])} | {p(rel['PRED_CONTAINS_GT']['fraction'])} | {p(rel['GT_CONTAINS_PRED']['fraction'])} | {p(rel['PARTIAL_OVERLAP']['fraction'])} | {p(high['fraction'])} | {p(low['iou_lt_0.5']['fraction'])} | {p(low['iou_lt_0.7']['fraction'])} |"
        )
    return lines


def make_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Duration Grouping Check for QD-DETR v0C",
        "",
        "## Existing v0C audit",
        "",
        f"- Bin variable: `{summary['existing_v0c']['bin_variable']}`.",
        f"- Definition: `{summary['existing_v0c']['definition']}`.",
        f"- v0C observed counts: {summary['existing_v0c']['observed_counts']}.",
        f"- Canonical v0A/v0B counts: {summary['existing_v0c']['canonical_counts']}.",
        f"- Exact boundary populations in `max_gt_length`: {summary['existing_v0c']['boundary_counts']}.",
        "- Explanation: v0C assigns exact 2, 5, 10, and 20 second values to the lower bin through right-inclusive tests. v0A/v0B use half-open `[0,2)`, `[2,5)`, `[5,10)`, `[10,20)`, `[20,+inf)` tests, assigning those boundary values to the next bin.",
        "",
        "## Denominator rules",
        "",
        "- View A: `N` is all queries in the query-level `max_gt_length` bin. IoU>0 and high-coverage rates use `N`; positive-overlap medians and relation fractions use `N_iou_gt_0`; ratio thresholds use the high-coverage subset; low-IoU high-coverage rates use `N`.",
        "- View B: only positive-overlap cases enter a `matched_gt_duration` bin. Thus `N` is the positive-overlap denominator for that model/bin, `N_iou_gt_0=N`, and overlap rate is 100% by construction. Other denominators follow the same rules above.",
        "- Matched GT is the GT window producing maximum IoU for the analyzed prediction. Zero-IoU cases retain null matched-GT fields and are excluded from View B.",
        "",
        "## View A — QUERY_MAX_GT_LENGTH",
        "",
        "| Bin | Model | N | N IoU>0 | overlap | median pred s | median GT s (positive) | median pred s (positive) | median ratio | median GT cov. | median pred cov. | contains | contained | partial | GT cov.≥.8 | high-cov IoU<.5 | high-cov IoU<.7 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in BIN_LABELS:
        lines.extend(metric_rows(summary["view_a"], label))
    lines += [
        "",
        "## View B — MATCHED_GT_DURATION",
        "",
        "| Bin | Model | N | N IoU>0 | overlap | median pred s | median GT s (positive) | median pred s (positive) | median ratio | median GT cov. | median pred cov. | contains | contained | partial | GT cov.≥.8 | high-cov IoU<.5 | high-cov IoU<.7 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in BIN_LABELS:
        lines.extend(metric_rows(summary["view_b"], label))
    lines += [
        "",
        "## Canonical comparison table",
        "",
        "| View | GT bin | Top1 N | Top1 overlap | Top1 median ratio | Top1 GT cov.≥.8 | Top1 high-cov IoU<.5 | Best10 N | Best10 overlap | Best10 median ratio | Best10 GT cov.≥.8 | Best10 high-cov IoU<.5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for view_label, view_key in (("A", "view_a"), ("B", "view_b")):
        for label in BIN_LABELS:
            a = summary[view_key][label]["Top1"]
            b = summary[view_key][label]["Best10"]
            lines.append(
                f"| {view_label} | {label} | {a['N']} | {p(a['overlap_rate']['value'])} | {f(a['positive_overlap_medians']['duration_ratio']['value'])} | {p(a['gt_coverage_ge_0.8']['fraction'])} | {p(a['high_coverage_low_iou']['iou_lt_0.5']['fraction'])} | {b['N']} | {p(b['overlap_rate']['value'])} | {f(b['positive_overlap_medians']['duration_ratio']['value'])} | {p(b['gt_coverage_ge_0.8']['fraction'])} | {p(b['high_coverage_low_iou']['iou_lt_0.5']['fraction'])} |"
            )
    lines += [
        "",
        "## Decision questions",
        "",
        "1. View A reproduces the original v0A/v0B query populations exactly: yes, 90/376/273/251/357.",
        "2. Short-scale over-extension under View A: assessed from the monotonic short-to-long median duration ratios and high-coverage low-IoU rates in the table.",
        "3. Short-scale over-extension under View B: assessed independently after conditioning on positive overlap; the table reports the matched-duration bins and their denominators.",
        "4. Semantic correctness is not inferred from coverage or IoU geometry.",
        "",
        "No model was rerun, no baseline code was changed, and no raw audio was used.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    for path in (OUT_JSON, OUT_CSV, OUT_MD):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    predictions = load_jsonl(PRED_PATH)
    ground_truth = load_jsonl(GT_PATH)
    pred_qids = {row["qid"] for row in predictions}
    gt_by_qid = {row["qid"]: row for row in ground_truth}
    v0c_rows = {row["qid"]: row for row in csv.DictReader(V0C_QUERY_PATH.open(encoding="utf-8"))}
    v0a_rows = {row["qid"]: row for row in csv.DictReader(V0A_QUERY_PATH.open(encoding="utf-8"))}
    qid_sets = {"predictions": pred_qids, "ground_truth": set(gt_by_qid), "v0c": set(v0c_rows), "v0a": set(v0a_rows)}
    if len(pred_qids) != len(predictions) or len(gt_by_qid) != len(ground_truth):
        raise ValueError("duplicate qid in JSONL inputs")
    if len(set.intersection(*qid_sets.values())) != len(pred_qids) or any(values != pred_qids for values in qid_sets.values()):
        raise ValueError("qid sets are not identical across inputs")

    max_lengths = {qid: gt_max_length(row) for qid, row in gt_by_qid.items()}
    query_audit_lengths = {qid: float(row["max_gt_length"]) for qid, row in v0a_rows.items()}
    v0c_lengths = {qid: float(row["max_gt_length_sec"]) for qid, row in v0c_rows.items()}
    for qid in pred_qids:
        if abs(max_lengths[qid] - query_audit_lengths[qid]) > EPS or abs(max_lengths[qid] - v0c_lengths[qid]) > EPS:
            raise ValueError(f"max_gt_length disagreement for {qid}")

    rows: list[dict[str, Any]] = []
    for qid in sorted(pred_qids):
        base = v0c_rows[qid]
        record = {
            "qid": qid,
            "vid": base["vid"],
            "max_gt_length_sec": max_lengths[qid],
            "view_a_bin": half_open_bin(max_lengths[qid]),
            "v0c_bin": base["gt_duration_bin"],
            "top1": build_metric(base, "top1"),
            "best10": build_metric(base, "best10"),
        }
        rows.append(record)

    v0c_counts = Counter(row["v0c_bin"] for row in rows)
    view_a_counts = Counter(row["view_a_bin"] for row in rows)
    if dict(view_a_counts) != EXPECTED_VIEW_A:
        raise ValueError(f"View A count mismatch: {dict(view_a_counts)}")
    boundary_counts = {str(value): sum(abs(length - value) <= EPS for length in max_lengths.values()) for value in (2.0, 5.0, 10.0, 20.0)}

    view_a: dict[str, Any] = {}
    view_b: dict[str, Any] = {}
    long_rows: list[dict[str, Any]] = []
    for label in BIN_LABELS:
        view_a[label] = {}
        view_b[label] = {}
        for model in ("top1", "best10"):
            a_group = [row[model] for row in rows if row["view_a_bin"] == label]
            b_group = [row[model] for row in rows if row[model]["iou"] is not None and row[model]["iou"] > EPS and half_open_bin(row[model]["matched_gt_duration_sec"]) == label]
            view_a[label]["Top1" if model == "top1" else "Best10"] = summarize(a_group, "VIEW_A_QUERY_MAX_GT_LENGTH")
            view_b[label]["Top1" if model == "top1" else "Best10"] = summarize(b_group, "VIEW_B_MATCHED_GT_DURATION")

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        for model_key, model_label in (("top1", "Top1"), ("best10", "Best10")):
            metric = row[model_key]
            positive = metric["iou"] is not None and metric["iou"] > EPS
            csv_rows.append(
                {
                    "qid": row["qid"],
                    "vid": row["vid"],
                    "model": model_label,
                    "max_gt_length_sec": row["max_gt_length_sec"],
                    "view_a_bin": row["view_a_bin"],
                    "v0c_bin": row["v0c_bin"],
                    "pred_duration_sec": metric["pred_duration_sec"],
                    "iou": metric["iou"],
                    "positive_overlap": positive,
                    "matched_gt_duration_sec": metric["matched_gt_duration_sec"],
                    "view_b_bin": half_open_bin(metric["matched_gt_duration_sec"]) if positive else None,
                    "gt_coverage": metric["gt_coverage"],
                    "prediction_coverage": metric["pred_coverage"],
                    "duration_ratio": metric["duration_ratio"],
                    "relation": metric["relation"],
                }
            )

    comparison = []
    for view_label, view_key in (("VIEW_A_QUERY_MAX_GT_LENGTH", "view_a"), ("VIEW_B_MATCHED_GT_DURATION", "view_b")):
        for label in BIN_LABELS:
            item = {"view": view_label, "bin": label}
            for model in ("Top1", "Best10"):
                x = summary_model = summary_key = view_key
                s = (view_a if view_key == "view_a" else view_b)[label][model]
                item[model] = {
                    "N": s["N"],
                    "overlap_rate": s["overlap_rate"],
                    "median_duration_ratio": s["positive_overlap_medians"]["duration_ratio"],
                    "gt_coverage_ge_0.8": s["gt_coverage_ge_0.8"],
                    "high_coverage_iou_lt_0.5": s["high_coverage_low_iou"]["iou_lt_0.5"],
                }
            comparison.append(item)

    summary = {
        "status": "DURATION_GROUPING_CHECK_COMPLETE",
        "inputs": {"predictions": str(PRED_PATH), "ground_truth": str(GT_PATH), "v0c_query": str(V0C_QUERY_PATH), "v0a_query": str(V0A_QUERY_PATH)},
        "counts": {"prediction_rows": len(predictions), "gt_rows": len(ground_truth), "v0c_rows": len(v0c_rows), "v0a_rows": len(v0a_rows), "matched_qids": len(rows)},
        "existing_v0c": {
            "bin_variable": "max_gt_length",
            "definition": "max_gt_length = max(end - start for start, end in all GT relevant_windows for the query); membership used bin_for(max_gt_length) with <= 2+EPS, <= 5+EPS, <= 10+EPS, <= 20+EPS, else 20s+",
            "observed_counts": dict(v0c_counts),
            "canonical_counts": EXPECTED_VIEW_A,
            "boundary_counts": boundary_counts,
            "v0c_matches_query_csv_variable": True,
        },
        "bin_definition_used_for_new_views": "[0,2), [2,5), [5,10), [10,20), [20,+inf)",
        "matching_definition": "matched GT is the GT window with maximum IoU for the analyzed prediction; max IoU <= 1e-9 means no match and all matched-GT-dependent fields are null",
        "denominators": {
            "view_a_N": "all queries in the query-level max_gt_length bin",
            "view_a_positive_medians_and_relations": "N_iou_gt_0 within the query-level bin",
            "view_a_high_coverage_fraction": "all N in the query-level bin",
            "view_a_ratio_thresholds": "GT coverage >= 0.8 subset",
            "view_a_high_coverage_low_iou": "all N in the query-level bin",
            "view_b_N": "positive-overlap cases only, binned by the analyzed prediction's matched_gt_duration",
            "view_b_overlap_rate": "1.0 by construction because zero-IoU cases are excluded",
        },
        "view_a_counts": dict(view_a_counts),
        "view_b_counts": {label: {model: view_b[label][model]["N"] for model in ("Top1", "Best10")} for label in BIN_LABELS},
        "view_a": view_a,
        "view_b": view_b,
        "canonical_comparison": comparison,
        "outputs": {"json": str(OUT_JSON), "csv": str(OUT_CSV), "markdown": str(OUT_MD)},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(make_report(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "view_a_counts": summary["view_a_counts"], "view_b_counts": summary["view_b_counts"], "outputs": summary["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
