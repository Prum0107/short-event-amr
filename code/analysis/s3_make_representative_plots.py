#!/usr/bin/env python3
"""Make S3 representative plots from cached frame scores only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/private/research-artifact")
OUT = ROOT / "amr_failure_audit/outputs/clap_evidence"


def iou(proposal, gt):
    ps, pe = proposal[:2]
    best = 0.0
    for gs, ge in gt:
        inter = max(0.0, min(pe, ge) - max(ps, gs))
        union = max(pe, ge) - min(ps, gs)
        best = max(best, inter / union if union > 0 else 0.0)
    return best


def best10(proposals, gt):
    candidates = [(iou(p, gt), idx, p) for idx, p in enumerate(proposals[:10])]
    return max(candidates, key=lambda x: (x[0], -x[1]))[2] if candidates else None


def add_interval(ax, proposal, color, label, linestyle="-"):
    if proposal is not None:
        ax.axvspan(float(proposal[0]), float(proposal[1]), color=color, alpha=0.18, label=label, linestyle=linestyle)


def main():
    summary = json.loads((OUT / "clap_temporal_evidence_summary.json").read_text())
    gt = {}
    with (ROOT / "dcase2026_task6_baseline/data/castella_test_release.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            gt[item["qid"]] = item
    qd = {}
    with (ROOT / "dcase2026_task6_baseline/results/submission.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            qd[item["qid"]] = item
    uv = {}
    with (ROOT / "amr_failure_audit/outputs/uvcom/submission_normalized.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            uv[item["qid"]] = item

    records = {r["qid"]: r for r in summary["per_query"]}
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for rep in summary["representative_cases"]:
        row = records[rep["qid"]]
        item = gt[rep["qid"]]
        gt_windows = [(float(s), float(e)) for s, e in item["relevant_windows"]]
        qd_props = qd[rep["qid"]]["pred_relevant_windows"]
        uv_props = uv[rep["qid"]]["pred_relevant_windows"]
        qd_best = best10(qd_props, gt_windows)
        uv_best = best10(uv_props, gt_windows)
        fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
        x = row["frame_starts"]
        ax.plot(x, row["frame_scores"], color="black", linewidth=0.9, label="MS-CLAP cosine")
        for idx, (s, e) in enumerate(gt_windows):
            ax.axvspan(s, e, color="forestgreen", alpha=0.22, label="GT window" if idx == 0 else None)
        add_interval(ax, qd_props[0] if qd_props else None, "tab:red", "QD Top1")
        add_interval(ax, uv_props[0] if uv_props else None, "tab:blue", "UVCOM Top1")
        add_interval(ax, qd_best, "tab:red", "QD Best10 candidate", linestyle="--")
        add_interval(ax, uv_best, "tab:blue", "UVCOM Best10 candidate", linestyle="--")
        ax.axvline(row["global_peak_start"], color="purple", linewidth=1.0, linestyle=":", label="global peak")
        if row["best_gt_frame_index"] is not None:
            ax.axvline(row["best_gt_frame_index"], color="darkorange", linewidth=1.0, linestyle="-.", label="best GT frame")
        ax.set_title(
            f"{rep['category']} | {rep['qid']} | {row['query']}\n"
            f"GT bin={row['duration_bin']}, max GT={row['max_gt_length']:.2f}s, "
            f"Hit@5={bool(row['clap_hit5'])}, GT width80={row['gt_peak_width_80']}, "
            f"QD Top1 IoU={row['QD_top1_iou']:.3f}, UVCOM Top1 IoU={row['UV_top1_iou']:.3f}"
        )
        ax.set_xlabel("time (sec; 1-sec frame support)")
        ax.set_ylabel("cosine similarity")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(OUT / rep["figure"], bbox_inches="tight")
        plt.close(fig)

    print(json.dumps({"figures": len(summary["representative_cases"]), "directory": str(fig_dir)}))


if __name__ == "__main__":
    main()
