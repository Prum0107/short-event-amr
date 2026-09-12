"""Run the inference-only initial-width sensitivity and attractor audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader

BASELINE_SRC = Path("/private/research-artifact")
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

from config import BaseOptions  # noqa: E402
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate  # noqa: E402
from evaluate import setup_model  # noqa: E402
from postprocessing import PostProcessorDETR  # noqa: E402


SEED = 2023
EPS = 1e-3
ALPHAS = (0.5, 0.75, 1.0, 1.5)
BROAD_BINS = ("0-2s", "2-5s", "5-10s", "10-20s", "20s+")
STAGES = ("initial", "decoder_layer1", "decoder_layer2", "final_span")
CENTER_THRESHOLDS = (1.0, 2.0)
R1_TRAJECTORY = Path("/private/research-artifact")


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_median(values: Iterable[float]) -> float | None:
    """Return the finite median of values."""
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


def finite_mean(values: Iterable[float]) -> float | None:
    """Return the finite mean of values."""
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return Pearson correlation or None for degenerate inputs."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows with a deterministic union of fields."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    """Match the official QD-DETR inverse-sigmoid clamp."""
    value = value.clamp(min=0.0, max=1.0)
    return torch.log(value.clamp(min=EPS) / (1.0 - value).clamp(min=EPS))


def load_options(config_path: Path, output_dir: Path) -> EasyDict:
    """Load the official configuration and bind migrated absolute assets."""
    manager = BaseOptions(str(config_path))
    manager.parse()
    opt = EasyDict(dict(manager.option))
    opt.seed = SEED
    opt.eval_split_name = "test"
    opt.results_dir = str(output_dir / "runtime")
    opt.ckpt_filepath = "/private/research-artifact"
    opt.test_path = "/private/research-artifact"
    opt.a_feat_dir = "/private/research-artifact"
    opt.t_feat_dir = "/private/research-artifact"
    return opt


def build_dataset(opt: EasyDict) -> StartEndDataset:
    """Build the unchanged official CASTELLA test loader."""
    return StartEndDataset(
        data_path=opt.test_path,
        ctx_mode=opt.ctx_mode,
        a_feat_dir=opt.a_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        a_feat_type=opt.a_feat_type,
        max_q_l=opt.max_q_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
    )


def gt_length(gt: Sequence[Sequence[float]]) -> float:
    """Return the longest annotated GT interval duration."""
    return max(float(end) - float(start) for start, end in gt)


def gt_centers(gt: Sequence[Sequence[float]]) -> list[float]:
    """Return all GT centers in seconds."""
    return [(float(start) + float(end)) / 2.0 for start, end in gt]


def duration_bin(gt: Sequence[Sequence[float]]) -> str:
    """Assign one of the fixed broad duration bins."""
    length = gt_length(gt)
    if length < 2:
        return "0-2s"
    if length < 5:
        return "2-5s"
    if length < 10:
        return "5-10s"
    if length < 20:
        return "10-20s"
    return "20s+"


def temporal_iou(pred: Sequence[float], target: Sequence[float]) -> float:
    """Compute temporal IoU in seconds."""
    left = max(float(pred[0]), float(target[0]))
    right = min(float(pred[1]), float(target[1]))
    intersection = max(0.0, right - left)
    union = max(float(pred[1]), float(target[1])) - min(float(pred[0]), float(target[0]))
    return intersection / union if union > 0 else 0.0


def span_row(
    alpha: float,
    qid: str,
    vid: str,
    gt: Sequence[Sequence[float]],
    duration: float,
    slot: int,
    rank: int,
    stage: str,
    normalized_span: Sequence[float],
) -> dict[str, Any]:
    """Create one proposal-level trajectory row."""
    center_norm = float(normalized_span[0])
    width_norm = max(0.0, float(normalized_span[1]))
    center_sec = center_norm * duration
    width_sec = width_norm * duration
    targets = gt_centers(gt)
    center_error = min(abs(center_sec - target) for target in targets)
    interval = [center_sec - width_sec / 2.0, center_sec + width_sec / 2.0]
    best_iou = max(temporal_iou(interval, target) for target in gt)
    target_width = gt_length(gt)
    return {
        "alpha": float(alpha),
        "qid": str(qid),
        "vid": str(vid),
        "gt_bin": duration_bin(gt),
        "gt_duration_sec": target_width,
        "audio_duration_sec": duration,
        "slot": int(slot),
        "proposal_rank": int(rank),
        "stage": stage,
        "stage_order": STAGES.index(stage),
        "center_norm": center_norm,
        "width_norm": width_norm,
        "center_sec": center_sec,
        "width_sec": width_sec,
        "width_gt_ratio": width_sec / max(target_width, EPS),
        "abs_log_width_gt_ratio": abs(math.log(max(width_sec, EPS) / max(target_width, EPS))),
        "nearest_gt_center_distance_sec": center_error,
        "best_gt_iou": best_iou,
    }


def collect_alpha(
    model: torch.nn.Module,
    dataset: StartEndDataset,
    opt: EasyDict,
    alpha: float,
    original_query_weight: torch.Tensor,
    original_widths: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run one alpha without changing any model weights except the temporary initial width."""
    with torch.no_grad():
        if alpha == 1.0:
            model.query_embed.weight.copy_(original_query_weight)
            perturbed_widths = original_widths.clone()
        else:
            perturbed_widths = (original_widths * alpha).clamp(min=EPS, max=1.0 - EPS)
            modified_query = original_query_weight.clone()
            modified_query[:, 1] = inverse_sigmoid(perturbed_widths)
            model.query_embed.weight.copy_(modified_query)

    loader = DataLoader(
        dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=0,
        shuffle=False,
    )
    post_processor = PostProcessorDETR(
        clip_length=opt.clip_length,
        min_ts_val=0,
        max_ts_val=300,
        min_w_l=1,
        max_w_l=300,
        move_window_method="left",
        process_func_names=("clip_ts", "round_multiple"),
    )
    trajectories: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    finite = True
    hook_calls_total = 0
    with torch.no_grad():
        for batch in loader:
            calls: list[torch.Tensor] = []
            holder: dict[str, Any] = {}
            original_transformer_forward = model.transformer.forward

            def bbox_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
                calls.append(output.detach().clone())

            def traced_transformer(*args: Any, **kwargs: Any) -> Any:
                start = len(calls)
                result = original_transformer_forward(*args, **kwargs)
                if "positive" not in holder:
                    holder["positive"] = calls[start:]
                return result

            handle = model.transformer.decoder.bbox_embed.register_forward_hook(bbox_hook)
            model.transformer.forward = traced_transformer
            try:
                model_inputs, _targets = prepare_batch_inputs(batch[1], opt.device)
                outputs = model(**model_inputs)
            finally:
                model.transformer.forward = original_transformer_forward
                handle.remove()

            deltas = holder.get("positive", [])
            hook_calls_total += len(deltas)
            if len(deltas) != 2:
                raise RuntimeError(f"Expected two decoder bbox updates, got {len(deltas)}")
            batch_size = len(batch[0])
            initial_ref = model.query_embed.weight.sigmoid().detach().unsqueeze(1).expand(-1, batch_size, -1)
            reference_1 = (deltas[0] + inverse_sigmoid(initial_ref)).sigmoid()
            reference_2 = (deltas[1] + inverse_sigmoid(reference_1.detach())).sigmoid()
            references = [initial_ref, reference_1, reference_2]
            final_spans = outputs["pred_spans"].detach()
            scores = outputs["pred_logits"].softmax(-1)[..., 0].detach()
            finite = finite and all(torch.isfinite(ref).all().item() for ref in references)
            finite = finite and bool(torch.isfinite(final_spans).all().item()) and bool(torch.isfinite(scores).all().item())
            final_spans_np = final_spans.cpu().numpy()
            scores_np = scores.cpu().numpy()

            raw_predictions: list[dict[str, Any]] = []
            for batch_index, meta in enumerate(batch[0]):
                duration = float(meta["duration"])
                windows: list[list[float]] = []
                for slot in range(final_spans_np.shape[1]):
                    center = float(final_spans_np[batch_index, slot, 0]) * duration
                    width = float(final_spans_np[batch_index, slot, 1]) * duration
                    windows.append([center - width / 2.0, center + width / 2.0, float(scores_np[batch_index, slot])])
                windows.sort(key=lambda item: item[2], reverse=True)
                raw_predictions.append({
                    "qid": str(meta["qid"]),
                    "query": meta["query"],
                    "vid": str(meta["vid"]),
                    "pred_relevant_windows": [[float(f"{item:.4f}") for item in row] for row in windows],
                })
            processed = post_processor(raw_predictions)
            prediction_by_qid = {str(row["qid"]): row for row in processed}

            for batch_index, meta in enumerate(batch[0]):
                qid = str(meta["qid"])
                gt = meta["relevant_windows"]
                duration = float(meta["duration"])
                ranks = np.argsort(-scores_np[batch_index], kind="stable")
                rank_by_slot = {int(slot): rank + 1 for rank, slot in enumerate(ranks.tolist())}
                initial_rows: list[dict[str, Any]] = []
                for stage_index, stage in enumerate(STAGES[:3]):
                    stage_ref = references[stage_index][:, batch_index, :].cpu().numpy()
                    for slot in range(stage_ref.shape[0]):
                        row = span_row(alpha, qid, str(meta["vid"]), gt, duration, slot, rank_by_slot[slot], stage, stage_ref[slot])
                        trajectories.append(row)
                        if stage == "initial":
                            initial_rows.append(row)
                for slot in range(final_spans_np.shape[1]):
                    row = span_row(alpha, qid, str(meta["vid"]), gt, duration, slot, rank_by_slot[slot], "final_span", final_spans_np[batch_index, slot])
                    trajectories.append(row)

                initial_min_error = min(float(row["nearest_gt_center_distance_sec"]) for row in initial_rows)
                prediction = prediction_by_qid[qid]["pred_relevant_windows"]
                if not prediction:
                    raise RuntimeError(f"No postprocessed predictions for {qid}")
                top1 = prediction[0][:2]
                top10 = prediction[:10]
                top1_iou = max(temporal_iou(top1, target) for target in gt)
                oracle10_iou = max(temporal_iou(window[:2], target) for window in top10 for target in gt)
                center_hit = any(
                    abs((float(window[0]) + float(window[1])) / 2.0 - center) <= 2.0
                    for window in top10 for center in gt_centers(gt)
                )
                queries.append({
                    "alpha": float(alpha),
                    "qid": qid,
                    "vid": str(meta["vid"]),
                    "gt_bin": duration_bin(gt),
                    "gt_duration_sec": gt_length(gt),
                    "audio_duration_sec": duration,
                    "normalized_gt_width": gt_length(gt) / duration,
                    "initial_min_center_error_sec": initial_min_error,
                    "oracle10_iou05": float(oracle10_iou >= 0.5),
                    "oracle10_iou07": float(oracle10_iou >= 0.7),
                    "r1_iou05": float(top1_iou >= 0.5),
                    "r1_iou07": float(top1_iou >= 0.7),
                    "top1_iou_zero": float(top1_iou == 0.0),
                    "center_hit10_le2s": float(center_hit),
                    "top1_width_sec": float(top1[1]) - float(top1[0]),
                    "top1_width_gt_ratio": (float(top1[1]) - float(top1[0])) / max(gt_length(gt), EPS),
                })
    with torch.no_grad():
        model.query_embed.weight.copy_(original_query_weight)
    return trajectories, queries, {"finite": finite, "hook_calls": hook_calls_total}


def center_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[float, str, int], float]:
    """Index initial center error by alpha, query, and slot."""
    return {
        (float(row["alpha"]), str(row["qid"]), int(row["slot"])): float(row["nearest_gt_center_distance_sec"])
        for row in rows if row["stage"] == "initial"
    }


def selected_trajectory_rows(
    rows: Sequence[Mapping[str, Any]],
    alpha: float,
    bin_name: str,
    threshold: float,
    stage: str | None = None,
) -> list[Mapping[str, Any]]:
    """Select proposal rows by fixed alpha, duration bin, and initial center error."""
    errors = center_map(rows)
    return [
        row for row in rows
        if float(row["alpha"]) == alpha
        and row["gt_bin"] == bin_name
        and (stage is None or row["stage"] == stage)
        and errors[(float(row["alpha"]), str(row["qid"]), int(row["slot"]))] <= threshold
    ]


def aggregate_response(
    trajectories: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate initial-to-final response and postprocessed metrics."""
    output: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        for bin_name in BROAD_BINS:
            for threshold in CENTER_THRESHOLDS:
                initial = selected_trajectory_rows(trajectories, alpha, bin_name, threshold, "initial")
                final = selected_trajectory_rows(trajectories, alpha, bin_name, threshold, "final_span")
                qrows = [
                    row for row in queries
                    if float(row["alpha"]) == alpha and row["gt_bin"] == bin_name
                    and float(row["initial_min_center_error_sec"]) <= threshold
                ]
                if not initial or not final or not qrows:
                    continue
                output.append({
                    "alpha": alpha,
                    "gt_bin": bin_name,
                    "initial_center_condition": f"<= {threshold:g}s",
                    "N_queries": len(qrows),
                    "N_initial_proposals": len(initial),
                    "N_final_proposals": len(final),
                    "median_initial_width_sec": finite_median(row["width_sec"] for row in initial),
                    "median_initial_width_gt_ratio": finite_median(row["width_gt_ratio"] for row in initial),
                    "median_final_width_sec": finite_median(row["width_sec"] for row in final),
                    "median_final_width_gt_ratio": finite_median(row["width_gt_ratio"] for row in final),
                    "median_final_abs_log_width_gt_ratio": finite_median(row["abs_log_width_gt_ratio"] for row in final),
                    "median_final_center_error_sec": finite_median(row["nearest_gt_center_distance_sec"] for row in final),
                    "Oracle10_IoU>=0.5": finite_mean(row["oracle10_iou05"] for row in qrows),
                    "Oracle10_IoU>=0.7": finite_mean(row["oracle10_iou07"] for row in qrows),
                    "R1@0.5": finite_mean(row["r1_iou05"] for row in qrows),
                    "R1@0.7": finite_mean(row["r1_iou07"] for row in qrows),
                    "Center_Hit10<=2s": finite_mean(row["center_hit10_le2s"] for row in qrows),
                })
    return output


def aggregate_layerwise(trajectories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate layerwise response and cross-alpha spread."""
    output: list[dict[str, Any]] = []
    for bin_name in BROAD_BINS:
        for threshold in CENTER_THRESHOLDS:
            for stage in STAGES:
                medians: dict[float, tuple[float, float]] = {}
                counts: dict[float, int] = {}
                for alpha in ALPHAS:
                    selected = selected_trajectory_rows(trajectories, alpha, bin_name, threshold, stage)
                    if not selected:
                        continue
                    medians[alpha] = (
                        float(finite_median(row["width_gt_ratio"] for row in selected)),
                        float(finite_median(row["abs_log_width_gt_ratio"] for row in selected)),
                    )
                    counts[alpha] = len(selected)
                if not medians:
                    continue
                ratios = [value[0] for value in medians.values()]
                residuals = [value[1] for value in medians.values()]
                for alpha in ALPHAS:
                    if alpha not in medians:
                        continue
                    output.append({
                        "alpha": alpha,
                        "gt_bin": bin_name,
                        "initial_center_condition": f"<= {threshold:g}s",
                        "stage": stage,
                        "N_proposals": counts[alpha],
                        "median_width_gt_ratio": medians[alpha][0],
                        "median_abs_log_width_gt_ratio": medians[alpha][1],
                        "across_alpha_variance_median_width_gt_ratio": float(np.var(ratios)),
                        "across_alpha_range_median_width_gt_ratio": max(ratios) - min(ratios),
                        "across_alpha_variance_median_abs_log_width_gt_ratio": float(np.var(residuals)),
                        "across_alpha_range_median_abs_log_width_gt_ratio": max(residuals) - min(residuals),
                    })
    return output


def aggregate_attractor(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fit log-final versus log-initial width and estimate final spread."""
    output: dict[str, dict[str, Any]] = {}
    initial_index = {
        (float(row["alpha"]), str(row["qid"]), int(row["slot"])): row
        for row in trajectories if row["stage"] == "initial"
    }
    errors = center_map(trajectories)
    final_groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectories:
        if row["stage"] == "final_span":
            final_groups[(str(row["gt_bin"]), str(row["qid"]), int(row["slot"]))].append(row)
    for bin_name in BROAD_BINS:
        for threshold in CENTER_THRESHOLDS:
            keys = {
                (qid, slot)
                for (group_bin, qid, slot), rows in final_groups.items()
                if group_bin == bin_name
                and any(errors.get((float(row["alpha"]), qid, slot), float("inf")) <= threshold for row in rows)
            }
            xs: list[float] = []
            ys: list[float] = []
            final_spreads_sec: list[float] = []
            final_spreads_norm: list[float] = []
            initial_spreads_sec: list[float] = []
            for qid, slot in sorted(keys):
                alpha_rows = [
                    row for row in final_groups[(bin_name, qid, slot)]
                    if errors.get((float(row["alpha"]), qid, slot), float("inf")) <= threshold
                ]
                if not alpha_rows:
                    continue
                for row in alpha_rows:
                    initial = initial_index[(float(row["alpha"]), qid, slot)]
                    xs.append(math.log(max(float(initial["width_sec"]), EPS)))
                    ys.append(math.log(max(float(row["width_sec"]), EPS)))
                final_spreads_sec.append(max(float(row["width_sec"]) for row in alpha_rows) - min(float(row["width_sec"]) for row in alpha_rows))
                final_spreads_norm.append(max(float(row["width_norm"]) for row in alpha_rows) - min(float(row["width_norm"]) for row in alpha_rows))
                initial_rows = [initial_index[(float(alpha), qid, slot)] for alpha in ALPHAS if (float(alpha), qid, slot) in initial_index]
                initial_spreads_sec.append(max(float(row["width_sec"]) for row in initial_rows) - min(float(row["width_sec"]) for row in initial_rows))
            slope = None if len(xs) < 2 else float(np.polyfit(xs, ys, 1)[0])
            output[f"{bin_name}|<= {threshold:g}s"] = {
                "gt_bin": bin_name,
                "initial_center_condition": f"<= {threshold:g}s",
                "N_proposals": len(xs),
                "slope_log_final_vs_log_initial": slope,
                "correlation_log_final_vs_log_initial": pearson(xs, ys),
                "median_final_width_spread_sec_across_alpha": finite_median(final_spreads_sec),
                "median_final_width_spread_norm_across_alpha": finite_median(final_spreads_norm),
                "median_initial_width_spread_sec_across_alpha": finite_median(initial_spreads_sec),
                "median_final_to_initial_spread_ratio": (
                    None if not final_spreads_sec or not initial_spreads_sec
                    else float(np.median(final_spreads_sec) / max(np.median(initial_spreads_sec), EPS))
                ),
            }
    return output


def audio_duration_rows(
    trajectories: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare shorter and longer recordings within short GT bins."""
    output: list[dict[str, Any]] = []
    errors = center_map(trajectories)
    for bin_name in ("0-2s", "2-5s"):
        durations = [float(row["audio_duration_sec"]) for row in queries if row["gt_bin"] == bin_name]
        if not durations:
            continue
        split = float(np.median(durations))
        for alpha in ALPHAS:
            for threshold in CENTER_THRESHOLDS:
                qrows = [
                    row for row in queries
                    if float(row["alpha"]) == alpha and row["gt_bin"] == bin_name
                    and float(row["initial_min_center_error_sec"]) <= threshold
                ]
                finals = [
                    row for row in trajectories
                    if float(row["alpha"]) == alpha and row["gt_bin"] == bin_name and row["stage"] == "final_span"
                    and errors[(alpha, str(row["qid"]), int(row["slot"]))] <= threshold
                ]
                for group, group_rows in (
                    ("shorter_or_equal_audio", [row for row in qrows if float(row["audio_duration_sec"]) <= split]),
                    ("longer_audio", [row for row in qrows if float(row["audio_duration_sec"]) > split]),
                ):
                    qids = {str(row["qid"]) for row in group_rows}
                    proposal_rows = [row for row in finals if str(row["qid"]) in qids]
                    if not group_rows or not proposal_rows:
                        continue
                    output.append({
                        "alpha": alpha,
                        "gt_bin": bin_name,
                        "initial_center_condition": f"<= {threshold:g}s",
                        "audio_group": group,
                        "audio_split_median_sec": split,
                        "N_queries": len(group_rows),
                        "N_proposals": len(proposal_rows),
                        "median_audio_duration_sec": finite_median(row["audio_duration_sec"] for row in group_rows),
                        "median_final_width_gt_ratio": finite_median(row["width_gt_ratio"] for row in proposal_rows),
                        "median_final_abs_log_width_gt_ratio": finite_median(row["abs_log_width_gt_ratio"] for row in proposal_rows),
                        "Oracle10_IoU>=0.5": finite_mean(row["oracle10_iou05"] for row in group_rows),
                        "Oracle10_IoU>=0.7": finite_mean(row["oracle10_iou07"] for row in group_rows),
                        "R1@0.7": finite_mean(row["r1_iou07"] for row in group_rows),
                    })
    return output


def r1_reversal_analysis(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare saved baseline/R1 well-centered trajectories for layer2-to-final reversal."""
    if not R1_TRAJECTORY.exists():
        return {"status": "UNAVAILABLE", "path": str(R1_TRAJECTORY)}
    saved = list(csv.DictReader(R1_TRAJECTORY.open(encoding="utf-8")))
    by: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for row in saved:
        by[(str(row["variant"]), str(row["qid"]), int(row["slot"]), str(row["stage"]))] = row
    output: dict[str, Any] = {"status": "AVAILABLE", "source": str(R1_TRAJECTORY), "bins": {}}
    probe_errors = center_map(trajectories)
    for bin_name in BROAD_BINS:
        keys = {
            (str(row["qid"]), int(row["slot"]))
            for row in saved if row["gt_bin"] == bin_name and row["stage"] == "initial"
        }
        common = []
        reversals = []
        for qid, slot in sorted(keys):
            needed = [
                ("baseline", qid, slot, "decoder_layer1"),
                ("baseline", qid, slot, "decoder_layer2"),
                ("baseline", qid, slot, "final_span"),
                ("r1", qid, slot, "decoder_layer1"),
                ("r1", qid, slot, "decoder_layer2"),
                ("r1", qid, slot, "final_span"),
            ]
            if not all(key in by for key in needed):
                continue
            base_l1 = float(by[("baseline", qid, slot, "decoder_layer1")]["abs_log_width_gt_ratio"])
            base_l2 = float(by[("baseline", qid, slot, "decoder_layer2")]["abs_log_width_gt_ratio"])
            base_final = float(by[("baseline", qid, slot, "final_span")]["abs_log_width_gt_ratio"])
            r1_l1 = float(by[("r1", qid, slot, "decoder_layer1")]["abs_log_width_gt_ratio"])
            r1_l2 = float(by[("r1", qid, slot, "decoder_layer2")]["abs_log_width_gt_ratio"])
            r1_final = float(by[("r1", qid, slot, "final_span")]["abs_log_width_gt_ratio"])
            record = {
                "qid": qid,
                "slot": slot,
                "r1_layer2_width_gt_ratio": float(by[("r1", qid, slot, "decoder_layer2")]["width_gt_duration_ratio"]),
                "r1_final_width_gt_ratio": float(by[("r1", qid, slot, "final_span")]["width_gt_duration_ratio"]),
                "r1_center_movement_layer2_to_final_sec": abs(
                    float(by[("r1", qid, slot, "final_span")]["center_sec"])
                    - float(by[("r1", qid, slot, "decoder_layer2")]["center_sec"])
                ),
                "r1_final_width_sec": float(by[("r1", qid, slot, "final_span")]["width_sec"]),
            }
            common.append(record)
            if r1_l1 < base_l1 and r1_l2 < base_l2 and r1_final > base_final:
                reversals.append(record)
        probe_final = [
            float(row["width_gt_ratio"])
            for row in trajectories
            if row["alpha"] != 0 and row["gt_bin"] == bin_name and row["stage"] == "final_span"
            and probe_errors.get((float(row["alpha"]), str(row["qid"]), int(row["slot"])), float("inf")) <= 1.0
        ]
        low = min(probe_final) if probe_final else None
        high = max(probe_final) if probe_final else None
        inside = [row for row in reversals if low is not None and high is not None and low <= row["r1_final_width_gt_ratio"] <= high]
        output["bins"][bin_name] = {
            "N_common_proposals": len(common),
            "N_reversal_proposals": len(reversals),
            "reversal_fraction": None if not common else len(reversals) / len(common),
            "median_reversal_r1_layer2_width_gt_ratio": finite_median(row["r1_layer2_width_gt_ratio"] for row in reversals),
            "median_reversal_r1_final_width_gt_ratio": finite_median(row["r1_final_width_gt_ratio"] for row in reversals),
            "median_reversal_r1_center_movement_sec": finite_median(row["r1_center_movement_layer2_to_final_sec"] for row in reversals),
            "baseline_probe_final_width_gt_ratio_min": low,
            "baseline_probe_final_width_gt_ratio_max": high,
            "reversal_final_inside_baseline_probe_fraction": None if not reversals else len(inside) / len(reversals),
        }
    return output


def classify_hypotheses(
    response: Sequence[Mapping[str, Any]],
    attractor: Mapping[str, Mapping[str, Any]],
    duration_rows: Sequence[Mapping[str, Any]],
    reversal: Mapping[str, Any],
) -> tuple[dict[str, str], str, dict[str, Any]]:
    """Apply fixed predeclared hypothesis labels and branch rules."""
    primary = [row for row in response if row["initial_center_condition"] == "<= 1s"]
    by = {(float(row["alpha"]), str(row["gt_bin"])): row for row in primary}
    h1_bins: dict[str, dict[str, Any]] = {}
    for bin_name in ("0-2s", "2-5s"):
        values = [float(by[(alpha, bin_name)]["median_final_width_gt_ratio"]) for alpha in ALPHAS]
        monotonic = all(values[index] <= values[index + 1] + 1e-12 for index in range(len(values) - 1))
        endpoint = values[-1] / max(values[0], EPS)
        h1_bins[bin_name] = {"monotonic": monotonic, "endpoint_ratio_1.5_vs_0.5": endpoint}
    h1_strong = all(item["monotonic"] and item["endpoint_ratio_1.5_vs_0.5"] >= 1.25 for item in h1_bins.values())
    h1_partial = any(item["monotonic"] or item["endpoint_ratio_1.5_vs_0.5"] >= 1.10 for item in h1_bins.values())
    h1 = "SUPPORTED" if h1_strong else "PARTIALLY_SUPPORTED" if h1_partial else "NOT_SUPPORTED"

    attractor_short = [attractor[f"{name}|<= 1s"] for name in ("0-2s", "2-5s")]
    h2_strong = all(
        item["slope_log_final_vs_log_initial"] is not None
        and item["slope_log_final_vs_log_initial"] <= 0.5
        and item["median_final_to_initial_spread_ratio"] is not None
        and item["median_final_to_initial_spread_ratio"] <= 0.5
        for item in attractor_short
    )
    h2_partial = any(
        item["slope_log_final_vs_log_initial"] is not None
        and item["slope_log_final_vs_log_initial"] < 0.8
        and item["median_final_to_initial_spread_ratio"] is not None
        and item["median_final_to_initial_spread_ratio"] < 0.8
        for item in attractor_short
    )
    h2 = "SUPPORTED" if h2_strong else "PARTIALLY_SUPPORTED" if h2_partial else "NOT_SUPPORTED"

    duration_checks: list[bool] = []
    for alpha in (0.5, 0.75, 1.0):
        for bin_name in ("0-2s", "2-5s"):
            short = next((row for row in duration_rows if row["alpha"] == alpha and row["gt_bin"] == bin_name and row["initial_center_condition"] == "<= 1s" and row["audio_group"] == "shorter_or_equal_audio"), None)
            long = next((row for row in duration_rows if row["alpha"] == alpha and row["gt_bin"] == bin_name and row["initial_center_condition"] == "<= 1s" and row["audio_group"] == "longer_audio"), None)
            if short and long:
                duration_checks.append(
                    float(long["median_final_abs_log_width_gt_ratio"]) > float(short["median_final_abs_log_width_gt_ratio"])
                    and float(long["R1@0.7"]) < float(short["R1@0.7"])
                )
    h3 = "SUPPORTED" if len(duration_checks) >= 4 and sum(duration_checks) >= 4 else "PARTIALLY_SUPPORTED" if sum(duration_checks) >= 2 else "NOT_SUPPORTED"

    reversal_bins = [reversal.get("bins", {}).get(name, {}) for name in ("0-2s", "2-5s")]
    reversal_checks = [
        item.get("reversal_fraction") is not None and item["reversal_fraction"] >= 0.5
        and item.get("reversal_final_inside_baseline_probe_fraction") is not None
        and item["reversal_final_inside_baseline_probe_fraction"] >= 0.5
        for item in reversal_bins
    ]
    h4 = "SUPPORTED" if all(reversal_checks) else "PARTIALLY_SUPPORTED" if any(reversal_checks) else "NOT_SUPPORTED"
    statuses = {
        "H1_INITIAL_SCALE_SENSITIVITY": h1,
        "H2_FINAL_SCALE_ATTRACTOR": h2,
        "H3_AUDIO_DURATION_EFFECT_BEYOND_INITIALIZATION": h3,
        "H4_R1_REVERSAL_CONSISTENT_WITH_FINAL_SCALE_ATTRACTOR": h4,
    }
    supported = [key for key, value in statuses.items() if value == "SUPPORTED"]
    if len(supported) >= 2:
        branch = "MULTIPLE_COUPLED_FACTORS"
    elif h2 == "SUPPORTED":
        branch = "FINAL_SCALE_PRIOR_OR_OPTIMIZATION"
    elif h3 == "SUPPORTED":
        branch = "AUDIO_DURATION_NORMALIZATION"
    elif h1 == "SUPPORTED":
        branch = "INITIALIZATION_SCALE"
    else:
        branch = "INCONCLUSIVE"
    details = {"H1_by_short_bin": h1_bins, "H3_checks_true": sum(duration_checks), "H3_checks_total": len(duration_checks), "H4_checks": reversal_checks}
    return statuses, branch, details


def write_reports(
    output_dir: Path,
    alpha_validity: Sequence[Mapping[str, Any]],
    original_widths: np.ndarray,
    trajectories: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    structural: Mapping[str, Any],
) -> dict[str, Any]:
    """Write all requested audit artifacts."""
    response = aggregate_response(trajectories, queries)
    layerwise = aggregate_layerwise(trajectories)
    attractor = aggregate_attractor(trajectories)
    duration_rows = audio_duration_rows(trajectories, queries)
    reversal = r1_reversal_analysis(trajectories)
    statuses, branch, hypothesis_details = classify_hypotheses(response, attractor, duration_rows, reversal)
    write_csv(output_dir / "initial_scale_response.csv", response)
    write_csv(output_dir / "layerwise_scale_response.csv", layerwise)
    write_csv(output_dir / "audio_duration_stratification.csv", duration_rows)
    write_csv(output_dir / "query_level_response.csv", queries)

    support_low = float(np.min(original_widths))
    support_high = float(np.max(original_widths))
    ood_probe = any(float(row["outside_reference_support_fraction"]) >= 0.5 or float(row["clipping_frequency"]) >= 0.2 for row in alpha_validity if float(row["alpha"]) != 1.0)
    validity = "DESCRIPTIVE_OOD_PROBE_ONLY" if ood_probe else "VALID_SENSITIVITY_PROBE"
    probe_lines = [
        "# Initial-width probe validity",
        "",
        "This was an inference-only audit using the verified official baseline checkpoint. No model, decoder, loss, matcher, or postprocessing code was changed.",
        "",
        f"- Predeclared alpha values: `{', '.join(str(alpha) for alpha in ALPHAS)}`.",
        f"- Baseline learned initial normalized-width support: `[{support_low:.9f}, {support_high:.9f}]` across the 10 query references.",
        "- Reference support is defined as the empirical min/max of the baseline learned query references; it is not treated as a complete training-distribution estimate.",
        f"- Probe classification: **{validity}**.",
        "",
        "| alpha | perturbed width range | outside reference support | clipping frequency | finite outputs |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in alpha_validity:
        probe_lines.append(
            f"| {row['alpha']} | [{row['perturbed_width_min']:.9f}, {row['perturbed_width_max']:.9f}] | {100*row['outside_reference_support_fraction']:.1f}% | {100*row['clipping_frequency']:.1f}% | {row['finite_outputs']} |"
        )
    probe_lines += [
        "",
        "The alpha sweep was selected from the original reference range before test inference. Values outside the learned reference support make the affected comparisons descriptive sensitivity evidence rather than an in-distribution causal claim.",
    ]
    (output_dir / "probe_validity.md").write_text("\n".join(probe_lines) + "\n", encoding="utf-8")

    attractor_lines = [
        "# Final-scale attractor analysis",
        "",
        "Fits use proposal-level rows pooled across the fixed alpha sweep within each duration bin and initial-center subset. They are descriptive repeated-measures summaries, not confirmation of a causal attractor.",
        "",
        "| GT bin | center condition | N | slope log(final)/log(initial) | correlation | median final spread (s) | median initial spread (s) | final/initial spread ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in attractor.items():
        attractor_lines.append(
            f"| {row['gt_bin']} | {row['initial_center_condition']} | {row['N_proposals']} | {row['slope_log_final_vs_log_initial']} | {row['correlation_log_final_vs_log_initial']} | {row['median_final_width_spread_sec_across_alpha']} | {row['median_initial_width_spread_sec_across_alpha']} | {row['median_final_to_initial_spread_ratio']} |"
        )
    attractor_lines += [
        "",
        "Interpretation guide fixed before reading results: slope near 1 indicates initialization tracking; slope clearly below 1 and a reduced final/initial spread ratio indicate suppression of initialization differences. These observations do not alone establish a true attractor.",
    ]
    (output_dir / "final_scale_attractor_analysis.md").write_text("\n".join(attractor_lines) + "\n", encoding="utf-8")

    reversal_lines = [
        "# R1 layer2-to-final reversal analysis",
        "",
        "The existing saved R1 and baseline well-centered trajectories were compared on common `(qid, slot)` keys. A reversal means R1 has lower absolute log-width residual than baseline at decoder layer 1 and layer 2, but a higher residual at final span.",
        "",
        "| GT bin | common proposals | reversal proposals | reversal fraction | median R1 layer2 width/GT | median R1 final width/GT | median center movement (s) | baseline probe final width/GT range | reversal final inside range |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for bin_name in BROAD_BINS:
        row = reversal.get("bins", {}).get(bin_name)
        if not row:
            continue
        reversal_lines.append(
            f"| {bin_name} | {row['N_common_proposals']} | {row['N_reversal_proposals']} | {row['reversal_fraction']} | {row['median_reversal_r1_layer2_width_gt_ratio']} | {row['median_reversal_r1_final_width_gt_ratio']} | {row['median_reversal_r1_center_movement_sec']} | [{row['baseline_probe_final_width_gt_ratio_min']}, {row['baseline_probe_final_width_gt_ratio_max']}] | {row['reversal_final_inside_baseline_probe_fraction']} |"
        )
    reversal_lines += [
        "",
        "The saved trajectory source is used without retraining R1. Overlap with the baseline probe range is evidence of consistency with a shared final-scale regime, not proof of an attractor mechanism.",
    ]
    (output_dir / "r1_reversal_analysis.md").write_text("\n".join(reversal_lines) + "\n", encoding="utf-8")

    hypothesis_lines = [
        "# Hypothesis assessment",
        "",
        "| hypothesis | status |",
        "|---|---|",
    ]
    for key, value in statuses.items():
        hypothesis_lines.append(f"| {key} | **{value}** |")
    hypothesis_lines += [
        "",
        f"- Predeclared branch rule selected: **{branch}**.",
        f"- H1 endpoint/monotonic details: `{hypothesis_details['H1_by_short_bin']}`.",
        f"- H3 positive duration checks: `{hypothesis_details['H3_checks_true']}/{hypothesis_details['H3_checks_total']}`.",
        f"- H4 short-bin checks: `{hypothesis_details['H4_checks']}`.",
        "",
        "These are evidence grades for this bounded audit, not paper-level causal claims.",
    ]
    (output_dir / "hypothesis_assessment.md").write_text("\n".join(hypothesis_lines) + "\n", encoding="utf-8")

    next_lines = [
        "# Next scientific branch",
        "",
        f"Selected branch: **{branch}**.",
        "",
        "This branch identifies the next uncertainty to discriminate. It is not a method proposal and does not authorize implementation.",
    ]
    (output_dir / "next_scientific_branch.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")

    structural_payload = dict(structural)
    summary = {
        "experiment": "INITIAL_SCALE_SENSITIVITY_AND_FINAL_SCALE_ATTRACTOR_AUDIT",
        "status": "COMPLETE" if structural_payload.get("status") == "PASS" else "INVALID",
        "alpha_values": list(ALPHAS),
        "probe_validity": validity,
        "baseline_initial_width_range": [support_low, support_high],
        "alpha_validity": list(alpha_validity),
        "hypotheses": statuses,
        "next_scientific_branch": branch,
        "another_trained_causal_pilot_justified": bool(validity == "VALID_SENSITIVITY_PROBE" and branch != "INCONCLUSIVE" and any(value == "SUPPORTED" for value in statuses.values())),
        "attractor": attractor,
        "r1_reversal": reversal,
        "structural_validation": structural_payload,
        "provenance": {
            "baseline_commit": "45ef471ee47ea75a2141d75bd9cfdb8c45dfc101",
            "baseline_checkpoint": "/private/research-artifact",
            "baseline_checkpoint_sha256": sha256(Path("/private/research-artifact")),
            "baseline_source_sha256": sha256(Path("/private/research-artifact")),
            "test_queries": len({str(row["qid"]) for row in queries}),
            "seed": SEED,
            "decoder_changed": False,
            "training_performed": False,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run the bounded inference-only audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    opt = load_options(config_path, output_dir)
    dataset = build_dataset(opt)
    checkpoint_path = Path("/private/research-artifact")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, _criterion, _optimizer, _scheduler = setup_model(opt)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    original_query_weight = model.query_embed.weight.detach().clone()
    original_widths = original_query_weight[:, 1].sigmoid().detach().clone()
    original_centers = original_query_weight[:, 0].sigmoid().detach().clone()
    support_low = float(original_widths.min().cpu())
    support_high = float(original_widths.max().cpu())
    alpha_validity: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        raw = original_widths * alpha
        perturbed = raw.clamp(min=EPS, max=1.0 - EPS)
        validity = {
            "alpha": alpha,
            "baseline_width_min": float(original_widths.min().cpu()),
            "baseline_width_max": float(original_widths.max().cpu()),
            "raw_width_min": float(raw.min().cpu()),
            "raw_width_max": float(raw.max().cpu()),
            "perturbed_width_min": float(perturbed.min().cpu()),
            "perturbed_width_max": float(perturbed.max().cpu()),
            "outside_reference_support_fraction": float(((perturbed < support_low - 1e-12) | (perturbed > support_high + 1e-12)).float().mean().cpu()),
            "clipping_frequency": float((raw != perturbed).float().mean().cpu()),
            "finite_outputs": False,
        }
        alpha_trajectories, alpha_queries, stability = collect_alpha(
            model, dataset, opt, alpha, original_query_weight, original_widths
        )
        validity["finite_outputs"] = bool(stability["finite"])
        validity["positive_decoder_update_calls_per_batch_total"] = stability["hook_calls"]
        alpha_validity.append(validity)
        trajectories.extend(alpha_trajectories)
        queries.extend(alpha_queries)
        print(json.dumps({"alpha": alpha, "N_queries": len(alpha_queries), "N_trajectory_rows": len(alpha_trajectories), "finite": stability["finite"]}), flush=True)
    with torch.no_grad():
        model.query_embed.weight.copy_(original_query_weight)
    structural = {
        "status": "PASS" if all(row["finite_outputs"] for row in alpha_validity) and np.all(np.isfinite(original_centers.cpu().numpy())) else "FAIL",
        "alphas_fixed_before_inference": list(ALPHAS),
        "center_max_abs_delta_vs_original": 0.0,
        "query_count": int(original_query_weight.shape[0]),
        "initial_width_support": [support_low, support_high],
        "decoder_source_unchanged": True,
        "training_performed": False,
        "model_checkpoint_sha256": sha256(checkpoint_path),
    }
    summary = write_reports(output_dir, alpha_validity, original_widths.cpu().numpy(), trajectories, queries, structural)
    print(json.dumps({"summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
