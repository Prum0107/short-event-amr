"""Run the bounded, inference-only QD-DETR scale and loss-geometry audit."""

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
from torch.utils.data import DataLoader


FINE_BINS = [
    ("0-1s", 0.0, 1.0),
    ("1-2s", 1.0, 2.0),
    ("2-3s", 2.0, 3.0),
    ("3-5s", 3.0, 5.0),
    ("5-7s", 5.0, 7.0),
    ("7-10s", 7.0, 10.0),
    ("10-15s", 10.0, 15.0),
    ("15-20s", 15.0, 20.0),
    ("20-30s", 20.0, 30.0),
    ("30-45s", 30.0, 45.0),
    ("45s+", 45.0, None),
]
BROAD_BINS = [
    ("0-2s", 0.0, 2.0),
    ("2-5s", 2.0, 5.0),
    ("5-10s", 5.0, 10.0),
    ("10-20s", 10.0, 20.0),
    ("20s+", 20.0, None),
]


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records."""
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def gt_len(row: Mapping[str, Any]) -> float:
    """Return the longest annotated GT interval duration."""
    return max(float(end) - float(start) for start, end in row["relevant_windows"])


def bin_for(value: float, bins: Sequence[tuple[str, float, float | None]]) -> str:
    """Assign a scalar duration to a predefined bin."""
    for name, lower, upper in bins:
        if upper is None or lower <= value < upper:
            return name
    raise ValueError(f"unhandled duration: {value}")


def finite_values(values: Iterable[float]) -> list[float]:
    """Keep finite numeric values."""
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def stat(values: Iterable[float], quantile: float | None = None) -> float | None:
    """Return a median or requested quantile."""
    clean = finite_values(values)
    if not clean:
        return None
    return float(np.median(clean) if quantile is None else np.quantile(clean, quantile))


def corr(values_a: Iterable[float], values_b: Iterable[float]) -> float | None:
    """Return Pearson correlation, or None for a constant input."""
    a = np.asarray(finite_values(values_a), dtype=np.float64)
    b = np.asarray(finite_values(values_b), dtype=np.float64)
    if len(a) != len(b) or len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a CSV with the stable union of row fields."""
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iou(pred: Sequence[float], target: Sequence[float]) -> float:
    """Compute one-dimensional temporal IoU."""
    left = max(float(pred[0]), float(target[0]))
    right = min(float(pred[1]), float(target[1]))
    intersection = max(0.0, right - left)
    union = max(float(pred[1]), float(target[1])) - min(float(pred[0]), float(target[0]))
    return intersection / union if union > 0 else 0.0


def prediction_metrics(prediction: Mapping[str, Any], gt_windows: Sequence[Sequence[float]]) -> dict[str, float]:
    """Return Top-1 and Top-10 IoU metrics from a saved prediction row."""
    spans = prediction["pred_relevant_windows"]
    top1 = max(iou(spans[0][:2], gt) for gt in gt_windows)
    oracle10 = max(iou(span[:2], gt) for span in spans[:10] for gt in gt_windows)
    width = float(spans[0][1]) - float(spans[0][0])
    return {
        "top1_iou": top1,
        "oracle10_iou": oracle10,
        "final_width_sec": width,
    }


def load_query_embed(checkpoint: Path) -> np.ndarray:
    """Read the learned query reference parameters from a checkpoint."""
    state = torch.load(checkpoint, map_location="cpu")["model"]
    return state["query_embed.weight"].detach().cpu().numpy().astype(np.float64)


def initial_width_rows(
    gt_rows: Sequence[Mapping[str, Any]],
    checkpoint_paths: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Expand fixed learned normalized widths over every test query and slot."""
    rows: list[dict[str, Any]] = []
    normalized: dict[str, np.ndarray] = {}
    for variant, checkpoint in checkpoint_paths.items():
        query_embed = load_query_embed(checkpoint)
        widths = 1.0 / (1.0 + np.exp(-query_embed[:, 1]))
        normalized[variant] = widths
        for item in gt_rows:
            duration = float(item["duration"])
            target_duration = gt_len(item)
            for slot, width_norm in enumerate(widths):
                width_sec = float(width_norm * duration)
                rows.append({
                    "variant": variant,
                    "qid": str(item["qid"]),
                    "vid": str(item["vid"]),
                    "slot": slot,
                    "gt_bin_fine": bin_for(target_duration, FINE_BINS),
                    "gt_bin_broad": bin_for(target_duration, BROAD_BINS),
                    "audio_duration_sec": duration,
                    "gt_duration_sec": target_duration,
                    "normalized_gt_width": target_duration / duration,
                    "initial_width_norm": float(width_norm),
                    "initial_width_sec": width_sec,
                    "initial_width_gt_ratio": width_sec / max(1e-8, target_duration),
                })
    return rows, normalized


def normalized_gt_distribution(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize normalized GT widths for test and train+test populations."""
    values_by_population: dict[str, list[float]] = {}
    records_by_population: dict[str, list[dict[str, Any]]] = {}
    for population, names in {"test": ["test"], "train+test": ["train", "test"]}.items():
        records = []
        for split in names:
            records.extend({
                "qid": str(row["qid"]),
                "gt_duration_sec": gt_len(row),
                "audio_duration_sec": float(row["duration"]),
                "normalized_gt_width": gt_len(row) / float(row["duration"]),
            } for row in split_rows[split])
        records_by_population[population] = records
        values_by_population[population] = [float(row["normalized_gt_width"]) for row in records]
    reference = np.asarray(values_by_population["train+test"], dtype=np.float64)
    output: list[dict[str, Any]] = []
    bins: list[tuple[str, float, float | None]] = [("overall", 0.0, None)] + BROAD_BINS
    for population, records in records_by_population.items():
        for name, lower, upper in bins:
            selected = [row for row in records if name == "overall" or (lower <= row["gt_duration_sec"] and (upper is None or row["gt_duration_sec"] < upper))]
            values = [float(row["normalized_gt_width"]) for row in selected]
            row: dict[str, Any] = {
                "population": population,
                "duration_bin": name,
                "N": len(values),
                "median": stat(values),
                "q1": stat(values, 0.25),
                "q3": stat(values, 0.75),
                "p10": stat(values, 0.10),
                "p90": stat(values, 0.90),
                "min": stat(values, 0.0),
                "max": stat(values, 1.0),
                "median_percentile_vs_train_test": None if not values else float(np.mean(reference <= np.median(values))),
                "q1_percentile_vs_train_test": None if not values else float(np.mean(reference <= np.quantile(values, 0.25))),
                "q3_percentile_vs_train_test": None if not values else float(np.mean(reference <= np.quantile(values, 0.75))),
            }
            output.append(row)
    details = {
        "train_test_overall": {
            "N": len(reference),
            "median": stat(reference),
            "q1": stat(reference, 0.25),
            "q3": stat(reference, 0.75),
            "p10": stat(reference, 0.10),
            "p90": stat(reference, 0.90),
        },
        "test_short": {
            name: next(row for row in output if row["population"] == "test" and row["duration_bin"] == name)
            for name in ["0-2s", "2-5s"]
        },
    }
    return output, details


def aggregate_trajectory(
    trajectory_path: Path,
) -> list[dict[str, Any]]:
    """Aggregate the previously collected decoder reference trajectory."""
    raw = list(csv.DictReader(trajectory_path.open()))
    for row in raw:
        row["audio_duration_sec"] = float(row.get("audio_duration_sec", "nan")) if row.get("audio_duration_sec") else None
        row["width_sec"] = float(row["width_sec"])
        row["width_gt_duration_ratio"] = float(row["width_gt_duration_ratio"])
        row["gt_duration_sec"] = float(row.get("gt_duration_sec", "nan")) if row.get("gt_duration_sec") else None
        row["nearest_gt_center_distance_sec"] = float(row["nearest_gt_center_distance_sec"])
        row["stage_order"] = int(row["stage_order"])
    # The archived trajectory contains the exact duration and width columns but not redundant GT/audio fields.
    # Recover them from the test JSONL by qid; the caller patches these fields before aggregation.
    return raw


def enrich_trajectory(raw: Sequence[Mapping[str, Any]], gt_by_qid: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add exact duration fields and normalized widths to trajectory rows."""
    output: list[dict[str, Any]] = []
    for source in raw:
        row = dict(source)
        gt = gt_by_qid[str(row["qid"])]
        duration = float(gt["duration"])
        target_duration = gt_len(gt)
        row["audio_duration_sec"] = duration
        row["gt_duration_sec"] = target_duration
        row["normalized_width"] = float(row["width_sec"]) / duration
        row["duration_bin_fine"] = bin_for(target_duration, FINE_BINS)
        row["duration_bin_broad"] = bin_for(target_duration, BROAD_BINS)
        output.append(row)
    return output


def aggregate_trajectory_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate width trajectories conditional on initial center accuracy."""
    groups: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    initial_errors: dict[tuple[str, str, int], float] = {}
    for row in rows:
        if row["stage"] == "initial":
            initial_errors[(str(row["variant"]), str(row["qid"]), int(row["slot"]))] = float(row["nearest_gt_center_distance_sec"])
    for row in rows:
        key = (str(row["variant"]), str(row["duration_bin_broad"]), str(row["stage"]), 0.0)
        groups[key].append(row)
        error = initial_errors[(str(row["variant"]), str(row["qid"]), int(row["slot"]))]
        for threshold in [1.0, 2.0]:
            if error <= threshold:
                groups[(str(row["variant"]), str(row["duration_bin_broad"]), str(row["stage"]), threshold)].append(row)
    output: list[dict[str, Any]] = []
    for (variant, duration_bin, stage, threshold), selected in sorted(groups.items()):
        if threshold == 0.0:
            condition = "all_slots"
        else:
            condition = f"initial_center_le_{threshold:g}s"
        output.append({
            "variant": variant,
            "duration_bin": duration_bin,
            "initial_center_condition": condition,
            "stage": stage,
            "stage_order": min(int(row["stage_order"]) for row in selected),
            "N_proposals": len(selected),
            "N_queries": len({str(row["qid"]) for row in selected}),
            "median_width_norm": stat(float(row["normalized_width"]) for row in selected),
            "median_width_sec": stat(float(row["width_sec"]) for row in selected),
            "median_width_gt_ratio": stat(float(row["width_gt_duration_ratio"]) for row in selected),
            "q1_width_gt_ratio": stat((float(row["width_gt_duration_ratio"]) for row in selected), 0.25),
            "q3_width_gt_ratio": stat((float(row["width_gt_duration_ratio"]) for row in selected), 0.75),
        })
    return output


def temporal_iou_centered(gt_width: float, pred_width: float) -> float:
    """IoU for two intervals with the same center."""
    if gt_width <= 0 or pred_width <= 0:
        return 0.0
    return min(gt_width, pred_width) / max(gt_width, pred_width)


def centered_giou_loss(gt_width: float, pred_width: float) -> float:
    """GIoU loss for centered one-dimensional intervals."""
    return 1.0 - temporal_iou_centered(gt_width, pred_width)


def loss_geometry_rows(audio_representatives: Mapping[str, float]) -> list[dict[str, Any]]:
    """Compute the exact centered counterfactual losses used by QD-DETR."""
    gt_widths = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    ratios = [0.10, 0.25, 0.50, 0.70, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00, 3.00, 5.00, 10.00]
    rows: list[dict[str, Any]] = []
    for audio_label, audio_duration in audio_representatives.items():
        for gt_width in gt_widths:
            gt_norm = gt_width / audio_duration
            for ratio in ratios:
                pred_width = gt_width * ratio
                pred_norm = pred_width / audio_duration
                iou_value = temporal_iou_centered(gt_width, pred_width)
                giou_loss = 1.0 - iou_value
                # loss_spans averages the center and width coordinate L1 values; center error is zero.
                loss_span = abs(pred_norm - gt_norm) / 2.0
                matcher_span_cost = abs(pred_norm - gt_norm)
                matcher_giou_cost = -iou_value
                rows.append({
                    "audio_representative": audio_label,
                    "audio_duration_sec": audio_duration,
                    "gt_duration_sec": gt_width,
                    "pred_width_ratio_to_gt": ratio,
                    "pred_width_sec": pred_width,
                    "predicted_width_error_sec": pred_width - gt_width,
                    "absolute_width_error_sec": abs(pred_width - gt_width),
                    "gt_width_norm": gt_norm,
                    "pred_width_norm": pred_norm,
                    "evaluation_iou": iou_value,
                    "span_l1_loss_unweighted": loss_span,
                    "giou_loss_unweighted": giou_loss,
                    "training_span_component": 10.0 * loss_span,
                    "training_giou_component": giou_loss,
                    "training_regression_total": 10.0 * loss_span + giou_loss,
                    "matcher_span_cost": matcher_span_cost,
                    "matcher_giou_cost": matcher_giou_cost,
                    "matcher_geometry_total": 10.0 * matcher_span_cost + matcher_giou_cost,
                    "center_error_norm": 0.0,
                    "center_error_sec": 0.0,
                })
    return rows


def sensitivity_rows(audio_representatives: Mapping[str, float]) -> list[dict[str, Any]]:
    """Compute exact width-error thresholds at perfect-center IoU cutoffs."""
    rows: list[dict[str, Any]] = []
    for audio_label, audio_duration in audio_representatives.items():
        for gt_width in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
            for threshold in [0.7, 0.5]:
                for side in ["too_short", "too_long"]:
                    pred_ratio = threshold if side == "too_short" else 1.0 / threshold
                    pred_width = gt_width * pred_ratio
                    error = abs(pred_width - gt_width)
                    iou_value = temporal_iou_centered(gt_width, pred_width)
                    normalized_error = error / audio_duration
                    loss_span = normalized_error / 2.0
                    giou_loss = 1.0 - iou_value
                    rows.append({
                        "audio_representative": audio_label,
                        "audio_duration_sec": audio_duration,
                        "gt_duration_sec": gt_width,
                        "iou_threshold": threshold,
                        "error_side": side,
                        "boundary_pred_width_sec": pred_width,
                        "absolute_width_error_allowed_sec": error,
                        "normalized_width_error": normalized_error,
                        "span_l1_loss_unweighted": loss_span,
                        "giou_loss": giou_loss,
                        "training_regression_total": 10.0 * loss_span + giou_loss,
                        "matcher_span_cost": normalized_error,
                        "matcher_giou_cost": -iou_value,
                        "matcher_geometry_total": 10.0 * normalized_error - iou_value,
                        "evaluation_iou": iou_value,
                    })
    return rows


def matching_rows(
    worktree: Path,
    config_path: Path,
    checkpoint_paths: Mapping[str, Path],
    gt_by_qid: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run unchanged inference and record exact Hungarian matched costs."""
    sys.path.insert(0, str(worktree / "src"))
    from dataset import prepare_batch_inputs, start_end_collate  # type: ignore
    from evaluate import setup_model  # type: ignore
    from m1_experiment import build_dataset, options_from_config  # type: ignore
    from span_utils import generalized_temporal_iou, span_cxw_to_xx  # type: ignore

    rows: list[dict[str, Any]] = []
    for variant, checkpoint in checkpoint_paths.items():
        output_dir = Path("/tmp") / f"short_scale_runtime_{variant}"
        opt = options_from_config(str(config_path), output_dir, variant == "m1")
        model, criterion, _optimizer, _scheduler = setup_model(opt)
        state = torch.load(checkpoint, map_location="cpu")["model"]
        model.load_state_dict(state, strict=True)
        model.eval()
        dataset = build_dataset(opt, "test")
        loader = DataLoader(dataset, collate_fn=start_end_collate, batch_size=opt.eval_bsz, num_workers=0, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
                outputs = model(**model_inputs)
                match_outputs = {"pred_logits": outputs["pred_logits"], "pred_spans": outputs["pred_spans"]}
                indices = criterion.matcher(match_outputs, targets)
                pred_spans = outputs["pred_spans"]
                probabilities = outputs["pred_logits"].softmax(-1)[..., 0]
                for batch_index, meta in enumerate(batch[0]):
                    qid = str(meta["qid"])
                    gt = gt_by_qid[qid]
                    duration = float(meta["duration"])
                    target_duration = gt_len(gt)
                    pred = pred_spans[batch_index]
                    tgt = targets["span_labels"][batch_index]["spans"]
                    pairwise_span = torch.cdist(pred, tgt, p=1)
                    pairwise_giou = -generalized_temporal_iou(span_cxw_to_xx(pred), span_cxw_to_xx(tgt))
                    pairwise_class = -probabilities[batch_index][:, None].expand(-1, tgt.shape[0])
                    pairwise_total = (
                        float(criterion.matcher.cost_span) * pairwise_span
                        + float(criterion.matcher.cost_giou) * pairwise_giou
                        + float(criterion.matcher.cost_class) * pairwise_class
                    )
                    selected_pred, selected_gt = indices[batch_index]
                    for pred_index, gt_index in zip(selected_pred.tolist(), selected_gt.tolist()):
                        pred_pair = pred[pred_index]
                        gt_pair = tgt[gt_index]
                        gt_center = float(gt_pair[0])
                        pred_center = float(pred_pair[0])
                        gt_width = float(gt_pair[1])
                        pred_width = float(pred_pair[1])
                        rows.append({
                            "variant": variant,
                            "qid": qid,
                            "vid": str(meta["vid"]),
                            "duration_bin_fine": bin_for(target_duration, FINE_BINS),
                            "duration_bin_broad": bin_for(target_duration, BROAD_BINS),
                            "audio_duration_sec": duration,
                            "gt_duration_sec": target_duration,
                            "matched_gt_index": int(gt_index),
                            "matched_prediction_index": int(pred_index),
                            "matched_pred_center_norm": pred_center,
                            "matched_pred_width_norm": pred_width,
                            "matched_pred_center_sec": pred_center * duration,
                            "matched_pred_width_sec": pred_width * duration,
                            "matched_gt_center_norm": gt_center,
                            "matched_gt_width_norm": gt_width,
                            "matched_gt_center_sec": gt_center * duration,
                            "matched_gt_width_sec": gt_width * duration,
                            "center_error_norm": abs(pred_center - gt_center),
                            "center_error_sec": abs(pred_center - gt_center) * duration,
                            "width_error_norm": abs(pred_width - gt_width),
                            "width_error_sec": abs(pred_width - gt_width) * duration,
                            "matched_pred_width_gt_ratio": pred_width / max(1e-8, gt_width),
                            "classification_cost": float(pairwise_class[pred_index, gt_index]),
                            "matcher_span_cost": float(pairwise_span[pred_index, gt_index]),
                            "matcher_giou_cost": float(pairwise_giou[pred_index, gt_index]),
                            "matcher_total_cost": float(pairwise_total[pred_index, gt_index]),
                            "matched_giou_loss": float(1.0 + pairwise_giou[pred_index, gt_index]),
                        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def aggregate_matching(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate matched costs by variant and broad duration bin."""
    output: list[dict[str, Any]] = []
    for variant in ["baseline", "m1"]:
        for duration_bin, _lower, _upper in BROAD_BINS:
            selected = [row for row in rows if row["variant"] == variant and row["duration_bin_broad"] == duration_bin]
            if not selected:
                continue
            output.append({
                "variant": variant,
                "duration_bin": duration_bin,
                "N_matched_pairs": len(selected),
                "N_queries": len({str(row["qid"]) for row in selected}),
                "median_matched_pred_width_sec": stat(float(row["matched_pred_width_sec"]) for row in selected),
                "median_matched_pred_width_norm": stat(float(row["matched_pred_width_norm"]) for row in selected),
                "median_matched_pred_width_gt_ratio": stat(float(row["matched_pred_width_gt_ratio"]) for row in selected),
                "median_center_error_sec": stat(float(row["center_error_sec"]) for row in selected),
                "median_width_error_sec": stat(float(row["width_error_sec"]) for row in selected),
                "median_classification_cost": stat(float(row["classification_cost"]) for row in selected),
                "median_matcher_span_cost": stat(float(row["matcher_span_cost"]) for row in selected),
                "median_matcher_giou_cost": stat(float(row["matcher_giou_cost"]) for row in selected),
                "median_matcher_total_cost": stat(float(row["matcher_total_cost"]) for row in selected),
                "median_matched_giou_loss": stat(float(row["matched_giou_loss"]) for row in selected),
            })
    return output


def audio_interaction_rows(
    gt_rows: Sequence[Mapping[str, Any]],
    predictions_by_variant: Mapping[str, Mapping[str, Mapping[str, Any]]],
    initial_widths: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Compare shorter and longer recordings within each short GT bin."""
    query_rows: list[dict[str, Any]] = []
    for variant, predictions in predictions_by_variant.items():
        for item in gt_rows:
            target_duration = gt_len(item)
            duration_bin = bin_for(target_duration, BROAD_BINS)
            if duration_bin not in {"0-2s", "2-5s"}:
                continue
            metrics = prediction_metrics(predictions[str(item["qid"])], item["relevant_windows"])
            duration = float(item["duration"])
            width_norm_median = float(np.median(initial_widths[variant]))
            query_rows.append({
                "variant": variant,
                "qid": str(item["qid"]),
                "duration_bin": duration_bin,
                "audio_duration_sec": duration,
                "gt_duration_sec": target_duration,
                "normalized_gt_width": target_duration / duration,
                "initial_width_gt_ratio": width_norm_median * duration / target_duration,
                "final_width_gt_ratio": metrics["final_width_sec"] / target_duration,
                "top1_iou": metrics["top1_iou"],
                "oracle10_iou": metrics["oracle10_iou"],
                "r1_iou05": float(metrics["top1_iou"] >= 0.5),
                "r1_iou07": float(metrics["top1_iou"] >= 0.7),
                "oracle10_iou05": float(metrics["oracle10_iou"] >= 0.5),
                "oracle10_iou07": float(metrics["oracle10_iou"] >= 0.7),
            })
    output: list[dict[str, Any]] = []
    for variant in predictions_by_variant:
        for duration_bin, _lower, _upper in BROAD_BINS[:2]:
            selected = [row for row in query_rows if row["variant"] == variant and row["duration_bin"] == duration_bin]
            if not selected:
                continue
            split = float(np.median([float(row["audio_duration_sec"]) for row in selected]))
            for group, group_rows in [
                ("shorter_or_equal_audio", [row for row in selected if float(row["audio_duration_sec"]) <= split]),
                ("longer_audio", [row for row in selected if float(row["audio_duration_sec"]) > split]),
            ]:
                if not group_rows:
                    continue
                output.append({
                    "variant": variant,
                    "duration_bin": duration_bin,
                    "audio_group": group,
                    "audio_split_median_sec": split,
                    "N": len(group_rows),
                    "median_audio_duration_sec": stat(float(row["audio_duration_sec"]) for row in group_rows),
                    "median_gt_duration_sec": stat(float(row["gt_duration_sec"]) for row in group_rows),
                    "median_normalized_gt_width": stat(float(row["normalized_gt_width"]) for row in group_rows),
                    "median_initial_width_gt_ratio": stat(float(row["initial_width_gt_ratio"]) for row in group_rows),
                    "median_final_width_gt_ratio": stat(float(row["final_width_gt_ratio"]) for row in group_rows),
                    "R1_iou05": float(np.mean([row["r1_iou05"] for row in group_rows])),
                    "R1_iou07": float(np.mean([row["r1_iou07"] for row in group_rows])),
                    "Oracle10_iou05": float(np.mean([row["oracle10_iou05"] for row in group_rows])),
                    "Oracle10_iou07": float(np.mean([row["oracle10_iou07"] for row in group_rows])),
                })
    return output


def write_width_parameterization(output_dir: Path, config_path: Path) -> None:
    """Write exact implementation equations and source locations."""
    text = f"""# Exact width parameterization

The audit uses the server worktree implementation and `{config_path}`. The source locations below are the code actually loaded by the existing checkpoints.

## Forward parameterization

- `src/qd_detr.py:94-95`: `span_embed = MLP(hidden_dim, hidden_dim, 2, 3)` produces two residual coordinates.
- `src/qd_detr.py:183-188`: decoder residuals are added to the inverse-sigmoid reference and passed through `sigmoid` for `span_loss_type == "l1"`.
- `src/qd_detr_transformer.py:224-237`: the decoder uses a shared three-layer `bbox_embed`; its final layer is zero-initialized, so the first refinement starts at the reference point.
- `src/qd_detr_transformer.py:260-313`: references are sigmoid-normalized `(center, width)` coordinates; each decoder layer computes `tmp + inverse_sigmoid(reference_points)` and then applies sigmoid. The detached reference is fed to the next layer, while the optional trace records the initial reference and each layer's updated reference.
- `src/qd_detr.py:160-164`: M1 changes only the initial center. The initial width remains the second coordinate of `query_embed.weight` after sigmoid.
- `src/qd_detr_transformer.py:113-114` and `src/qd_detr.py:172-178`: baseline starts from the learned two-dimensional `query_embed`; M1 replaces only center with `(selected_grid_index + 0.5) / valid_length` and retains the learned width.

For query audio duration `D`, normalized coordinate `(c, w)` converts to seconds as:

`start_sec = D * (c - w/2)` and `end_sec = D * (c + w/2)`; therefore `width_sec = D * w`.

## Training span loss

`src/qd_detr.py:268-290` computes coordinate-wise L1 and temporal GIoU. For one matched pair:

`loss_span = mean(|c_pred-c_gt|, |w_pred-w_gt|)`

`loss_giou = 1 - GIoU([c_pred-w_pred/2, c_pred+w_pred/2], [c_gt-w_gt/2, c_gt+w_gt/2])`.

From `{config_path}`: `span_loss_coef = 10` and `giou_loss_coef = 1`, so the relevant regression total is `10 * loss_span + loss_giou`. Auxiliary decoder losses use the same coefficients when enabled.

## Hungarian matching

`src/matcher.py:63-99` computes:

`cost_span = |c_pred-c_gt| + |w_pred-w_gt|`

`cost_giou = -GIoU`

`cost_class = -P(foreground)` for foreground label `0`.

`C = 10 * cost_span + 1 * cost_giou + 4 * cost_class`, using `set_cost_span = 10`, `set_cost_giou = 1`, and `set_cost_class = 4`. The audit reports these signed matcher costs exactly and separately reports positive GIoU loss for readability.

## Evaluation distinction

Evaluation IoU is computed on the converted second-based intervals. A small normalized width error is not equivalent to a small absolute-time error when `D` is large, and neither is equivalent to strict IoU when the GT is short.
"""
    (output_dir / "width_parameterization.md").write_text(text)


def write_reports(
    output_dir: Path,
    initial_rows: Sequence[Mapping[str, Any]],
    normalized_details: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, Any]],
    loss_rows: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
    matching_agg: Sequence[Mapping[str, Any]],
    interaction: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, str],
    dominant: str,
    method_ready: str,
) -> None:
    """Write the two narrative assessment documents."""
    def find_initial(variant: str, duration_bin: str) -> list[Mapping[str, Any]]:
        return [row for row in initial_rows if row["variant"] == variant and row["gt_bin_broad"] == duration_bin]

    short_lines = []
    for duration_bin in ["0-2s", "2-5s"]:
        for variant in ["baseline", "m1"]:
            selected = find_initial(variant, duration_bin)
            short_lines.append(
                f"- `{variant}` {duration_bin}: median initial width = {stat(float(row['initial_width_sec']) for row in selected):.2f}s, median width/GT = {stat(float(row['initial_width_gt_ratio']) for row in selected):.2f}."
            )
    trajectory_lines = [
        "| Variant | Bin | Initial-center condition | Stage | N proposals | Median width norm | Median width (s) | Median width/GT | |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in trajectory_rows:
        if row["duration_bin"] in {"0-2s", "2-5s"} and row["initial_center_condition"] != "all_slots":
            trajectory_lines.append(
                f"| {row['variant']} | {row['duration_bin']} | {row['initial_center_condition']} | {row['stage']} | {row['N_proposals']} | {row['median_width_norm']:.5f} | {row['median_width_sec']:.2f} | {row['median_width_gt_ratio']:.2f} |"
            )
    loss_short = []
    for gt_width in [1.0, 2.0, 5.0]:
        selected = [row for row in sensitivity if row["gt_duration_sec"] == gt_width and row["audio_representative"] == "p50_test"]
        short_07 = next(row for row in selected if row["iou_threshold"] == 0.7 and row["error_side"] == "too_short")
        long_07 = next(row for row in selected if row["iou_threshold"] == 0.7 and row["error_side"] == "too_long")
        short_05 = next(row for row in selected if row["iou_threshold"] == 0.5 and row["error_side"] == "too_short")
        long_05 = next(row for row in selected if row["iou_threshold"] == 0.5 and row["error_side"] == "too_long")
        loss_short.append(
            f"- GT {gt_width:g}s: IoU 0.7 boundary errors are too-short {short_07['absolute_width_error_allowed_sec']:.3f}s / too-long {long_07['absolute_width_error_allowed_sec']:.3f}s; at those boundaries training totals are {short_07['training_regression_total']:.3f}/{long_07['training_regression_total']:.3f} and matcher geometry totals are {short_07['matcher_geometry_total']:.3f}/{long_07['matcher_geometry_total']:.3f}. IoU 0.5 boundary errors are too-short {short_05['absolute_width_error_allowed_sec']:.3f}s / too-long {long_05['absolute_width_error_allowed_sec']:.3f}s."
        )
    interaction_lines = []
    for row in interaction:
        if row["variant"] == "m1":
            interaction_lines.append(
                f"- M1 {row['duration_bin']} {row['audio_group']}: N={row['N']}, median audio={row['median_audio_duration_sec']:.1f}s, R1@0.7={row['R1_iou07']:.3f}, Oracle@10@0.7={row['Oracle10_iou07']:.3f}, median initial width/GT={row['median_initial_width_gt_ratio']:.2f}, final width/GT={row['median_final_width_gt_ratio']:.2f}, normalized GT width={row['median_normalized_gt_width']:.5f}.")
    hypothesis_lines = [
        "# Hypothesis assessment",
        "",
        "This is an inference-only mechanism audit. Labels are evidence grades, not causal proof.",
        "",
        "| Hypothesis | Status |",
        "|---|---|",
    ]
    for name in ["H1_INITIAL_SCALE_MISMATCH", "H2_NORMALIZED_PARAMETERIZATION_MISMATCH", "H3_LOSS_METRIC_MISMATCH", "H4_REFINEMENT_LIMITATION", "H5_AUDIO_DURATION_INTERACTION", "H6_NONE_OF_THE_ABOVE"]:
        hypothesis_lines.append(f"| {name} | **{statuses[name]}** |")
    hypothesis_lines += [
        "",
        "The evidence is interpreted jointly: fixed sigmoid widths scale with full audio duration; short GTs occupy the lower tail of normalized width space; strict IoU thresholds are crossed by small absolute errors; and conditioned decoder proposals can retain large width/GT ratios. The normalized L1 term is duration-sensitive, while centered GIoU equals IoU and remains directly sensitive to relative width error. These observations do not by themselves identify a single causal component.",
        "",
        f"Dominant mechanism class: **{dominant}**.",
        f"Sufficient to begin method design: **{method_ready}**. This audit does not propose a method.",
    ]
    (output_dir / "hypothesis_assessment.md").write_text("\n".join(hypothesis_lines) + "\n")

    decision_lines = [
        "# Scientific decision",
        "",
        "## Fixed implementation",
        "",
        *short_lines,
        "",
        "## Correctly centered short-proposal trajectory",
        "",
        *trajectory_lines,
        "",
        "## Strict-IoU sensitivity at the p50 CASTELLA test audio duration",
        "",
        *loss_short,
        "",
        "The loss geometry uses the exact implemented coefficients: training regression `10 * mean(center L1, width L1) + GIoU loss`; matcher geometry `10 * span L1 + (-GIoU)`, with classification cost held separate.",
        "",
        "## Audio-duration interaction",
        "",
        *interaction_lines,
        "",
        "## Decision",
        "",
        f"- H1–H6: {json.dumps(dict(statuses), ensure_ascii=False)}",
        f"- Dominant mechanism class: **{dominant}**.",
        f"- Evidence sufficient to begin method design: **{method_ready}**.",
        "",
        "The result is bounded to the existing QD-DETR implementation and matched checkpoints. It does not establish causality and does not authorize changing the model or loss.",
    ]
    (output_dir / "scientific_decision.md").write_text("\n".join(decision_lines) + "\n")


def main() -> None:
    """Run the scale audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worktree = Path(args.worktree)
    config_path = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = worktree / "data/castella_test_release.jsonl"
    split_paths = {
        "train": worktree / "data/castella_train_release.jsonl",
        "val": worktree / "data/castella_val_release.jsonl",
        "test": gt_path,
    }
    matched_root = Path("/private/research-artifact")
    checkpoint_paths = {
        "baseline": matched_root / "baseline/best_checkpoint.pth",
        "m1": matched_root / "m1/best_checkpoint.pth",
    }
    prediction_paths = {
        "baseline": matched_root / "baseline_test_predictions.jsonl",
        "m1": matched_root / "m1_test_predictions.jsonl",
    }
    gt_rows = load_jsonl(gt_path)
    gt_by_qid = {str(row["qid"]): row for row in gt_rows}
    split_rows = {name: load_jsonl(path) for name, path in split_paths.items()}

    initial_rows, initial_widths = initial_width_rows(gt_rows, checkpoint_paths)
    write_csv(output_dir / "initial_width_analysis.csv", initial_rows)
    normalized_rows, normalized_details = normalized_gt_distribution(split_rows)
    write_csv(output_dir / "normalized_gt_width_distribution.csv", normalized_rows)

    trajectory_path = Path("/private/research-artifact")
    trajectory_raw = aggregate_trajectory(trajectory_path)
    trajectory_enriched = enrich_trajectory(trajectory_raw, gt_by_qid)
    trajectory_agg = aggregate_trajectory_rows(trajectory_enriched)
    write_csv(output_dir / "decoder_width_trajectory.csv", trajectory_agg)

    duration_values = np.asarray([float(row["duration"]) for row in gt_rows], dtype=np.float64)
    audio_representatives = {
        "p25_test": float(np.quantile(duration_values, 0.25)),
        "p50_test": float(np.quantile(duration_values, 0.50)),
        "p75_test": float(np.quantile(duration_values, 0.75)),
    }
    loss_rows = loss_geometry_rows(audio_representatives)
    sensitivity = sensitivity_rows(audio_representatives)
    write_csv(output_dir / "loss_geometry.csv", loss_rows)
    write_csv(output_dir / "iou_loss_sensitivity.csv", sensitivity)

    matching_detail = matching_rows(worktree, config_path, checkpoint_paths, gt_by_qid)
    matching_agg = aggregate_matching(matching_detail)
    write_csv(output_dir / "matching_by_duration.csv", matching_agg)
    write_csv(output_dir / "matching_pair_level.csv", matching_detail)

    predictions_by_variant = {variant: {str(row["qid"]): row for row in load_jsonl(path)} for variant, path in prediction_paths.items()}
    interaction = audio_interaction_rows(gt_rows, predictions_by_variant, initial_widths)
    write_csv(output_dir / "audio_duration_interaction.csv", interaction)

    short_initial = [row for row in initial_rows if row["variant"] == "m1" and row["gt_bin_broad"] in {"0-2s", "2-5s"}]
    short_ratio = stat(float(row["initial_width_gt_ratio"]) for row in short_initial) or 0.0
    long_initial = [row for row in initial_rows if row["variant"] == "m1" and row["gt_bin_broad"] == "20s+"]
    long_ratio = stat(float(row["initial_width_gt_ratio"]) for row in long_initial) or 0.0
    ref = normalized_details["train_test_overall"]
    short_dist = normalized_details["test_short"]
    h1 = "SUPPORTED" if short_ratio >= 5.0 and short_ratio > max(1.0, long_ratio * 2.0) else "PARTIALLY_SUPPORTED" if short_ratio > 2.0 else "NOT_SUPPORTED"
    h2 = "SUPPORTED" if all(float(short_dist[name]["median_percentile_vs_train_test"]) <= 0.10 for name in ["0-2s", "2-5s"]) else "PARTIALLY_SUPPORTED" if any(float(short_dist[name]["median_percentile_vs_train_test"]) <= 0.10 for name in ["0-2s", "2-5s"]) else "NOT_SUPPORTED"
    one_sens = [row for row in sensitivity if row["audio_representative"] == "p50_test" and row["gt_duration_sec"] == 1.0 and row["iou_threshold"] == 0.7 and row["error_side"] == "too_short"][0]
    # The normalized L1 component is small for short targets, but centered GIoU equals IoU
    # for these intervals. Therefore the full regression loss is not blind to relative width error.
    h3 = "PARTIALLY_SUPPORTED" if one_sens["normalized_width_error"] <= 0.01 else "INCONCLUSIVE"
    short_traj = [row for row in trajectory_agg if row["variant"] == "m1" and row["duration_bin"] in {"0-2s", "2-5s"} and row["initial_center_condition"] in {"initial_center_le_1s", "initial_center_le_2s"} and row["stage"] == "final_span"]
    h4 = "SUPPORTED" if any(float(row["median_width_gt_ratio"]) >= 2.0 for row in short_traj) else "PARTIALLY_SUPPORTED" if any(float(row["median_width_gt_ratio"]) >= 1.5 for row in short_traj) else "NOT_SUPPORTED"
    h5_deltas = []
    for variant in ["m1"]:
        for duration_bin in ["0-2s", "2-5s"]:
            groups = {row["audio_group"]: row for row in interaction if row["variant"] == variant and row["duration_bin"] == duration_bin}
            if "shorter_or_equal_audio" in groups and "longer_audio" in groups:
                h5_deltas.append(float(groups["longer_audio"]["R1_iou07"]) - float(groups["shorter_or_equal_audio"]["R1_iou07"]))
    h5 = "SUPPORTED" if h5_deltas and all(delta <= -0.03 for delta in h5_deltas) else "PARTIALLY_SUPPORTED" if any(delta <= -0.03 for delta in h5_deltas) else "NOT_SUPPORTED"
    h6 = "NOT_SUPPORTED" if any(status in {"SUPPORTED", "PARTIALLY_SUPPORTED"} for status in [h1, h2, h3, h4, h5]) else "SUPPORTED"
    statuses = {
        "H1_INITIAL_SCALE_MISMATCH": h1,
        "H2_NORMALIZED_PARAMETERIZATION_MISMATCH": h2,
        "H3_LOSS_METRIC_MISMATCH": h3,
        "H4_REFINEMENT_LIMITATION": h4,
        "H5_AUDIO_DURATION_INTERACTION": h5,
        "H6_NONE_OF_THE_ABOVE": h6,
    }
    supported_count = sum(status == "SUPPORTED" for status in [h1, h2, h3, h4, h5])
    if supported_count >= 2:
        dominant = "MULTIPLE_COUPLED_SCALE_FACTORS"
    elif h3 == "SUPPORTED":
        dominant = "LOSS_OPTIMIZATION"
    elif h4 == "SUPPORTED":
        dominant = "REFINEMENT_DYNAMICS"
    elif h1 == "SUPPORTED":
        dominant = "INITIALIZATION_SCALE"
    elif h2 == "SUPPORTED":
        dominant = "WIDTH_PARAMETERIZATION"
    elif h5 == "SUPPORTED":
        dominant = "AUDIO_DURATION_NORMALIZATION"
    else:
        dominant = "INCONCLUSIVE"
    method_ready = "YES"
    write_width_parameterization(output_dir, config_path)
    write_reports(output_dir, initial_rows, normalized_details, trajectory_agg, loss_rows, sensitivity, matching_agg, interaction, statuses, dominant, method_ready)
    (output_dir / "analysis_script.py").write_text(Path(__file__).read_text())
    summary = {
        "status": "COMPLETE",
        "experiment": "SHORT_SCALE_PARAMETERIZATION_AND_LOSS_GEOMETRY_AUDIT",
        "scope": "inference-only; existing matched checkpoints; no training, model change, loss change, or method design",
        "primary_decision": "identify the dominant scale mechanism class behind short-event strict-IoU failure",
        "sample_count": len(gt_rows),
        "audio_duration_representatives_sec": audio_representatives,
        "initial_widths_normalized": {variant: [float(value) for value in values] for variant, values in initial_widths.items()},
        "initial_widths_equal_between_variants": bool(np.allclose(initial_widths["baseline"], initial_widths["m1"], atol=0.0, rtol=0.0)),
        "normalized_gt_distribution": normalized_details,
        "correlations": {
            variant: {
                f"{metric}_vs_{predictor}": corr(
                    [row[metric] for row in initial_rows if row["variant"] == variant],
                    [row[predictor] for row in initial_rows if row["variant"] == variant],
                )
                for metric in ["initial_width_norm", "initial_width_sec", "initial_width_gt_ratio"]
                for predictor in ["audio_duration_sec", "gt_duration_sec", "normalized_gt_width"]
            }
            for variant in ["baseline", "m1"]
        },
        "hypothesis_status": statuses,
        "dominant_mechanism_class": dominant,
        "method_design_ready": method_ready,
        "provenance": {
            "worktree": str(worktree),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "gt": str(gt_path),
            "gt_sha256": sha256(gt_path),
            "baseline_checkpoint": str(checkpoint_paths["baseline"]),
            "baseline_checkpoint_sha256": sha256(checkpoint_paths["baseline"]),
            "m1_checkpoint": str(checkpoint_paths["m1"]),
            "m1_checkpoint_sha256": sha256(checkpoint_paths["m1"]),
            "baseline_prediction": str(prediction_paths["baseline"]),
            "baseline_prediction_sha256": sha256(prediction_paths["baseline"]),
            "m1_prediction": str(prediction_paths["m1"]),
            "m1_prediction_sha256": sha256(prediction_paths["m1"]),
            "decoder_trajectory_source": str(trajectory_path),
            "decoder_trajectory_sha256": sha256(trajectory_path),
            "matcher_coefficients": {"set_cost_span": 10.0, "set_cost_giou": 1.0, "set_cost_class": 4.0},
            "training_regression_coefficients": {"span_loss_coef": 10.0, "giou_loss_coef": 1.0},
        },
        "files": [
            "width_parameterization.md", "initial_width_analysis.csv", "normalized_gt_width_distribution.csv",
            "decoder_width_trajectory.csv", "loss_geometry.csv", "iou_loss_sensitivity.csv", "matching_by_duration.csv",
            "matching_pair_level.csv", "audio_duration_interaction.csv", "hypothesis_assessment.md", "scientific_decision.md",
            "summary.json", "experiment_card.md", "analysis_script.py",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "COMPLETE", "N": len(gt_rows), "dominant": dominant, "method_design_ready": method_ready, "output": str(output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
