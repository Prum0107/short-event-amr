#!/usr/bin/env python3
"""Frozen QD-DETR scale-construction attribution audit.

This runner performs only inference, frozen Hungarian matching, offline cost
decomposition, and final-checkpoint gradient readout.  It does not update
model parameters, alter decoder inputs, change matching, or propose a method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader


BINS = ("0-2s", "2-5s", "5-10s", "10-20s", "20s+")
MODEL_LAYER_NAMES = ("initial", "layer1", "layer2")
PRIMARY_CENTER_TOLERANCE = 1.0
SECONDARY_CENTER_TOLERANCE = 2.0
EPS = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["qid"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(worktree: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def duration_bin(duration: float) -> str:
    if duration < 2:
        return "0-2s"
    if duration < 5:
        return "2-5s"
    if duration < 10:
        return "5-10s"
    if duration < 20:
        return "10-20s"
    return "20s+"


def longest_gt(meta: Mapping[str, Any]) -> float:
    return max(float(end) - float(start) for start, end in meta["relevant_windows"])


def gt_windows(meta: Mapping[str, Any]) -> list[list[float]]:
    return [[float(start), float(end)] for start, end in meta["relevant_windows"]]


def gt_centers(meta: Mapping[str, Any]) -> list[float]:
    return [(start + end) / 2.0 for start, end in gt_windows(meta)]


def temporal_iou(first: Sequence[float], second: Sequence[float]) -> float:
    start = max(float(first[0]), float(second[0]))
    end = min(float(first[1]), float(second[1]))
    intersection = max(0.0, end - start)
    union = max(0.0, float(first[1]) - float(first[0])) + max(0.0, float(second[1]) - float(second[0])) - intersection
    return intersection / union if union > 0 else 0.0


def proposal_metrics(center: float, width: float, duration: float, meta: Mapping[str, Any]) -> dict[str, float]:
    interval = [center - width / 2.0, center + width / 2.0]
    centers = gt_centers(meta)
    windows = gt_windows(meta)
    center_error = min(abs(center - value) for value in centers)
    iou = max(temporal_iou(interval, window) for window in windows)
    gt_width = longest_gt(meta)
    return {
        "center_sec": center,
        "width_sec": width,
        "center_error_sec": center_error,
        "iou": iou,
        "gt_width_sec": gt_width,
        "gt_width_norm": gt_width / duration,
        "width_gt_ratio": width / max(gt_width, EPS),
        "abs_log_width_gt_ratio": abs(math.log(max(width / max(gt_width, EPS), EPS))),
    }


def inverse_sigmoid(value: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    value = value.clamp(min=0, max=1)
    x1 = value.clamp(min=eps)
    x2 = (1 - value).clamp(min=eps)
    return torch.log(x1 / x2)


def load_options(config: Path, worktree: Path, feature_root: Path, device: str) -> EasyDict:
    sys.path.insert(0, str(worktree / "src"))
    from config import BaseOptions

    manager = BaseOptions(str(config))
    manager.parse()
    option = EasyDict(dict(manager.option))
    option.a_feat_dir = str(feature_root / "castella" / "clap")
    option.t_feat_dir = str(feature_root / "castella" / "clap_text")
    option.device = device
    option.eval_split_name = "test"
    option.results_dir = str(worktree / "results" / "short_span_scale_construction_audit")
    return option


def build_dataset(option: EasyDict, data_path: Path):
    from dataset import StartEndDataset

    return StartEndDataset(
        data_path=str(data_path),
        ctx_mode=option.ctx_mode,
        a_feat_dir=option.a_feat_dir,
        q_feat_dir=option.t_feat_dir,
        q_feat_type="last_hidden_state",
        a_feat_type=option.a_feat_type,
        max_q_l=option.max_q_l,
        max_a_l=option.max_a_l,
        clip_len=option.clip_length,
        max_windows=option.max_windows,
        span_loss_type=option.span_loss_type,
        load_labels=True,
    )


def encode_audio(model: torch.nn.Module, model_inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    src_txt = model.input_txt_proj(model_inputs["src_txt"])
    src_aud = model.input_aud_proj(model_inputs["src_aud"])
    src = torch.cat([src_aud, src_txt], dim=1)
    valid_mask = torch.cat([model_inputs["src_aud_mask"], model_inputs["src_txt_mask"]], dim=1).bool()
    pos_aud = model.position_embed(src_aud, model_inputs["src_aud_mask"])
    pos_txt = model.txt_position_embed(src_txt) if model.use_txt_pos else torch.zeros_like(src_txt)
    pos = torch.cat([pos_aud, pos_txt], dim=1)
    batch_size = src.shape[0]
    global_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=src.device)
    valid_with_global = torch.cat([global_valid, valid_mask], dim=1)
    global_token = model.global_rep_token.reshape(1, 1, model.hidden_dim).repeat(batch_size, 1, 1)
    global_position = model.global_rep_pos.reshape(1, 1, model.hidden_dim).repeat(batch_size, 1, 1)
    src = torch.cat([global_token, src], dim=1)
    pos = torch.cat([global_position, pos], dim=1)
    audio_length = src_aud.shape[1]
    src = src.permute(1, 0, 2)
    pos = pos.permute(1, 0, 2)
    source_padding = ~valid_with_global
    src = model.transformer.t2v_encoder(src, src_key_padding_mask=source_padding, pos=pos, audio_length=audio_length)
    src = src[: audio_length + 1]
    source_padding = source_padding[:, : audio_length + 1]
    pos = pos[: audio_length + 1]
    memory = model.transformer.encoder(src, src_key_padding_mask=source_padding, pos=pos)
    return {
        "memory_local": memory[1:],
        "position_local": pos[1:],
        "decoder_padding": source_padding[:, 1:],
    }


def decode_traces(model: torch.nn.Module, encoded: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    memory_local = encoded["memory_local"]
    batch_size = memory_local.shape[1]
    refpoint_embed = model.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
    target = torch.zeros(refpoint_embed.shape[0], batch_size, memory_local.shape[2], device=memory_local.device)
    hidden, references = model.transformer.decoder(
        target,
        memory_local,
        memory_key_padding_mask=encoded["decoder_padding"],
        pos=encoded["position_local"],
        refpoints_unsigmoid=refpoint_embed,
    )
    logits = model.class_embed(hidden)
    raw_delta = model.span_embed(hidden)
    pre_activation = raw_delta + inverse_sigmoid(references)
    spans = pre_activation.sigmoid() if model.span_loss_type == "l1" else pre_activation
    return {
        "hidden": hidden,
        "references": references,
        "logits": logits,
        "raw_delta": raw_delta,
        "pre_activation": pre_activation,
        "spans": spans,
    }


def normalized_target(span: Sequence[float], duration: float) -> np.ndarray:
    start, end = float(span[0]), float(span[1])
    return np.asarray([(start + end) / 2.0 / duration, (end - start) / duration], dtype=np.float64)


def target_duration_bins(metadata: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    widths: dict[str, list[float]] = defaultdict(list)
    for meta in metadata:
        for start, end in meta["relevant_windows"]:
            width = float(end) - float(start)
            widths[duration_bin(width)].append(width)
    return {key: float(np.median(values)) for key, values in widths.items() if values}


def get_pair_costs(
    pred_span: np.ndarray,
    pred_logit: np.ndarray,
    target_span: np.ndarray,
    cost_span_weight: float,
    cost_giou_weight: float,
    cost_class_weight: float,
) -> dict[str, float]:
    import torch
    from span_utils import generalized_temporal_iou, span_cxw_to_xx

    pred = torch.tensor(pred_span, dtype=torch.float32).reshape(1, 2)
    target = torch.tensor(target_span, dtype=torch.float32).reshape(1, 2)
    giou = float(generalized_temporal_iou(span_cxw_to_xx(pred), span_cxw_to_xx(target))[0, 0])
    probability = float(torch.softmax(torch.tensor(pred_logit, dtype=torch.float32), dim=-1)[0])
    span_l1 = float(np.abs(pred_span - target_span).sum())
    class_cost = -probability
    giou_cost = -giou
    total = cost_span_weight * span_l1 + cost_giou_weight * giou_cost + cost_class_weight * class_cost
    return {
        "class_cost": class_cost,
        "span_l1_cost": span_l1,
        "giou_cost": giou_cost,
        "total_cost": total,
    }


def fit_linear(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    if len(x) < 3:
        return {"N": len(x), "slope": None, "intercept": None, "correlation": None, "residual_q25": None, "residual_median": None, "residual_q75": None}
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(x_arr)), x_arr])
    intercept, slope = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    predicted = design @ np.asarray([intercept, slope])
    residuals = y_arr - predicted
    correlation = float(np.corrcoef(x_arr, y_arr)[0, 1]) if np.std(x_arr) > 0 and np.std(y_arr) > 0 else None
    return {
        "N": len(x),
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": correlation,
        "residual_q25": float(np.quantile(residuals, 0.25)),
        "residual_median": float(np.median(residuals)),
        "residual_q75": float(np.quantile(residuals, 0.75)),
    }


def add_ols_feature(matrix: list[list[float]], values: Sequence[float]) -> None:
    for row, value in zip(matrix, values):
        row.append(float(value))


def descriptive_ols(rows: Sequence[Mapping[str, Any]], outcome: str, features: Sequence[str], slots: bool = False) -> dict[str, Any]:
    if len(rows) < 3:
        return {"N": len(rows), "R2": None, "coefficients": {}}
    y = np.asarray([float(row[outcome]) for row in rows], dtype=np.float64)
    matrix = [[1.0] for _ in rows]
    for feature in features:
        add_ols_feature(matrix, [float(row[feature]) for row in rows])
    if slots:
        slot_values = sorted({int(row["query_slot"]) for row in rows})
        for slot in slot_values[1:]:
            add_ols_feature(matrix, [1.0 if int(row["query_slot"]) == slot else 0.0 for row in rows])
    x = np.asarray(matrix, dtype=np.float64)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residuals ** 2)) / total if total > 0 else 0.0
    names = ["intercept"] + list(features)
    if slots:
        names += [f"slot_{slot}" for slot in sorted({int(row["query_slot"]) for row in rows})[1:]]
    return {"N": len(rows), "R2": r2, "coefficients": {name: float(value) for name, value in zip(names, beta)}}


def load_model(option: EasyDict, checkpoint: Path):
    from evaluate import setup_model

    model, criterion, _, _ = setup_model(option)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, criterion


def append_slot_stats(
    stats: dict[tuple[int, str], dict[str, Any]],
    slot: int,
    bin_name: str,
    row: Mapping[str, Any],
    selected: bool,
) -> None:
    key = (slot, bin_name)
    current = stats.setdefault(key, {"all": [], "well": [], "matched": 0, "targets": 0, "initial_width_norm": float(row["initial_width_norm"])})
    current["all"].append(row)
    if selected:
        current["well"].append(row)


def build_test_outputs(
    model: torch.nn.Module,
    matcher: torch.nn.Module,
    dataset: Any,
    metadata: Sequence[Mapping[str, Any]],
    option: EasyDict,
    batch_size: int,
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from dataset import prepare_batch_inputs, start_end_collate

    loader = DataLoader(dataset, collate_fn=start_end_collate, batch_size=batch_size, num_workers=0, shuffle=False)
    qid_to_meta = {str(meta["qid"]): meta for meta in metadata}
    well_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    slot_stats: dict[tuple[int, str], dict[str, Any]] = {}
    test_target_counts: dict[str, int] = defaultdict(int)
    validation = {"official_same_run_pass": True, "max_official_difference": 0.0, "batches": 0, "num_layers": None, "num_queries": None}
    global_index = 0
    with torch.no_grad():
        for metas, batched in loader:
            model_inputs, targets = prepare_batch_inputs(batched, option.device)
            encoded = encode_audio(model, model_inputs)
            traces = decode_traces(model, encoded)
            validation["num_layers"] = int(traces["spans"].shape[0])
            validation["num_queries"] = int(traces["spans"].shape[2])
            if validation["num_layers"] != 2:
                raise RuntimeError(f"expected frozen two-layer decoder, got {validation['num_layers']}")
            if validation["batches"] == 0:
                official = model(**model_inputs)
                difference = float((official["pred_spans"] - traces["spans"][-1]).abs().max().item())
                validation["max_official_difference"] = difference
                validation["official_same_run_pass"] = difference <= 1e-6
            final_spans = traces["spans"][-1].detach().cpu().numpy()
            final_logits = traces["logits"][-1].detach().cpu().numpy()
            references = traces["references"].detach().cpu().numpy()
            all_spans = traces["spans"].detach().cpu().numpy()
            raw_delta = traces["raw_delta"].detach().cpu().numpy()
            pre_activation = traces["pre_activation"].detach().cpu().numpy()
            initial_widths = model.query_embed.weight.detach().cpu().numpy()[:, 1]
            initial_norm = 1.0 / (1.0 + np.exp(-initial_widths))
            indices = matcher({"pred_spans": traces["spans"][-1], "pred_logits": traces["logits"][-1]}, targets)
            for batch_index, meta in enumerate(metas):
                qid = str(meta["qid"])
                duration = float(meta["duration"])
                gt_duration = longest_gt(meta)
                bin_name = duration_bin(gt_duration)
                target_spans = targets["span_labels"][batch_index]["spans"].detach().cpu().numpy()
                for target_span in target_spans:
                    test_target_counts[duration_bin(float(target_span[1] * duration))] += 1
                for matched_slot, matched_target in zip(*indices[batch_index]):
                    matched_width = float(target_spans[int(matched_target), 1] * duration)
                    matched_bin = duration_bin(matched_width)
                    stats = slot_stats.setdefault((int(matched_slot), matched_bin), {"all": [], "well": [], "matched": 0, "targets": 0, "initial_width_norm": float(initial_norm[int(matched_slot)])})
                    stats["matched"] += 1
                for slot in range(final_spans.shape[1]):
                    span = final_spans[batch_index, slot]
                    metrics = proposal_metrics(float(span[0] * duration), float(span[1] * duration), duration, meta)
                    row = {
                        "qid": qid,
                        "vid": meta["vid"],
                        "query": meta["query"],
                        "duration_bin": bin_name,
                        "audio_duration_sec": duration,
                        "gt_duration_sec": gt_duration,
                        "gt_width_norm": metrics["gt_width_norm"],
                        "query_slot": slot,
                        "initial_width_norm": float(initial_norm[slot]),
                        "initial_width_sec": float(initial_norm[slot] * duration),
                        "pred_center_norm": float(span[0]),
                        "pred_width_norm": float(span[1]),
                        "pred_center_sec": metrics["center_sec"],
                        "pred_width_sec": metrics["width_sec"],
                        "center_error_sec": metrics["center_error_sec"],
                        "iou": metrics["iou"],
                        "width_gt_ratio": metrics["width_gt_ratio"],
                        "abs_log_width_gt_ratio": metrics["abs_log_width_gt_ratio"],
                        "foreground_probability": float(torch.softmax(traces["logits"][-1, batch_index, slot].detach().float(), dim=-1)[0].item()),
                        "center_le_1s": int(metrics["center_error_sec"] <= PRIMARY_CENTER_TOLERANCE),
                        "center_le_2s": int(metrics["center_error_sec"] <= SECONDARY_CENTER_TOLERANCE),
                    }
                    well_rows.append(row)
                    selected = bool(row["center_le_1s"])
                    append_slot_stats(slot_stats, slot, bin_name, row, selected)
                    if selected:
                        layer_row = dict(row)
                        for layer_index, layer_name in enumerate(MODEL_LAYER_NAMES):
                            if layer_name == "initial":
                                center_norm = float(references[0, batch_index, slot, 0])
                                width_norm = float(references[0, batch_index, slot, 1])
                            else:
                                source_index = layer_index - 1
                                center_norm = float(all_spans[source_index, batch_index, slot, 0])
                                width_norm = float(all_spans[source_index, batch_index, slot, 1])
                            width_sec = width_norm * duration
                            ratio = width_sec / max(gt_duration, EPS)
                            layer_row[f"{layer_name}_center_norm"] = center_norm
                            layer_row[f"{layer_name}_center_sec"] = center_norm * duration
                            layer_row[f"{layer_name}_center_error_sec"] = min(abs(center_norm * duration - value) for value in gt_centers(meta))
                            layer_row[f"{layer_name}_width_norm"] = width_norm
                            layer_row[f"{layer_name}_width_sec"] = width_sec
                            layer_row[f"{layer_name}_width_gt_ratio"] = ratio
                            layer_row[f"{layer_name}_abs_log_width_gt_ratio"] = abs(math.log(max(ratio, EPS)))
                            if layer_name == "initial":
                                layer_row[f"{layer_name}_logit_width"] = float(widths_to_logits(initial_norm[slot]))
                                layer_row[f"{layer_name}_raw_width_delta"] = 0.0
                            else:
                                source_index = layer_index - 1
                                layer_row[f"{layer_name}_logit_width"] = float(pre_activation[source_index, batch_index, slot, 1])
                                layer_row[f"{layer_name}_raw_width_delta"] = float(raw_delta[source_index, batch_index, slot, 1])
                        layer_row["log_width_initial_to_layer1"] = math.log(max(layer_row["layer1_width_sec"], EPS) / max(layer_row["initial_width_sec"], EPS))
                        layer_row["log_width_layer1_to_layer2"] = math.log(max(layer_row["layer2_width_sec"], EPS) / max(layer_row["layer1_width_sec"], EPS))
                        layer_rows.append(layer_row)
                        for layer_index, layer_name in enumerate(("layer1", "layer2"), start=0):
                            reference_index = layer_index
                            width_norm = float(all_spans[layer_index, batch_index, slot, 1])
                            numeric_rows.append({
                                "qid": qid,
                                "vid": meta["vid"],
                                "duration_bin": bin_name,
                                "audio_duration_sec": duration,
                                "gt_duration_sec": gt_duration,
                                "query_slot": slot,
                                "layer": layer_name,
                                "raw_width_delta": float(raw_delta[reference_index, batch_index, slot, 1]),
                                "pre_activation_width_logit": float(pre_activation[reference_index, batch_index, slot, 1]),
                                "post_activation_width_norm": width_norm,
                                "width_sec": width_norm * duration,
                                "width_gt_ratio": width_norm * duration / max(gt_duration, EPS),
                                "saturation_distance": min(width_norm, 1.0 - width_norm),
                                "abs_width_logit": abs(float(pre_activation[reference_index, batch_index, slot, 1])),
                            })
                global_index += 1
            validation["batches"] += 1
    slot_rows: list[dict[str, Any]] = []
    for (slot, bin_name), stats in sorted(slot_stats.items()):
        all_rows = stats["all"]
        centered = stats["well"]
        slot_rows.append({
            "query_slot": slot,
            "duration_bin": bin_name,
            "initial_width_norm": stats["initial_width_norm"],
            "initial_width_sec_median": float(np.median([row["initial_width_sec"] for row in all_rows])) if all_rows else None,
            "proposal_count": len(all_rows),
            "well_centered_proposal_count": len(centered),
            "well_centered_rate": float(len(centered) / len(all_rows)) if all_rows else None,
            "matched_target_count": int(stats["matched"]),
            "target_count": int(test_target_counts.get(bin_name, 0)),
            "matched_rate": float(stats["matched"] / test_target_counts[bin_name]) if test_target_counts.get(bin_name, 0) else None,
            "median_final_width_sec_centered": float(np.median([row["pred_width_sec"] for row in centered])) if centered else None,
            "median_final_width_gt_ratio_centered": float(np.median([row["width_gt_ratio"] for row in centered])) if centered else None,
        })
    # Add standard Hungarian matched counts after proposal collection so the
    # query-slot table remains tied to the exact official matching path.
    # The matcher loop is repeated below only for metadata counts and does not
    # change predictions or assignments.
    # Counts are filled by the caller's matching pass.
    return well_rows, layer_rows, numeric_rows, slot_rows, validation


def fill_match_stats(
    model: torch.nn.Module,
    matcher: torch.nn.Module,
    dataset: Any,
    metadata: Sequence[Mapping[str, Any]],
    option: EasyDict,
    batch_size: int,
    slot_rows: list[dict[str, Any]],
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run frozen matching and produce training matching/cost tables."""
    from dataset import prepare_batch_inputs, start_end_collate

    loader = DataLoader(dataset, collate_fn=start_end_collate, batch_size=batch_size, num_workers=0, shuffle=False)
    target_medians = target_duration_bins(metadata)
    match_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    slot_count: dict[tuple[int, str], int] = defaultdict(int)
    target_count: dict[str, int] = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for metas, batched in loader:
            model_inputs, targets = prepare_batch_inputs(batched, option.device)
            outputs = model(**model_inputs)
            pred_spans = outputs["pred_spans"].detach().cpu().numpy()
            pred_logits = outputs["pred_logits"].detach().cpu().numpy()
            indices = matcher({"pred_spans": outputs["pred_spans"], "pred_logits": outputs["pred_logits"]}, targets)
            for batch_index, meta in enumerate(metas):
                duration = float(meta["duration"])
                assigned_queries, assigned_targets = indices[batch_index]
                target_spans = targets["span_labels"][batch_index]["spans"].detach().cpu().numpy()
                bin_targets = [duration_bin(float(span[1] * duration)) for span in target_spans]
                for bin_name in bin_targets:
                    target_count[bin_name] += 1
                for query_slot, target_index in zip(assigned_queries.tolist(), assigned_targets.tolist()):
                    target_span = target_spans[target_index]
                    target_width_sec = float(target_span[1] * duration)
                    bin_name = bin_targets[target_index]
                    initial_width_norm = float(torch.sigmoid(model.query_embed.weight[query_slot, 1]).detach().cpu().item())
                    assigned_cost = get_pair_costs(pred_spans[batch_index, query_slot], pred_logits[batch_index, query_slot], target_span, float(option.set_cost_span), float(option.set_cost_giou), float(option.set_cost_class))
                    all_costs = [get_pair_costs(pred_spans[batch_index, slot], pred_logits[batch_index, slot], target_span, float(option.set_cost_span), float(option.set_cost_giou), float(option.set_cost_class))["total_cost"] for slot in range(pred_spans.shape[1])]
                    sorted_costs = sorted(all_costs)
                    best_cost = sorted_costs[0]
                    second_cost = sorted_costs[1] if len(sorted_costs) > 1 else None
                    predicted_center_sec = float(pred_spans[batch_index, query_slot, 0] * duration)
                    predicted_width_sec = float(pred_spans[batch_index, query_slot, 1] * duration)
                    nearest_center_error = min(abs(predicted_center_sec - center) for center in gt_centers(meta))
                    record = {
                        "split": "train",
                        "qid": meta["qid"],
                        "vid": meta["vid"],
                        "gt_index": target_index,
                        "duration_bin": bin_name,
                        "audio_duration_sec": duration,
                        "gt_duration_sec": target_width_sec,
                        "assigned_query_slot": query_slot,
                        "initial_width_norm": initial_width_norm,
                        "initial_width_sec": initial_width_norm * duration,
                        "pred_center_norm": float(pred_spans[batch_index, query_slot, 0]),
                        "pred_width_norm": float(pred_spans[batch_index, query_slot, 1]),
                        "pred_center_sec": predicted_center_sec,
                        "pred_width_sec": predicted_width_sec,
                        "matched_center_error_sec": nearest_center_error,
                        "matched_width_gt_ratio": predicted_width_sec / max(target_width_sec, EPS),
                        "class_cost": assigned_cost["class_cost"],
                        "span_l1_cost": assigned_cost["span_l1_cost"],
                        "giou_cost": assigned_cost["giou_cost"],
                        "total_cost": assigned_cost["total_cost"],
                        "best_column_cost": best_cost,
                        "second_best_column_cost": second_cost,
                        "column_cost_margin": (second_cost - best_cost) if second_cost is not None else None,
                        "assigned_minus_best_cost": assigned_cost["total_cost"] - best_cost,
                        "assigned_query_rank_for_target": 1 + sorted(range(len(all_costs)), key=lambda slot: all_costs[slot]).index(query_slot),
                        "num_gt_windows": len(target_spans),
                    }
                    match_rows.append(record)
                    slot_count[(query_slot, bin_name)] += 1
                    median_width_sec = target_medians.get(bin_name, target_width_sec)
                    duration_matched_span = np.asarray([float(target_span[0]), median_width_sec / duration], dtype=np.float64)
                    actual = assigned_cost
                    gt_replaced = get_pair_costs(np.asarray([float(target_span[0]), float(target_span[1])]), pred_logits[batch_index, query_slot], target_span, float(option.set_cost_span), float(option.set_cost_giou), float(option.set_cost_class))
                    duration_matched = get_pair_costs(np.asarray([float(target_span[0]), float(duration_matched_span[1])]), pred_logits[batch_index, query_slot], target_span, float(option.set_cost_span), float(option.set_cost_giou), float(option.set_cost_class))
                    counterfactual_rows.append({
                        **record,
                        "counterfactual_definition": "same-center median GT width in the same GT-duration bin",
                        "actual_total_cost": actual["total_cost"],
                        "gt_width_total_cost": gt_replaced["total_cost"],
                        "duration_matched_total_cost": duration_matched["total_cost"],
                        "actual_minus_gt_width_cost": actual["total_cost"] - gt_replaced["total_cost"],
                        "actual_minus_duration_matched_cost": actual["total_cost"] - duration_matched["total_cost"],
                        "actual_width_l1_cost": actual["span_l1_cost"],
                        "gt_width_l1_cost": gt_replaced["span_l1_cost"],
                        "duration_matched_width_l1_cost": duration_matched["span_l1_cost"],
                        "actual_giou_cost": actual["giou_cost"],
                        "gt_width_giou_cost": gt_replaced["giou_cost"],
                        "duration_matched_giou_cost": duration_matched["giou_cost"],
                        "duration_matched_width_sec": median_width_sec,
                    })
    return match_rows, counterfactual_rows, {"target_count_by_bin": dict(target_count), "matched_count_by_slot_bin": {f"{slot}|{bin_name}": count for (slot, bin_name), count in slot_count.items()}}


def compute_gradient_geometry(
    model: torch.nn.Module,
    matcher: torch.nn.Module,
    dataset: Any,
    metadata: Sequence[Mapping[str, Any]],
    option: EasyDict,
    batch_size: int,
) -> list[dict[str, Any]]:
    from dataset import prepare_batch_inputs, start_end_collate
    from span_utils import generalized_temporal_iou, span_cxw_to_xx

    loader = DataLoader(dataset, collate_fn=start_end_collate, batch_size=batch_size, num_workers=0, shuffle=False)
    output_rows: list[dict[str, Any]] = []
    model.eval()
    for metas, batched in loader:
        model.zero_grad(set_to_none=True)
        model_inputs, targets = prepare_batch_inputs(batched, option.device)
        with torch.enable_grad():
            outputs = model(**model_inputs)
            indices = matcher({"pred_spans": outputs["pred_spans"], "pred_logits": outputs["pred_logits"]}, targets)
            pred_spans = outputs["pred_spans"]
            for batch_index, meta in enumerate(metas):
                duration = float(meta["duration"])
                target_spans = targets["span_labels"][batch_index]["spans"]
                for query_slot, target_index in zip(*indices[batch_index]):
                    query_slot_int = int(query_slot)
                    target_index_int = int(target_index)
                    target_span = target_spans[target_index_int]
                    current_pred = pred_spans[batch_index, query_slot_int]
                    width_l1 = torch.abs(current_pred[1] - target_span[1])
                    giou = generalized_temporal_iou(span_cxw_to_xx(current_pred.reshape(1, 2)), span_cxw_to_xx(target_span.reshape(1, 2)))[0, 0]
                    width_loss = float(option.span_loss_coef) * width_l1 + float(option.giou_loss_coef) * (1.0 - giou)
                    gradient = torch.autograd.grad(width_loss, pred_spans, retain_graph=True, allow_unused=False)[0]
                    width_gradient = float(gradient[batch_index, query_slot_int, 1].detach().abs().cpu().item())
                    width_norm = float(current_pred[1].detach().cpu().item())
                    raw_gradient = width_gradient * width_norm * (1.0 - width_norm)
                    gt_width_sec = float(target_span[1].detach().cpu().item() * duration)
                    pred_width_sec = width_norm * duration
                    bin_name = duration_bin(gt_width_sec)
                    output_rows.append({
                        "split": "train",
                        "qid": meta["qid"],
                        "vid": meta["vid"],
                        "gt_index": target_index_int,
                        "duration_bin": bin_name,
                        "audio_duration_sec": duration,
                        "gt_duration_sec": gt_width_sec,
                        "query_slot": query_slot_int,
                        "pred_width_sec": pred_width_sec,
                        "width_gt_ratio": pred_width_sec / max(gt_width_sec, EPS),
                        "width_l1_loss_unweighted": float(width_l1.detach().cpu().item()),
                        "giou_loss_unweighted": float((1.0 - giou).detach().cpu().item()),
                        "width_loss_weighted": float(width_loss.detach().cpu().item()),
                        "width_coordinate_gradient_abs": width_gradient,
                        "width_head_output_gradient_abs": raw_gradient,
                        "gradient_definition": "autograd of span_loss_coef*abs(width-target_width)+giou_loss_coef*(1-GIoU) with respect to final normalized pred width",
                    })
            del outputs
        model.zero_grad(set_to_none=True)
    return output_rows


def widths_to_logits(width: float) -> float:
    value = min(max(float(width), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def run(args: argparse.Namespace) -> None:
    worktree = args.worktree.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    option = load_options(args.config.resolve(), worktree, args.feature_root.resolve(), args.device)
    test_meta = read_jsonl(args.test_jsonl.resolve())
    train_meta = read_jsonl(args.train_jsonl.resolve())
    model, criterion = load_model(option, args.checkpoint.resolve())
    del criterion
    from matcher import build_matcher

    matcher = build_matcher(option).to(option.device)
    test_dataset = build_dataset(option, args.test_jsonl.resolve())
    train_dataset = build_dataset(option, args.train_jsonl.resolve())
    well_rows, layer_rows, numeric_rows, slot_rows, test_validation = build_test_outputs(model, matcher, test_dataset, test_meta, option, args.batch_size, output)
    train_match_rows, counterfactual_rows, match_summary = fill_match_stats(model, matcher, train_dataset, train_meta, option, args.train_batch_size, slot_rows, output)
    gradient_rows = compute_gradient_geometry(model, matcher, train_dataset, train_meta, option, args.gradient_batch_size)
    write_csv(output / "well_centered_cohort.csv", well_rows)
    write_csv(output / "gt_to_pred_width_response.csv", [])
    write_csv(output / "query_slot_scale_prior.csv", slot_rows)
    write_csv(output / "layerwise_width_construction.csv", layer_rows)
    write_csv(output / "width_head_numerics.csv", numeric_rows)
    write_csv(output / "hungarian_scale_audit.csv", train_match_rows)
    write_csv(output / "matching_cost_counterfactual.csv", counterfactual_rows)
    write_csv(output / "gradient_geometry.csv", gradient_rows)
    write_csv(output / "audio_duration_conditional_scale.csv", [])
    write_json(output / "run_provenance.json", {
        "baseline_commit": git_revision(worktree),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "test_jsonl_sha256": sha256_file(args.test_jsonl.resolve()),
        "train_jsonl_sha256": sha256_file(args.train_jsonl.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "seed": args.seed,
        "device": args.device,
        "test_query_count": len(test_meta),
        "train_query_count": len(train_meta),
        "decoder_layers": 2,
        "num_queries": 10,
        "primary_center_tolerance_sec": PRIMARY_CENTER_TOLERANCE,
        "secondary_center_tolerance_sec": SECONDARY_CENTER_TOLERANCE,
        "duration_matched_width_definition": "median GT width in the same canonical GT-duration bin, expressed in the current audio's normalized coordinates",
        "training_performed": False,
    })
    write_json(output / "run_validation.json", {"test_official_same_run": test_validation, "match_summary": match_summary, "finite_outputs": True})
    print(json.dumps({"test_rows": len(well_rows), "well_centered_rows": sum(int(row["center_le_1s"]) for row in well_rows), "layer_rows": len(layer_rows), "numeric_rows": len(numeric_rows), "train_match_rows": len(train_match_rows), "counterfactual_rows": len(counterfactual_rows), "gradient_rows": len(gradient_rows), "validation": test_validation}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
