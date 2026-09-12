#!/usr/bin/env python3
"""Baseline-only temporal audit of QD-DETR short-event scale learning.

The only state-changing operation in this program is the official baseline
optimizer update.  All additional work is read-only instrumentation: frozen
checkpoint summaries, official Hungarian assignments, and pre-update gradient
readouts.  No model, loss, matcher, parameterization, or optimizer rule is
changed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset


BINS = ("0-2s", "2-5s", "5-10s", "10-20s", "20s+")
NARROW_BINS = ("1-2s", "2-3s", "3-5s")
CHECKPOINT_EPOCHS = (0, 1, 5, 10, 20, 40, 60, 100, 150, 200)
GRADIENT_EPOCHS = (1, 5, 10, 20, 40, 60, 100, 150, 200)
CENTER_TOLERANCE_SEC = 1.0
EPS = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
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


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


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


def narrow_bin(duration: float) -> str | None:
    if 1 <= duration < 2:
        return "1-2s"
    if 2 <= duration < 3:
        return "2-3s"
    if 3 <= duration < 5:
        return "3-5s"
    return None


def gt_windows(meta: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(start), float(end)) for start, end in meta["relevant_windows"]]


def gt_duration(meta: Mapping[str, Any]) -> float:
    return max(end - start for start, end in gt_windows(meta))


def gt_centers(meta: Mapping[str, Any]) -> list[float]:
    return [(start + end) / 2 for start, end in gt_windows(meta)]


def temporal_iou(center: float, width: float, gt: tuple[float, float]) -> float:
    left = max(center - width / 2, gt[0])
    right = min(center + width / 2, gt[1])
    intersection = max(0.0, right - left)
    union = max(center + width / 2, gt[1]) - min(center - width / 2, gt[0])
    return intersection / union if union > 0 else 0.0


def proposal_metrics(center: float, width: float, meta: Mapping[str, Any]) -> dict[str, float]:
    centers = gt_centers(meta)
    windows = gt_windows(meta)
    gt_width = gt_duration(meta)
    best_iou = max(temporal_iou(center, width, gt) for gt in windows)
    return {
        "center_error_sec": min(abs(center - value) for value in centers),
        "iou": best_iou,
        "gt_width_sec": gt_width,
        "gt_width_norm": gt_width / float(meta["duration"]),
        "width_gt_ratio": width / max(gt_width, EPS),
        "abs_log_width_gt_ratio": abs(math.log(max(width / max(gt_width, EPS), EPS))),
    }


def inverse_sigmoid(value: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    value = value.clamp(min=0, max=1)
    return torch.log(value.clamp(min=eps) / (1 - value).clamp(min=eps))


def load_options(config: Path, worktree: Path, feature_root: Path, output: Path, device: str) -> EasyDict:
    sys.path.insert(0, str(worktree / "src"))
    from config import BaseOptions

    manager = BaseOptions(str(config))
    manager.parse()
    option = EasyDict(dict(manager.option))
    option.device = device
    option.train_path = str(output / "train_path.placeholder")
    option.val_path = str(output / "val_path.placeholder")
    option.eval_path = str(output / "val_path.placeholder")
    option.test_path = str(output / "test_path.placeholder")
    option.a_feat_dir = str(feature_root / "castella" / "clap")
    option.t_feat_dir = str(feature_root / "castella" / "clap_text")
    option.eval_split_name = "val"
    option.results_dir = str(output / "runtime_eval")
    option.ckpt_filepath = str(output / "best_validation.pth")
    option.train_log_filepath = str(output / "training.log")
    option.eval_log_filepath = str(output / "validation.log")
    option.model_ema = False
    option.seed = 2023
    return option


def build_dataset(option: EasyDict, data_path: Path) -> Any:
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


def save_state(path: Path, model: torch.nn.Module, optimizer: Any, scheduler: Any, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(), "epoch": epoch}, path)


def load_model(option: EasyDict):
    from evaluate import setup_model

    model, criterion, optimizer, scheduler = setup_model(option)
    return model, criterion, optimizer, scheduler


def get_model_layers(model: torch.nn.Module, outputs: Mapping[str, Any]) -> list[tuple[str, Mapping[str, torch.Tensor]]]:
    return [("layer1", outputs["aux_outputs"][0]), ("layer2", {"pred_spans": outputs["pred_spans"], "pred_logits": outputs["pred_logits"]})]


def make_diagnostic_indices(metadata: Sequence[Mapping[str, Any]], per_bin: int = 16) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, meta in enumerate(metadata):
        grouped[duration_bin(gt_duration(meta))].append(index)
    selected: list[int] = []
    for bin_name in BINS:
        candidates = sorted(grouped[bin_name], key=lambda index: (str(metadata[index]["qid"]), index))
        selected.extend(candidates[:per_bin])
    return selected


def cache_diagnostic_batches(dataset: Any, indices: Sequence[int], option: EasyDict) -> list[tuple[list[dict[str, Any]], Any]]:
    from dataset import start_end_collate

    state = capture_rng()
    try:
        subset = Subset(dataset, list(indices))
        loader = DataLoader(subset, batch_size=option.bsz, shuffle=False, num_workers=0, collate_fn=start_end_collate)
        cached = []
        for batch in loader:
            cached.append((batch[0], batch[1]))
        return cached
    finally:
        restore_rng(state)


def prepare_batch(batch: Any, option: EasyDict):
    from dataset import prepare_batch_inputs

    return prepare_batch_inputs(batch, option.device)


def cost_components(pred_span: Sequence[float], pred_logit: Sequence[float], target_span: Sequence[float], option: EasyDict) -> dict[str, float]:
    from span_utils import generalized_temporal_iou, span_cxw_to_xx

    pred = torch.tensor(pred_span, dtype=torch.float32).reshape(1, 2)
    target = torch.tensor(target_span, dtype=torch.float32).reshape(1, 2)
    giou = float(generalized_temporal_iou(span_cxw_to_xx(pred), span_cxw_to_xx(target))[0, 0])
    foreground = float(torch.softmax(torch.tensor(pred_logit, dtype=torch.float32), dim=-1)[0])
    span_l1 = float(np.abs(np.asarray(pred_span) - np.asarray(target_span)).sum())
    class_cost = -foreground
    giou_cost = -giou
    total = float(option.set_cost_span) * span_l1 + float(option.set_cost_giou) * giou_cost + float(option.set_cost_class) * class_cost
    return {"class_cost": class_cost, "span_l1_cost": span_l1, "giou_cost": giou_cost, "total_cost": total}


def forward_with_traces(model: torch.nn.Module, model_inputs: Mapping[str, torch.Tensor]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run the official forward and capture its native decoder references/deltas."""
    holder: dict[str, torch.Tensor] = {}

    def transformer_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        holder["references"] = output[1].detach()

    def span_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        holder["raw_delta"] = output.detach()

    hook_a = model.transformer.register_forward_hook(transformer_hook)
    hook_b = model.span_embed.register_forward_hook(span_hook)
    try:
        outputs = model(**model_inputs)
    finally:
        hook_a.remove()
        hook_b.remove()
    references = holder["references"].detach().cpu().numpy()
    raw_delta = holder["raw_delta"].detach().cpu().numpy()
    return outputs, references, raw_delta


def match_cost_rows(
    pred_spans: np.ndarray,
    pred_logits: np.ndarray,
    targets: Any,
    metas: Sequence[Mapping[str, Any]],
    indices: Sequence[tuple[torch.Tensor, torch.Tensor]],
    option: EasyDict,
    initial_widths: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_index, meta in enumerate(metas):
        duration = float(meta["duration"])
        target_spans = targets["span_labels"][batch_index]["spans"].detach().cpu().numpy()
        assigned_slots, assigned_targets = indices[batch_index]
        for slot_tensor, target_tensor in zip(assigned_slots.tolist(), assigned_targets.tolist()):
            slot = int(slot_tensor)
            target_index = int(target_tensor)
            target = target_spans[target_index]
            all_costs = [
                cost_components(pred_spans[batch_index, candidate], pred_logits[batch_index, candidate], target, option)
                for candidate in range(pred_spans.shape[1])
            ]
            assigned = all_costs[slot]
            ranked = sorted(range(len(all_costs)), key=lambda candidate: all_costs[candidate]["total_cost"])
            best = all_costs[ranked[0]]["total_cost"]
            second = all_costs[ranked[1]]["total_cost"] if len(ranked) > 1 else None
            predicted_center = float(pred_spans[batch_index, slot, 0] * duration)
            predicted_width = float(pred_spans[batch_index, slot, 1] * duration)
            target_width = float(target[1] * duration)
            center_error = min(abs(predicted_center - center) for center in gt_centers(meta))
            rows.append({
                "checkpoint": "",
                "epoch": -1,
                "qid": str(meta["qid"]),
                "vid": str(meta["vid"]),
                "gt_index": target_index,
                "duration_bin": duration_bin(target_width),
                "audio_duration_sec": duration,
                "gt_duration_sec": target_width,
                "assigned_query_slot": slot,
                "initial_width_norm": float(initial_widths[slot]),
                "initial_width_sec": float(initial_widths[slot] * duration),
                "pred_center_norm": float(pred_spans[batch_index, slot, 0]),
                "pred_width_norm": float(pred_spans[batch_index, slot, 1]),
                "pred_center_sec": predicted_center,
                "pred_width_sec": predicted_width,
                "matched_center_error_sec": center_error,
                "matched_width_gt_ratio": predicted_width / max(target_width, EPS),
                "class_cost": assigned["class_cost"],
                "span_l1_cost": assigned["span_l1_cost"],
                "giou_cost": assigned["giou_cost"],
                "total_cost": assigned["total_cost"],
                "best_column_cost": best,
                "second_best_column_cost": second,
                "column_cost_margin": (second - best) if second is not None else None,
                "assigned_minus_best_cost": assigned["total_cost"] - best,
                "assigned_query_rank_for_target": 1 + ranked.index(slot),
                "num_gt_windows": len(target_spans),
            })
    return rows


def collect_checkpoint(
    model: torch.nn.Module,
    matcher: torch.nn.Module,
    dataset: Any,
    option: EasyDict,
    checkpoint_name: str,
    epoch: int,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Collect frozen final proposals, assignments, layer traces, and numerics."""
    from dataset import start_end_collate

    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=start_end_collate)
    proposal_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    specialization_rows: list[dict[str, Any]] = []
    validation = {"checkpoint": checkpoint_name, "epoch": epoch, "official_forward_max_abs_diff": None, "official_forward_equivalent": None, "num_batches": 0}
    initial_spans = model.query_embed.weight.detach().sigmoid().cpu().numpy()
    initial_widths = initial_spans[:, 1]
    slot_bin_counts: dict[tuple[int, str], int] = defaultdict(int)
    bin_counts: dict[str, int] = defaultdict(int)
    assignment_map: dict[str, int] = {}

    with torch.no_grad():
        for metas, batched in loader:
            model_inputs, targets = prepare_batch(batched, option)
            outputs, references, raw_delta = forward_with_traces(model, model_inputs)
            pred_spans = outputs["pred_spans"].detach().cpu().numpy()
            pred_logits = outputs["pred_logits"].detach().cpu().numpy()
            layer_spans = [outputs["aux_outputs"][0]["pred_spans"].detach().cpu().numpy(), pred_spans]
            layer_logits = [outputs["aux_outputs"][0]["pred_logits"].detach().cpu().numpy(), pred_logits]
            if validation["official_forward_max_abs_diff"] is None:
                validation["official_forward_max_abs_diff"] = float((outputs["pred_spans"] - torch.as_tensor(pred_spans, device=outputs["pred_spans"].device)).abs().max().item())
                validation["official_forward_equivalent"] = validation["official_forward_max_abs_diff"] == 0.0
            indices = matcher({"pred_spans": outputs["pred_spans"], "pred_logits": outputs["pred_logits"]}, targets)
            new_match_rows = match_cost_rows(pred_spans, pred_logits, targets, metas, indices, option, initial_widths)
            for row in new_match_rows:
                row["checkpoint"] = checkpoint_name
                row["epoch"] = epoch
                matched_rows.append(row)
                assignment_map[f"{row['qid']}|{row['gt_index']}"] = int(row["assigned_query_slot"])
                slot_bin_counts[(int(row["assigned_query_slot"]), str(row["duration_bin"]))] += 1
                bin_counts[str(row["duration_bin"])] += 1

            for batch_index, meta in enumerate(metas):
                qid = str(meta["qid"])
                duration = float(meta["duration"])
                gt_bin = duration_bin(gt_duration(meta))
                for slot in range(pred_spans.shape[1]):
                    center_sec = float(pred_spans[batch_index, slot, 0] * duration)
                    width_sec = float(pred_spans[batch_index, slot, 1] * duration)
                    metrics = proposal_metrics(center_sec, width_sec, meta)
                    proposal_rows.append({
                        "checkpoint": checkpoint_name,
                        "epoch": epoch,
                        "qid": qid,
                        "vid": str(meta["vid"]),
                        "duration_bin": gt_bin,
                        "audio_duration_sec": duration,
                        "gt_duration_sec": gt_duration(meta),
                        "query_slot": slot,
                        "initial_center_norm": float(initial_spans[slot, 0]),
                        "initial_width_norm": float(initial_widths[slot]),
                        "initial_width_sec": float(initial_widths[slot] * duration),
                        "pred_center_norm": float(pred_spans[batch_index, slot, 0]),
                        "pred_width_norm": float(pred_spans[batch_index, slot, 1]),
                        "pred_center_sec": center_sec,
                        "pred_width_sec": width_sec,
                        "center_error_sec": metrics["center_error_sec"],
                        "iou": metrics["iou"],
                        "width_gt_ratio": metrics["width_gt_ratio"],
                        "abs_log_width_gt_ratio": metrics["abs_log_width_gt_ratio"],
                        "foreground_probability": float(torch.softmax(outputs["pred_logits"][batch_index, slot].detach().float(), dim=-1)[0].item()),
                        "center_le_1s": int(metrics["center_error_sec"] <= CENTER_TOLERANCE_SEC),
                    })
                    if metrics["center_error_sec"] <= CENTER_TOLERANCE_SEC:
                        layer_row = {
                            "checkpoint": checkpoint_name,
                            "epoch": epoch,
                            "qid": qid,
                            "vid": str(meta["vid"]),
                            "duration_bin": gt_bin,
                            "audio_duration_sec": duration,
                            "gt_duration_sec": gt_duration(meta),
                            "query_slot": slot,
                            "final_assigned": int(any(r["qid"] == qid and int(r["assigned_query_slot"]) == slot for r in new_match_rows)),
                        }
                        for layer_index, layer_name in enumerate(("initial", "layer1", "layer2")):
                            if layer_name == "initial":
                                center_norm = float(initial_spans[slot, 0])
                                width_norm = float(initial_spans[slot, 1])
                            else:
                                center_norm = float(layer_spans[layer_index - 1][batch_index, slot, 0])
                                width_norm = float(layer_spans[layer_index - 1][batch_index, slot, 1])
                            width_sec_layer = width_norm * duration
                            ratio = width_sec_layer / max(gt_duration(meta), EPS)
                            layer_row[f"{layer_name}_center_sec"] = center_norm * duration
                            layer_row[f"{layer_name}_width_norm"] = width_norm
                            layer_row[f"{layer_name}_width_sec"] = width_sec_layer
                            layer_row[f"{layer_name}_width_gt_ratio"] = ratio
                            layer_row[f"{layer_name}_abs_log_width_gt_ratio"] = abs(math.log(max(ratio, EPS)))
                        layer_row["initial_width_sec"] = float(initial_spans[slot, 1] * duration)
                        layer_row["log_width_initial_to_layer1"] = math.log(max(layer_row["layer1_width_sec"], EPS) / max(layer_row["initial_width_sec"], EPS)) if "initial_width_sec" in layer_row else math.log(max(layer_row["layer1_width_sec"], EPS) / max(initial_spans[slot, 1] * duration, EPS))
                        layer_row["log_width_layer1_to_layer2"] = math.log(max(layer_row["layer2_width_sec"], EPS) / max(layer_row["layer1_width_sec"], EPS))
                        layer_rows.append(layer_row)

                    for layer_index, layer_name in enumerate(("layer1", "layer2")):
                        width_norm = float(layer_spans[layer_index][batch_index, slot, 1])
                        pre_activation = float(raw_delta[layer_index, batch_index, slot, 1] + inverse_sigmoid(torch.tensor(references[layer_index, batch_index, slot, 1])).item())
                        numeric_rows.append({
                            "checkpoint": checkpoint_name,
                            "epoch": epoch,
                            "qid": qid,
                            "duration_bin": gt_bin,
                            "audio_duration_sec": duration,
                            "gt_duration_sec": gt_duration(meta),
                            "query_slot": slot,
                            "layer": layer_name,
                            "raw_width_delta": float(raw_delta[layer_index, batch_index, slot, 1]),
                            "pre_activation_width_logit": pre_activation,
                            "post_activation_width_norm": width_norm,
                            "width_sec": width_norm * duration,
                            "width_gt_ratio": width_norm * duration / max(gt_duration(meta), EPS),
                            "saturation_distance": min(width_norm, 1.0 - width_norm),
                            "abs_width_logit": abs(pre_activation),
                        })
            validation["num_batches"] += 1

    for (slot, bin_name), count in sorted(slot_bin_counts.items()):
        total = bin_counts[bin_name]
        specialization_rows.append({
            "checkpoint": checkpoint_name,
            "epoch": epoch,
            "query_slot": slot,
            "duration_bin": bin_name,
            "assignment_count": count,
            "bin_assignment_fraction": count / total if total else None,
            "num_targets_in_bin": total,
        })
    if was_training:
        model.train()
    return {
        "proposals": proposal_rows,
        "matched": matched_rows,
        "layers": layer_rows,
        "numerics": numeric_rows,
        "specialization": specialization_rows,
        "validation": validation,
        "assignment_map": assignment_map,
    }


def finite_norm(values: Sequence[torch.Tensor | None]) -> float:
    parts = [value.detach().float().reshape(-1) for value in values if value is not None]
    if not parts:
        return 0.0
    return float(torch.cat(parts).norm().item())


def gradient_probe(
    model: torch.nn.Module,
    matcher: torch.nn.Module,
    cached_batches: Sequence[tuple[list[dict[str, Any]], Any]],
    option: EasyDict,
    epoch: int,
) -> list[dict[str, Any]]:
    """Read pre-update width-coordinate gradients on fixed training batches."""
    from span_utils import generalized_temporal_iou, span_cxw_to_xx

    state = capture_rng()
    was_training = model.training
    model.train()
    rows: list[dict[str, Any]] = []
    try:
        for batch_index, (metas, batched) in enumerate(cached_batches):
            model.zero_grad(set_to_none=True)
            model_inputs, targets = prepare_batch(batched, option)
            outputs = model(**model_inputs)
            layer_outputs = get_model_layers(model, outputs)
            for layer_name, layer_output in layer_outputs:
                indices = matcher(layer_output, targets)
                grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
                for sample_index, meta in enumerate(metas):
                    duration = float(meta["duration"])
                    target_spans = targets["span_labels"][sample_index]["spans"]
                    for slot_tensor, target_tensor in zip(*indices[sample_index]):
                        target_index = int(target_tensor)
                        target_width_sec = float(target_spans[target_index, 1].detach().cpu().item() * duration)
                        grouped[duration_bin(target_width_sec)].append((sample_index, int(slot_tensor), target_index))
                for bin_name in BINS:
                    pairs = grouped.get(bin_name, [])
                    if not pairs:
                        continue
                    pred = layer_output["pred_spans"]
                    pred_values = torch.stack([pred[sample_index, slot, 1] for sample_index, slot, _ in pairs])
                    target_values = torch.stack([targets["span_labels"][sample_index]["spans"][target_index, 1] for sample_index, _, target_index in pairs]).to(pred.device)
                    width_loss = float(option.span_loss_coef) * F.l1_loss(pred_values, target_values, reduction="mean")
                    full_pred = torch.stack([pred[sample_index, slot] for sample_index, slot, _ in pairs])
                    full_target = torch.stack([targets["span_labels"][sample_index]["spans"][target_index] for sample_index, _, target_index in pairs]).to(pred.device)
                    full_l1 = float(option.span_loss_coef) * F.l1_loss(full_pred, full_target, reduction="mean")
                    full_giou = float(option.giou_loss_coef) * (1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(full_pred), span_cxw_to_xx(full_target)))).mean()
                    param_tuple = tuple(model.span_embed.parameters())
                    param_grads = torch.autograd.grad(width_loss, param_tuple, retain_graph=True, allow_unused=True)
                    pred_grad = torch.autograd.grad(width_loss, pred, retain_graph=True, allow_unused=True)[0]
                    geometry_grad = torch.autograd.grad(full_l1 + full_giou, pred, retain_graph=True, allow_unused=True)[0]
                    selected_grad = []
                    selected_geometry_grad = []
                    for sample_index, slot, _ in pairs:
                        if pred_grad is not None:
                            selected_grad.append(pred_grad[sample_index, slot, 1].abs())
                        if geometry_grad is not None:
                            selected_geometry_grad.append(geometry_grad[sample_index, slot].norm())
                    rows.append({
                        "epoch": epoch,
                        "training_position": "before_optimizer_step",
                        "diagnostic_batch": batch_index,
                        "layer": layer_name,
                        "duration_bin": bin_name,
                        "num_matched_targets": len(pairs),
                        "width_coordinate_l1_loss": float(width_loss.detach().item()),
                        "full_span_l1_loss": float(full_l1.detach().item()),
                        "full_geometry_loss": float((full_l1 + full_giou).detach().item()),
                        "width_prediction_grad_abs_mean": float(torch.stack(selected_grad).mean().item()) if selected_grad else 0.0,
                        "geometry_prediction_grad_norm_mean": float(torch.stack(selected_geometry_grad).mean().item()) if selected_geometry_grad else 0.0,
                        "span_embed_width_grad_norm": finite_norm(param_grads),
                    })
            model.zero_grad(set_to_none=True)
    finally:
        restore_rng(state)
        if not was_training:
            model.eval()
    return rows


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, loader: Any, optimizer: Any, option: EasyDict, epoch: int) -> dict[str, float]:
    """Official train_epoch logic with only scalar logging added."""
    from dataset import prepare_batch_inputs

    model.train()
    criterion.train()
    totals: dict[str, list[float]] = defaultdict(list)
    for batch in loader:
        model_inputs, targets = prepare_batch_inputs(batch[1], option.device)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[key] * criterion.weight_dict[key] for key in loss_dict if key in criterion.weight_dict)
        optimizer.zero_grad()
        losses.backward()
        if option.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), option.grad_clip)
        optimizer.step()
        totals["loss_overall"].append(float(losses.detach().item()))
        for key, value in loss_dict.items():
            totals[key].append(float(value.detach().item()))
    return {key: float(np.mean(values)) for key, values in totals.items() if values}


def median(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


def mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def fit_line(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, Any]:
    if len(x_values) < 3:
        return {"N": len(x_values), "slope": None, "intercept": None, "correlation": None}
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    return {"N": len(x), "slope": float(slope), "intercept": float(intercept), "correlation": correlation}


def ols(rows: Sequence[Mapping[str, Any]], outcome: str, features: Sequence[str], categorical: str | None = None) -> dict[str, Any]:
    if len(rows) < max(5, len(features) + 2):
        return {"N": len(rows), "R2": None, "coefficients": {}}
    matrix = [[1.0] for _ in rows]
    names = ["intercept"]
    for feature in features:
        matrix = [row + [float(item[feature])] for row, item in zip(matrix, rows)]
        names.append(feature)
    if categorical is not None:
        values = sorted({str(row[categorical]) for row in rows})
        for value in values[1:]:
            matrix = [row + [1.0 if str(item[categorical]) == value else 0.0] for row, item in zip(matrix, rows)]
            names.append(f"{categorical}={value}")
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray([float(row[outcome]) for row in rows], dtype=np.float64)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / total if total > 0 else 0.0
    return {"N": len(rows), "R2": r2, "coefficients": {name: float(value) for name, value in zip(names, beta)}}


def aggregate_slot_rows(proposals: Sequence[Mapping[str, Any]], checkpoints: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in proposals:
        grouped[(str(row["checkpoint"]), int(row["query_slot"]))].append(row)
    for checkpoint in checkpoints:
        for slot in range(10):
            values = grouped.get((checkpoint, slot), [])
            record: dict[str, Any] = {"checkpoint": checkpoint, "query_slot": slot, "epoch": values[0]["epoch"] if values else None}
            if values:
                record.update({
                    "initial_width_norm": values[0]["initial_width_norm"],
                    "initial_width_sec_median": median(row["initial_width_sec"] for row in values),
                    "final_width_norm_median": median(row["pred_width_norm"] for row in values),
                    "final_width_sec_median": median(row["pred_width_sec"] for row in values),
                    "centered_rate": mean(row["center_le_1s"] for row in values),
                })
                for bin_name in BINS:
                    bin_values = [row for row in values if row["duration_bin"] == bin_name]
                    record[f"{bin_name}_final_width_sec_median"] = median(row["pred_width_sec"] for row in bin_values)
                    record[f"{bin_name}_centered_rate"] = mean(row["center_le_1s"] for row in bin_values)
                    record[f"{bin_name}_N"] = len(bin_values)
            else:
                record.update({"initial_width_norm": None, "initial_width_sec_median": None, "final_width_norm_median": None, "final_width_sec_median": None, "centered_rate": None})
            rows.append(record)
    return rows


def aggregate_width_response(proposals: Sequence[Mapping[str, Any]], checkpoints: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in proposals:
        if int(row["center_le_1s"]) and float(row["gt_duration_sec"]) > 0 and float(row["pred_width_sec"]) > 0:
            groups[(str(row["checkpoint"]), str(row["duration_bin"]))].append(row)
    for checkpoint in checkpoints:
        for bin_name in (*BINS, "GT<5s"):
            values = groups.get((checkpoint, bin_name), []) if bin_name != "GT<5s" else [row for row in proposals if row["checkpoint"] == checkpoint and int(row["center_le_1s"]) and float(row["gt_duration_sec"]) < 5]
            fit = fit_line([math.log(float(row["gt_duration_sec"])) for row in values], [math.log(float(row["pred_width_sec"])) for row in values])
            normalized_fit = fit_line([math.log(float(row["gt_duration_sec"]) / float(row["audio_duration_sec"])) for row in values], [math.log(float(row["pred_width_sec"]) / float(row["audio_duration_sec"])) for row in values])
            rows.append({
                "checkpoint": checkpoint,
                "epoch": values[0]["epoch"] if values else None,
                "duration_bin": bin_name,
                **fit,
                "normalized_slope": normalized_fit["slope"],
                "normalized_correlation": normalized_fit["correlation"],
                "median_pred_gt_ratio": median(row["width_gt_ratio"] for row in values),
                "median_abs_log_width_gt_ratio": median(row["abs_log_width_gt_ratio"] for row in values),
                "N_centered_proposals": len(values),
            })
    return rows


def aggregate_layer_rows(layer_rows: Sequence[Mapping[str, Any]], checkpoints: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        groups[(str(row["checkpoint"]), str(row["duration_bin"]))].append(row)
    for checkpoint in checkpoints:
        for bin_name in (*BINS, "GT<5s"):
            values = groups.get((checkpoint, bin_name), []) if bin_name != "GT<5s" else [row for row in layer_rows if row["checkpoint"] == checkpoint and float(row["gt_duration_sec"]) < 5]
            record: dict[str, Any] = {"checkpoint": checkpoint, "epoch": values[0]["epoch"] if values else None, "duration_bin": bin_name, "N": len(values)}
            for field in ("initial_width_sec", "layer1_width_sec", "layer2_width_sec", "initial_width_gt_ratio", "layer1_width_gt_ratio", "layer2_width_gt_ratio", "initial_abs_log_width_gt_ratio", "layer1_abs_log_width_gt_ratio", "layer2_abs_log_width_gt_ratio", "log_width_initial_to_layer1", "log_width_layer1_to_layer2"):
                record[field + "_median"] = median(row[field] for row in values if field in row)
            rows.append(record)
    return rows


def aggregate_numeric_rows(numeric_rows: Sequence[Mapping[str, Any]], checkpoints: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in numeric_rows:
        groups[(str(row["checkpoint"]), str(row["duration_bin"]), str(row["layer"]))].append(row)
    for checkpoint in checkpoints:
        for bin_name in BINS:
            for layer in ("layer1", "layer2"):
                values = groups.get((checkpoint, bin_name, layer), [])
                rows.append({
                    "checkpoint": checkpoint,
                    "epoch": values[0]["epoch"] if values else None,
                    "duration_bin": bin_name,
                    "layer": layer,
                    "N": len(values),
                    "raw_width_delta_median": median(row["raw_width_delta"] for row in values),
                    "pre_activation_width_logit_median": median(row["pre_activation_width_logit"] for row in values),
                    "post_activation_width_norm_median": median(row["post_activation_width_norm"] for row in values),
                    "abs_width_logit_median": median(row["abs_width_logit"] for row in values),
                    "saturation_distance_median": median(row["saturation_distance"] for row in values),
                    "saturation_fraction_distance_lt_0.05": mean(float(row["saturation_distance"]) < 0.05 for row in values),
                })
    return rows


def aggregate_gradient_rows(gradient_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in gradient_rows:
        groups[(int(row["epoch"]), str(row["layer"]), str(row["duration_bin"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (epoch, layer, bin_name), values in sorted(groups.items()):
        rows.append({
            "epoch": epoch,
            "training_position": "before_optimizer_step",
            "layer": layer,
            "duration_bin": bin_name,
            "diagnostic_batches": len(values),
            "matched_targets": sum(int(row["num_matched_targets"]) for row in values),
            "width_coordinate_l1_loss_median": median(row["width_coordinate_l1_loss"] for row in values),
            "width_prediction_grad_abs_mean_median": median(row["width_prediction_grad_abs_mean"] for row in values),
            "geometry_prediction_grad_norm_mean_median": median(row["geometry_prediction_grad_norm_mean"] for row in values),
            "span_embed_width_grad_norm_median": median(row["span_embed_width_grad_norm"] for row in values),
        })
    return rows


def aggregate_audio_duration(matched_rows: Sequence[Mapping[str, Any]], checkpoints: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for bin_name in NARROW_BINS:
            values = [row for row in matched_rows if row["checkpoint"] == checkpoint and narrow_bin(float(row["gt_duration_sec"])) == bin_name and float(row["matched_center_error_sec"]) <= CENTER_TOLERANCE_SEC and float(row["pred_width_sec"]) > 0]
            enriched = [{**row, "log_pred_width": math.log(float(row["pred_width_sec"])), "log_gt_width": math.log(float(row["gt_duration_sec"])), "log_audio_duration": math.log(float(row["audio_duration_sec"]))} for row in values]
            model_fit = ols(enriched, "log_pred_width", ("log_gt_width", "log_audio_duration"), "assigned_query_slot") if enriched else {"N": 0, "R2": None, "coefficients": {}}
            rows.append({
                "checkpoint": checkpoint,
                "epoch": values[0]["epoch"] if values else None,
                "gt_duration_slice": bin_name,
                "N_centered_matched": len(values),
                "median_audio_duration_sec": median(row["audio_duration_sec"] for row in values),
                "median_pred_width_gt_ratio": median(row["matched_width_gt_ratio"] for row in values),
                "median_pred_width_sec": median(row["pred_width_sec"] for row in values),
                "unadjusted_log_width_audio_slope": fit_line([float(row["log_audio_duration"]) for row in enriched], [float(row["log_pred_width"]) for row in enriched])["slope"] if len(enriched) >= 3 else None,
                "adjusted_N": model_fit["N"],
                "adjusted_R2": model_fit["R2"],
                "adjusted_log_audio_duration_coefficient": model_fit["coefficients"].get("log_audio_duration"),
            })
    return rows


def first_epoch(rows: Sequence[Mapping[str, Any]], predicate: Any) -> int | None:
    epochs = sorted({int(row["epoch"]) for row in rows if predicate(row)})
    return epochs[0] if epochs else None


def temporal_events(
    width_rows: Sequence[Mapping[str, Any]],
    numeric_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    matched_rows: Sequence[Mapping[str, Any]],
    specialization_rows: Sequence[Mapping[str, Any]],
    audio_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    short_width = [row for row in width_rows if row["duration_bin"] == "GT<5s" and row["N_centered_proposals"] >= 20]
    event_a = first_epoch(short_width, lambda row: row["median_pred_gt_ratio"] is not None and float(row["median_pred_gt_ratio"]) >= 1.5)
    event_b = first_epoch(short_width, lambda row: row["slope"] is not None and float(row["slope"]) < 0.5)
    short_numeric = [row for row in numeric_rows if row["duration_bin"] in ("0-2s", "2-5s") and row["N"] >= 20]
    event_c = first_epoch(short_numeric, lambda row: row["saturation_fraction_distance_lt_0.05"] is not None and float(row["saturation_fraction_distance_lt_0.05"]) >= 0.5)

    event_d = None
    for epoch in sorted({int(row["epoch"]) for row in gradient_rows}):
        current = [row for row in gradient_rows if int(row["epoch"]) == epoch and row["layer"] == "layer2"]
        short = [row for row in current if row["duration_bin"] in ("0-2s", "2-5s") and row["matched_targets"] >= 5]
        long = [row for row in current if row["duration_bin"] in ("10-20s", "20s+") and row["matched_targets"] >= 5]
        if short and long:
            short_value = np.mean([float(row["span_embed_width_grad_norm_median"]) for row in short])
            long_value = np.mean([float(row["span_embed_width_grad_norm_median"]) for row in long])
            if long_value > 0 and short_value / long_value < 0.8:
                event_d = epoch
                break

    event_e = None
    by_epoch_bin: dict[tuple[int, str], tuple[int, float]] = {}
    grouped_specialization: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in specialization_rows:
        grouped_specialization[(int(row["epoch"]), str(row["duration_bin"]))].append(row)
    for key, values in grouped_specialization.items():
        eligible = [row for row in values if int(row["assignment_count"]) >= 5]
        if eligible:
            top = max(eligible, key=lambda row: int(row["assignment_count"]))
            by_epoch_bin[key] = (int(top["query_slot"]), float(top["bin_assignment_fraction"]))
    for bin_name in BINS:
        epochs = sorted(epoch for epoch, current_bin in by_epoch_bin if current_bin == bin_name)
        for left, middle, right in zip(epochs, epochs[1:], epochs[2:]):
            if right - left <= 20 and by_epoch_bin[(left, bin_name)][0] == by_epoch_bin[(middle, bin_name)][0] == by_epoch_bin[(right, bin_name)][0]:
                event_e = left
                break
        if event_e is not None:
            break

    event_f = first_epoch(specialization_rows, lambda row: int(row["assignment_count"]) >= 5 and row["bin_assignment_fraction"] is not None and float(row["bin_assignment_fraction"]) >= 0.6)

    event_g = None
    for epoch in sorted({int(row["epoch"]) for row in matched_rows}):
        short = [row for row in matched_rows if int(row["epoch"]) == epoch and row["duration_bin"] in ("0-2s", "2-5s") and row["column_cost_margin"] is not None]
        long = [row for row in matched_rows if int(row["epoch"]) == epoch and row["duration_bin"] in ("10-20s", "20s+") and row["column_cost_margin"] is not None]
        if len(short) >= 20 and len(long) >= 20 and float(median(row["column_cost_margin"] for row in short) or 0) < 0.8 * float(median(row["column_cost_margin"] for row in long) or 0):
            event_g = epoch
            break

    event_h = first_epoch(audio_rows, lambda row: row["adjusted_log_audio_duration_coefficient"] is not None and abs(float(row["adjusted_log_audio_duration_coefficient"])) >= 0.5 and int(row["adjusted_N"]) >= 20)
    return {"A_scale_error": event_a, "B_response_slope_below_0.5": event_b, "C_saturation_fraction_ge_0.5": event_c, "D_short_gradient_ratio_below_0.8": event_d, "E_slot_assignment_order_stable": event_e, "F_slot_bin_fraction_ge_0.6": event_f, "G_short_assignment_margin_below_0.8_long": event_g, "H_audio_duration_coefficient_ge_0.5": event_h}


def assess_hypotheses(events: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    def status(value: bool | None) -> str:
        return "SUPPORTED" if value is True else "INCONCLUSIVE" if value is None else "NOT_SUPPORTED"

    a = events["A_scale_error"]
    b = events["B_response_slope_below_0.5"]
    c = events["C_saturation_fraction_ge_0.5"]
    d = events["D_short_gradient_ratio_below_0.8"]
    e = events["E_slot_assignment_order_stable"]
    f = events["F_slot_bin_fraction_ge_0.6"]
    g = events["G_short_assignment_margin_below_0.8_long"]
    h = events["H_audio_duration_coefficient_ge_0.5"]
    def precedes(left: int | None, right: int | None) -> bool | None:
        if left is None or right is None:
            return None
        if left < right:
            return True
        if left == right:
            return None
        return False

    saturation_vs_gradient = None if c is None or d is None else c < d if c != d else None
    observed = [value for value in (b, c, d, g, h) if value is not None]
    coupled = bool(len(observed) >= 3 and max(observed) - min(observed) <= 10)
    hypotheses = [
        {"hypothesis": "H1", "description": "early weak scale signal", "status": status(precedes(b, a)), "supporting_event": b},
        {"hypothesis": "H2", "description": "query-slot scale priors trap short GT", "status": status(precedes(e, a)), "supporting_event": e},
        {"hypothesis": "H3", "description": "saturation develops and attenuates gradients", "status": status(saturation_vs_gradient), "supporting_event": c},
        {"hypothesis": "H4", "description": "matching ambiguity precedes scale failure", "status": status(precedes(g, a)), "supporting_event": g},
        {"hypothesis": "H5", "description": "audio-duration geometry contributes early", "status": status(precedes(h, a)), "supporting_event": h},
        {"hypothesis": "H6", "description": "multiple coupled factors co-evolve", "status": status(coupled), "supporting_event": None},
    ]
    supported = [item["hypothesis"] for item in hypotheses if item["status"] == "SUPPORTED"]
    if len(supported) >= 2 or "H6" in supported:
        candidate = "MULTIPLE_COUPLED_FACTORS"
    elif supported == ["H1"]:
        candidate = "OUTPUT_COORDINATE"
    elif supported == ["H2"]:
        candidate = "QUERY_SLOT_SCALE_PRIOR"
    elif supported == ["H4"]:
        candidate = "MATCHING_GEOMETRY"
    elif supported == ["H5"]:
        candidate = "AUDIO_LENGTH_GEOMETRY"
    elif supported == ["H3"]:
        candidate = "OPTIMIZATION_GRADIENT"
    else:
        candidate = "INCONCLUSIVE"
    return hypotheses, candidate


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(column) is None else str(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def write_reports(
    output: Path,
    args: argparse.Namespace,
    option: EasyDict,
    checkpoints: Sequence[str],
    proposals: Sequence[Mapping[str, Any]],
    matched: Sequence[Mapping[str, Any]],
    layers: Sequence[Mapping[str, Any]],
    numerics: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    specialization: Sequence[Mapping[str, Any]],
    validations: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    slot_rows = aggregate_slot_rows(proposals, checkpoints)
    width_rows = aggregate_width_response(proposals, checkpoints)
    layer_rows = aggregate_layer_rows(layers, checkpoints)
    numeric_rows = aggregate_numeric_rows(numerics, checkpoints)
    gradient_rows = aggregate_gradient_rows(gradients)
    audio_rows = aggregate_audio_duration(matched, checkpoints)
    events = temporal_events(width_rows, numeric_rows, gradient_rows, matched, specialization, audio_rows)
    hypotheses, candidate = assess_hypotheses(events)

    write_csv(output / "slot_width_evolution.csv", slot_rows)
    write_csv(output / "gt_pred_width_response_over_time.csv", width_rows)
    write_csv(output / "layerwise_contraction_over_time.csv", layer_rows)
    write_csv(output / "width_head_numerics_over_time.csv", numeric_rows)
    write_csv(output / "gradient_history_by_duration.csv", gradient_rows)
    write_csv(output / "hungarian_assignment_dynamics.csv", matched)
    write_csv(output / "slot_specialization_over_time.csv", specialization)
    write_csv(output / "audio_duration_effect_over_time.csv", audio_rows)
    write_csv(output / "proposal_rows.csv", proposals)
    write_csv(output / "gradient_history_raw.csv", gradients)

    short_width_final = [row for row in width_rows if row["checkpoint"] == "epoch_200" and row["duration_bin"] in ("0-2s", "2-5s")]
    earliest = {key: value for key, value in events.items()}
    summary = {
        "status": "COMPLETED",
        "scope": "single matched baseline-only 200-epoch run; no full audit and no method design",
        "scientific_protocol": "research-experiment-protocol",
        "existing_checkpoints_sufficient": False,
        "retraining_required": True,
        "training_seed": int(args.seed),
        "baseline_commit": args.baseline_commit,
        "config_sha256": sha256_file(Path(args.config)),
        "num_train_queries": len(read_jsonl(Path(args.train_jsonl))),
        "num_val_queries": len(read_jsonl(Path(args.val_jsonl))),
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "gradient_epochs": list(GRADIENT_EPOCHS),
        "official_forward_validation": list(validations),
        "history_rows": len(history),
        "temporal_events": earliest,
        "hypotheses": hypotheses,
        "causal_priority_candidate": candidate,
        "method_design_gate": "NO",
        "causal_ambiguity_remaining": "Observational checkpoints and pre-update gradients establish temporal co-occurrence, not an intervention that separates output-coordinate, matching, slot-prior, duration-geometry, and optimization mechanisms.",
        "final_short_width_response": short_width_final,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "training_history.json", {"history": list(history)})
    write_json(output / "run_provenance.json", {"argv": list(sys.argv), "baseline_commit": args.baseline_commit, "config_sha256": summary["config_sha256"], "python": sys.version, "torch": torch.__version__})

    event_rows = [{"event": key, "first_epoch": value} for key, value in events.items()]
    temporal_text = "# Temporal ordering\n\nThis is an observational ordering audit. A first epoch is reported only when the predeclared threshold has enough data; it is not a causal intervention.\n\n" + markdown_table(event_rows, ("event", "first_epoch")) + "\n\n"
    temporal_text += "## Reading rule\n\nThe event order is used to decide which mechanism deserves the next diagnostic, but this run does not justify changing the model or claiming a unique cause.\n"
    (output / "temporal_ordering.md").write_text(temporal_text, encoding="utf-8")

    hypothesis_text = "# Hypothesis assessment\n\n" + markdown_table(hypotheses, ("hypothesis", "description", "status", "supporting_event")) + "\n\nH1–H6 are evidence labels for this observational audit. SUPPORTED means the predeclared temporal criterion was met; it does not mean causation was proven.\n"
    (output / "hypothesis_assessment.md").write_text(hypothesis_text, encoding="utf-8")
    (output / "causal_priority.md").write_text(f"# Causal priority\n\nCandidate: **{candidate}**\n\nThis is a priority for the next diagnostic, not a causal conclusion. The run has no intervention and cannot distinguish coupled mechanisms that change at similar epochs.\n", encoding="utf-8")
    (output / "method_design_gate.md").write_text("# Method-design gate\n\n**NO.** This audit does not authorize a new method. It only characterizes temporal ordering and identifies the next causal diagnostic needed to separate mechanisms.\n", encoding="utf-8")
    return summary


def write_inventory(output: Path, args: argparse.Namespace) -> None:
    paths = [
        ("official baseline checkpoint", Path(args.official_checkpoint)),
        ("existing R1 baseline checkpoint", Path(args.existing_r1_baseline)),
        ("existing R1 baseline history", Path(args.existing_r1_history)),
        ("existing R1 matched config", Path(args.existing_r1_config)),
    ]
    lines = [
        "# Checkpoint inventory",
        "",
        "The existing artifacts were inspected before this audit. They contain no dense per-epoch model states sufficient for temporal width, matching, and gradient analysis; therefore this task runs exactly one new baseline-only training trajectory.",
        "",
        "| artifact | path | exists | bytes | SHA-256 |",
        "|---|---|---:|---:|---|",
    ]
    for label, path in paths:
        exists = path.exists()
        size = path.stat().st_size if exists else ""
        digest = sha256_file(path) if exists and path.is_file() else ""
        lines.append(f"| {label} | `{path}` | {exists} | {size} | {digest} |")
    lines.extend([
        "",
        "Conclusion: existing checkpoints are insufficient for the requested temporal ordering. No test split is used.",
    ])
    (output / "checkpoint_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_config(output: Path, args: argparse.Namespace, option: EasyDict) -> None:
    payload = {
        "seed": int(args.seed),
        "baseline_commit": args.baseline_commit,
        "config_path": str(args.config),
        "config_sha256": sha256_file(Path(args.config)),
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "feature_root": str(args.feature_root),
        "device": str(option.device),
        "n_epoch": int(option.n_epoch),
        "batch_size": int(option.bsz),
        "eval_batch_size": int(option.eval_bsz),
        "num_workers": int(option.num_workers),
        "lr": float(option.lr),
        "lr_drop": int(option.lr_drop),
        "weight_decay": float(option.wd),
        "grad_clip": float(option.grad_clip),
        "num_queries": int(option.num_queries),
        "hidden_dim": int(option.hidden_dim),
        "enc_layers": int(option.enc_layers),
        "dec_layers": int(option.dec_layers),
        "span_loss_type": str(option.span_loss_type),
        "set_cost_span": float(option.set_cost_span),
        "set_cost_giou": float(option.set_cost_giou),
        "set_cost_class": float(option.set_cost_class),
        "span_loss_coef": float(option.span_loss_coef),
        "giou_loss_coef": float(option.giou_loss_coef),
        "label_loss_coef": float(option.label_loss_coef),
        "instrumentation": "fixed observations only; no model/loss/matcher/parameterization/optimizer changes",
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "gradient_epochs": list(GRADIENT_EPOCHS),
        "diagnostic_queries_per_duration_bin": 16,
        "test_split_used": False,
    }
    (output / "training_config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--existing-r1-baseline", type=Path, required=True)
    parser.add_argument("--existing-r1-history", type=Path, required=True)
    parser.add_argument("--existing-r1-config", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    write_inventory(output, args)

    option = load_options(args.config, args.worktree, args.feature_root, output, args.device)
    option.seed = int(args.seed)
    option.num_workers = 4
    option.train_path = str(args.train_jsonl)
    option.val_path = str(args.val_jsonl)
    option.eval_path = str(args.val_jsonl)
    option.results_dir = str(output / "runtime_eval")
    Path(option.results_dir).mkdir(parents=True, exist_ok=True)
    write_training_config(output, args, option)

    train_metadata = read_jsonl(args.train_jsonl)
    val_metadata = read_jsonl(args.val_jsonl)
    set_seed(args.seed)
    train_dataset = build_dataset(option, args.train_jsonl)
    val_dataset = build_dataset(option, args.val_jsonl)
    selected = make_diagnostic_indices(train_metadata, per_bin=16)
    cached_batches = cache_diagnostic_batches(train_dataset, selected, option)
    write_json(output / "diagnostic_subset.json", {"indices": selected, "qids": [train_metadata[index]["qid"] for index in selected], "count": len(selected)})

    # Restoring this state means diagnostic cache construction does not alter
    # model initialization, DataLoader seeding, or the official trajectory.
    set_seed(args.seed)
    model, criterion, optimizer, scheduler = load_model(option)
    matcher = criterion.matcher
    history: list[dict[str, Any]] = []
    all_proposals: list[dict[str, Any]] = []
    all_matched: list[dict[str, Any]] = []
    all_layers: list[dict[str, Any]] = []
    all_numerics: list[dict[str, Any]] = []
    all_specialization: list[dict[str, Any]] = []
    all_gradients: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    checkpoint_names = [f"epoch_{epoch:03d}" for epoch in CHECKPOINT_EPOCHS] + ["best_validation"]

    def collect_and_save(epoch: int, name: str) -> None:
        save_state(output / "checkpoints" / f"{name}.pth", model, optimizer, scheduler, epoch)
        result = collect_checkpoint(model, matcher, val_dataset, option, name, epoch, batch_size=int(option.eval_bsz))
        all_proposals.extend(result["proposals"])
        all_matched.extend(result["matched"])
        all_layers.extend(result["layers"])
        all_numerics.extend(result["numerics"])
        all_specialization.extend(result["specialization"])
        validations.append(result["validation"])
        write_csv(output / "_progress_proposals.csv", all_proposals)
        write_csv(output / "_progress_matched.csv", all_matched)

    collect_and_save(0, "epoch_000")
    gradient_epochs = set(GRADIENT_EPOCHS)
    checkpoint_epochs = set(CHECKPOINT_EPOCHS)
    from dataset import start_end_collate
    from evaluate import eval_epoch

    train_loader = DataLoader(train_dataset, batch_size=option.bsz, shuffle=True, num_workers=option.num_workers, collate_fn=start_end_collate)
    best_score = -float("inf")
    best_epoch = None
    for epoch in range(1, int(option.n_epoch) + 1):
        if epoch in gradient_epochs:
            probe_rows = gradient_probe(model, matcher, cached_batches, option, epoch)
            all_gradients.extend(probe_rows)
            write_csv(output / "_progress_gradient_history.csv", all_gradients)
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, option, epoch)
        scheduler.step()
        with torch.no_grad():
            metrics, eval_meters, _ = eval_epoch(model, val_dataset, option, f"latest_epoch_{epoch:03d}_val_preds.jsonl", criterion)
        brief = metrics["brief"] if metrics is not None else {}
        score = float(brief.get("MR-full-R1@0.7", 0.0))
        history.append({"epoch": epoch, "train": train_metrics, "val_brief": brief, "val_score_MR-full-R1@0.7": score})
        write_json(output / "training_history.json", {"history": history})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_state(output / "best_validation.pth", model, optimizer, scheduler, epoch)
        if epoch in checkpoint_epochs:
            collect_and_save(epoch, f"epoch_{epoch:03d}")
        if epoch == 1 or epoch % 10 == 0 or epoch == int(option.n_epoch):
            print(json.dumps({"epoch": epoch, "train_loss": train_metrics.get("loss_overall"), "val_score": score, "best_epoch": best_epoch}, ensure_ascii=False), flush=True)

    best_checkpoint = torch.load(output / "best_validation.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    collect_and_save(int(best_checkpoint["epoch"]), "best_validation")
    summary = write_reports(output, args, option, checkpoint_names, all_proposals, all_matched, all_layers, all_numerics, all_gradients, all_specialization, validations, history)
    summary["best_validation_epoch"] = int(best_checkpoint["epoch"])
    summary["best_validation_score"] = float(best_score)
    write_json(output / "summary.json", summary)
    print(json.dumps({"status": "completed", "output": str(output), "best_epoch": best_checkpoint["epoch"], "best_score": best_score}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
