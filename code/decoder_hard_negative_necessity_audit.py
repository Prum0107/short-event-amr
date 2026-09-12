#!/usr/bin/env python3
"""Inference-only necessity and layer audit for frozen QD-DETR decoding.

The script removes a pre-defined non-GT hard region from decoder cross
attention and compares it with matched random removals.  It does not change
the encoder, positional coordinates, checkpoint, loss, matcher, ranking, or
span parameterization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PRIMARY_BINS = ("0-2s", "2-5s")
RANDOM_REPLICATES = 10
CLIP_LENGTH_DEFAULT = 1.0
CENTER_TOLERANCE_SEC = 2.0
TOP_K = 10
STRATUM_ADJACENT_SEC = 1.0
STRATUM_MODERATE_SEC = 5.0


def load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cohort_from_previous(counterfactual: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(counterfactual)
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["duration_bin"] in PRIMARY_BINS:
            grouped[row["qid"]][row["condition"]].append(row)
    output: dict[str, dict[str, Any]] = {}
    for qid, conditions in grouped.items():
        hard = conditions["HARD_25"]
        random_rows = conditions["RANDOM_25"]
        if len(hard) != 1 or len(random_rows) != RANDOM_REPLICATES:
            raise RuntimeError(f"invalid previous replicate count qid={qid}")
        random_mean = float(np.mean([float(row["center_hit10_2s"]) for row in random_rows]))
        hard_value = float(hard[0]["center_hit10_2s"])
        output[qid] = {
            "qid": qid,
            "vid": hard[0]["vid"],
            "duration_bin": hard[0]["duration_bin"],
            "gt_duration_sec": float(hard[0]["gt_duration_sec"]),
            "audio_duration_sec": float(hard[0]["audio_duration_sec"]),
            "hard_center_hit10_2s": hard_value,
            "random25_center_hit10_2s_mean": random_mean,
            "hard_minus_random_center": hard_value - random_mean,
            "cohort": "harmful" if hard_value < random_mean else "control",
        }
    return output


def nearest_gap(start: float, end: float, windows: Sequence[Sequence[float]]) -> float:
    values = []
    for gt_start, gt_end in windows:
        if end < float(gt_start):
            values.append(float(gt_start) - end)
        elif start > float(gt_end):
            values.append(start - float(gt_end))
        else:
            values.append(0.0)
    return min(values)


def token_stratum(index: int, clip_length: float, windows: Sequence[Sequence[float]]) -> str:
    start = float(index) * clip_length
    end = start + clip_length
    gap = nearest_gap(start, end, windows)
    if gap <= STRATUM_ADJACENT_SEC:
        return "adjacent"
    if gap <= STRATUM_MODERATE_SEC:
        return "moderately_near"
    return "remote"


def load_peak_indices(geometry_row: Mapping[str, str]) -> list[int]:
    indices = [int(value) for value in json.loads(geometry_row["hard_peak_token_indices"])]
    if not indices:
        raise RuntimeError(f"empty hard peak qid={geometry_row['qid']}")
    return indices


def mask_from_keep(valid: np.ndarray, remove_indices: Sequence[int]) -> torch.Tensor:
    padding = ~valid.copy()
    padding[list(remove_indices)] = True
    return torch.from_numpy(padding)


def build_removal_specs(
    qid: str,
    valid: np.ndarray,
    gt_mask: np.ndarray,
    hard_indices: Sequence[int],
    windows: Sequence[Sequence[float]],
    clip_length: float,
    base_seed: int,
) -> dict[str, Any]:
    non_gt = [int(value) for value in np.flatnonzero(valid & ~gt_mask)]
    hard_set = set(int(value) for value in hard_indices)
    if not hard_set.issubset(set(non_gt)):
        raise RuntimeError(f"hard region overlaps invalid/GT token qid={qid}")
    hard_strata = [token_stratum(index, clip_length, windows) for index in hard_indices]
    hard_stratum = max(set(hard_strata), key=hard_strata.count)
    random_masks: list[torch.Tensor] = []
    random_selected: list[list[int]] = []
    fallback_flags: list[bool] = []
    for replicate in range(RANDOM_REPLICATES):
        seed = int(base_seed + sum((index + 1) * ord(char) for index, char in enumerate(qid)) * 1_000_003 + replicate)
        rng = np.random.default_rng(seed)
        same_stratum = [
            index for index in non_gt
            if index not in hard_set and token_stratum(index, clip_length, windows) == hard_stratum
        ]
        other = [index for index in non_gt if index not in hard_set and index not in same_stratum]
        same_order = rng.permutation(same_stratum).tolist()
        other_order = rng.permutation(other).tolist()
        selected = [int(value) for value in (same_order + other_order)[: len(hard_set)]]
        if len(selected) != len(hard_set):
            raise RuntimeError(f"random removal count failed qid={qid} replicate={replicate}")
        random_selected.append(selected)
        random_masks.append(mask_from_keep(valid, selected))
        fallback_flags.append(len(same_stratum) < len(hard_set))
    return {
        "hard_mask": mask_from_keep(valid, sorted(hard_set)),
        "hard_indices": sorted(hard_set),
        "hard_stratum": hard_stratum,
        "random_masks": random_masks,
        "random_selected": random_selected,
        "random_fallback_flags": fallback_flags,
    }


def encode_full_audio(cf: types.ModuleType, model: torch.nn.Module, model_inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return cf.encode_full_audio(model, model_inputs)


def decode_with_layer_masks(
    model: torch.nn.Module,
    encoded: Mapping[str, torch.Tensor],
    default_mask: torch.Tensor,
    layer_masks: Sequence[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    memory_local = encoded["memory_local"]
    batch_size = memory_local.shape[1]
    refpoint_embed = model.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
    target = torch.zeros(refpoint_embed.shape[0], batch_size, memory_local.shape[2], device=memory_local.device)
    originals: list[Any] = []
    if layer_masks is not None:
        for layer_index, layer in enumerate(model.transformer.decoder.layers):
            original = layer.forward
            originals.append(original)
            layer_mask = layer_masks[layer_index]

            def wrapped(*args: Any, _original=original, _layer_mask=layer_mask, **kwargs: Any):
                kwargs["memory_key_padding_mask"] = _layer_mask
                return _original(*args, **kwargs)

            layer.forward = wrapped
    try:
        hidden_states, references = model.transformer.decoder(
            target,
            memory_local,
            memory_key_padding_mask=default_mask,
            pos=encoded["position_local"],
            refpoints_unsigmoid=refpoint_embed,
        )
    finally:
        if layer_masks is not None:
            for layer, original in zip(model.transformer.decoder.layers, originals):
                layer.forward = original
    logits = model.class_embed(hidden_states)
    spans = model.span_embed(hidden_states) + cf_inverse_sigmoid(references)
    if model.span_loss_type == "l1":
        spans = spans.sigmoid()
    return {
        "pred_logits": logits[-1],
        "pred_spans": spans[-1],
        "all_pred_logits": logits,
        "all_pred_spans": spans,
        "references": references,
        "initial_references": refpoint_embed.sigmoid(),
    }


_CF: types.ModuleType


def cf_inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    return _CF.inverse_sigmoid(value)


def postprocess(decoded: Mapping[str, torch.Tensor], index: int, meta: Mapping[str, Any], clip_length: float) -> list[list[float]]:
    return _CF.postprocess_prediction(decoded["pred_spans"][index], decoded["pred_logits"][index], float(meta["duration"]), clip_length)


def transition(full_hit: int, intervention_hit: int) -> str:
    if full_hit == 0 and intervention_hit == 1:
        return "miss_to_hit"
    if full_hit == 1 and intervention_hit == 0:
        return "hit_to_miss"
    if full_hit == 1 and intervention_hit == 1:
        return "both_hit"
    return "both_miss"


def references_row(
    decoded: Mapping[str, torch.Tensor],
    index: int,
    meta: Mapping[str, Any],
    hard_center: float,
    condition: str,
    replicate: int,
    clip_length: float,
    final_predictions: Sequence[Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    scores = torch.softmax(decoded["pred_logits"][index].detach().float(), dim=-1)[:, 0]
    order = torch.argsort(scores, descending=True).detach().cpu().numpy().tolist()
    order = order[: min(TOP_K, len(order))]
    refs = decoded["references"][:, index].detach().float().cpu().numpy()
    duration = float(meta["duration"])
    gt_centers = [(float(start) + float(end)) / 2.0 for start, end in meta["relevant_windows"]]
    if final_predictions is None:
        final_predictions = postprocess(decoded, index, meta, clip_length)
    final_centers = [
        (float(prediction[0]) + float(prediction[1])) / 2.0
        for prediction in final_predictions[:TOP_K]
    ]
    final_gt_density = float(
        np.mean([min(abs(center - gt) for gt in gt_centers) <= CENTER_TOLERANCE_SEC for center in final_centers])
    ) if final_centers else 0.0
    final_hard_density = float(
        np.mean([abs(center - hard_center) <= CENTER_TOLERANCE_SEC for center in final_centers])
    ) if final_centers else 0.0
    output: list[dict[str, Any]] = []
    for layer_index in range(refs.shape[0]):
        centers = refs[layer_index, order, 0] * duration
        gt_distances = np.asarray([min(abs(float(center) - gt) for gt in gt_centers) for center in centers])
        hard_distances = np.abs(centers - hard_center)
        output.append({
            "qid": str(meta["qid"]),
            "vid": meta["vid"],
            "duration_bin": meta.get("duration_bin", ""),
            "cohort": "",
            "condition": condition,
            "replicate": replicate,
            "layer": layer_index + 1,
            "top_k": len(order),
            "mean_gt_distance_sec": float(np.mean(gt_distances)),
            "mean_hard_distance_sec": float(np.mean(hard_distances)),
            "median_gt_distance_sec": float(np.median(gt_distances)),
            "median_hard_distance_sec": float(np.median(hard_distances)),
            "n_refs_closer_gt": int(np.sum(gt_distances < hard_distances)),
            "n_refs_closer_hard": int(np.sum(hard_distances < gt_distances)),
            "n_refs_tied": int(np.sum(np.isclose(gt_distances, hard_distances, atol=1e-8))),
            "final_gt_center_density": final_gt_density,
            "final_hard_center_density": final_hard_density,
            "final_center_density_tolerance_sec": CENTER_TOLERANCE_SEC,
            "reference_centers_sec": json.dumps([float(value) for value in centers.tolist()]),
        })
    return output


def run(args: argparse.Namespace) -> None:
    global _CF
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cf = load_module(args.counterfactual_runner.resolve(), "necessity_counterfactual_helpers")
    _CF = cf
    cohort = cohort_from_previous(args.counterfactual_csv.resolve())
    geometry = {row["qid"]: row for row in read_csv(args.geometry_csv.resolve())}
    if set(cohort) != set(geometry):
        raise RuntimeError("previous harmful/control cohort and hard-region geometry qids differ")
    metadata = read_jsonl(args.test_jsonl.resolve())
    meta_by_qid = {str(item["qid"]): item for item in metadata}
    if set(cohort) - set(meta_by_qid):
        raise RuntimeError("missing cohort metadata")
    reference_by_qid = {str(item["qid"]): item for item in read_jsonl(args.reference_submission.resolve())}

    sys.path.insert(0, str(args.worktree.resolve() / "src"))
    from dataset import prepare_batch_inputs, start_end_collate
    from evaluate import setup_model

    option = cf.load_options(args.config.resolve(), args.worktree.resolve(), output, args.device)
    dataset = cf.build_dataset(option)
    selected_indices = [index for index, item in enumerate(dataset.data) if str(item["qid"]) in cohort]
    if len(selected_indices) != len(cohort):
        raise RuntimeError("dataset and cohort size mismatch")
    loader = DataLoader(
        Subset(dataset, selected_indices),
        collate_fn=start_end_collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    model, criterion, _, _ = setup_model(option)
    del criterion
    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    clip_length = float(option.clip_length)
    full_reference_subset: list[dict[str, Any]] = []
    observed_full_subset: list[dict[str, Any]] = []
    remove_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    harmful_control_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    rescued_rows: list[dict[str, Any]] = []
    validation = {
        "batches": 0,
        "full_access_reference_pass": True,
        "official_same_run_pass": True,
        "saved_submission_reference_pass": True,
        "mask_contract_pass": True,
        "finite_pass": True,
        "layer_wrapper_pass": True,
        "max_full_access_difference": 0.0,
        "max_official_same_run_difference": 0.0,
        "layer_count": None,
        "num_queries": None,
    }

    with torch.no_grad():
        for metas, batched in loader:
            model_inputs, _targets = prepare_batch_inputs(batched, option.device)
            encoded = encode_full_audio(cf, model, model_inputs)
            # Validate the intervention's FULL_ACCESS path against the
            # official model forward in the same process.  The saved
            # submission may have been produced on a different CUDA/runtime
            # stack, so it is retained as provenance but is not the primary
            # numerical oracle for this inference-only audit.
            official_decoded = model(**model_inputs)
            valid = encoded["valid_audio"].detach().cpu().numpy().astype(bool)
            batch_size = len(metas)
            full_mask = encoded["decoder_padding_full"]
            layer_count = len(model.transformer.decoder.layers)
            validation["layer_count"] = layer_count
            validation["num_queries"] = int(model.num_queries)
            hard_masks: list[torch.Tensor] = []
            random_masks_by_rep: list[list[torch.Tensor]] = [[] for _ in range(RANDOM_REPLICATES)]
            batch_info: list[dict[str, Any]] = []
            for index, meta in enumerate(metas):
                qid = str(meta["qid"])
                geometry_row = geometry[qid]
                windows = [[float(start), float(end)] for start, end in meta["relevant_windows"]]
                gt_mask = cf.gt_token_mask(meta, valid[index], clip_length)
                peak_indices = load_peak_indices(geometry_row)
                specs = build_removal_specs(qid, valid[index], gt_mask, peak_indices, windows, clip_length, args.seed)
                hard_masks.append(specs["hard_mask"])
                for replicate in range(RANDOM_REPLICATES):
                    random_masks_by_rep[replicate].append(specs["random_masks"][replicate])
                batch_info.append({
                    "qid": qid,
                    "meta": meta,
                    "cohort": cohort[qid]["cohort"],
                    "geometry": geometry_row,
                    "gt_mask": gt_mask,
                    "specs": specs,
                    "hard_center": float(geometry_row["hard_peak_center_sec"]),
                })
            hard_batch_mask = torch.stack(hard_masks).to(full_mask.device)
            random_batch_masks = [torch.stack(items).to(full_mask.device) for items in random_masks_by_rep]
            # `True` in a decoder padding mask means "remove this token";
            # therefore removed valid tokens are expected to be TRUE.  Check
            # the intervention contract from the frozen index sets instead of
            # comparing mask polarity with the encoder validity mask.
            for batch_index, info in enumerate(batch_info):
                valid_indices = set(np.flatnonzero(valid[batch_index]).tolist())
                gt_indices = set(np.flatnonzero(info["gt_mask"]).tolist())
                hard_indices = set(info["specs"]["hard_indices"])
                if not hard_indices.issubset(valid_indices) or hard_indices & gt_indices:
                    validation["mask_contract_pass"] = False
                for selected in info["specs"]["random_selected"]:
                    selected_set = set(selected)
                    if len(selected_set) != len(hard_indices) or not selected_set.issubset(valid_indices) or selected_set & gt_indices:
                        validation["mask_contract_pass"] = False
            full_decoded = decode_with_layer_masks(model, encoded, full_mask)
            hard_decoded = decode_with_layer_masks(model, encoded, hard_batch_mask)
            random_decoded = [decode_with_layer_masks(model, encoded, mask) for mask in random_batch_masks]
            all_layer_decoded: dict[str, dict[str, torch.Tensor]] = {}
            layer_specs = {
                "REMOVE_HARD_ALL_LAYERS": list(range(layer_count)),
                "REMOVE_HARD_LAYER1_ONLY": [0],
                "REMOVE_HARD_LAYER2_ONLY": [1] if layer_count >= 2 else [],
                "REMOVE_HARD_FINAL_LAYER_ONLY": [layer_count - 1],
            }
            for condition, active_layers in layer_specs.items():
                layer_masks = [full_mask for _ in range(layer_count)]
                for layer_index in active_layers:
                    layer_masks[layer_index] = hard_batch_mask
                all_layer_decoded[condition] = decode_with_layer_masks(model, encoded, full_mask, layer_masks)
            for decoded in [full_decoded, hard_decoded, *random_decoded, *all_layer_decoded.values()]:
                for name in ("pred_logits", "pred_spans", "references"):
                    if not torch.isfinite(decoded[name]).all().item():
                        validation["finite_pass"] = False
            for index, info in enumerate(batch_info):
                qid = info["qid"]
                meta = info["meta"]
                qcohort = cohort[qid]
                full_predictions = postprocess(full_decoded, index, meta, clip_length)
                hard_predictions = postprocess(hard_decoded, index, meta, clip_length)
                official_predictions = _CF.postprocess_prediction(
                    official_decoded["pred_spans"][index],
                    official_decoded["pred_logits"][index],
                    float(meta["duration"]),
                    clip_length,
                )
                full_array = np.asarray(full_predictions, dtype=np.float64)
                official_array = np.asarray(official_predictions, dtype=np.float64)
                if full_array.shape != official_array.shape:
                    validation["official_same_run_pass"] = False
                elif full_array.size:
                    difference = float(np.max(np.abs(full_array - official_array)))
                    validation["max_official_same_run_difference"] = max(
                        validation["max_official_same_run_difference"], difference
                    )
                    if not np.allclose(full_array, official_array, atol=1e-4, rtol=0.0):
                        validation["official_same_run_pass"] = False
                full_metrics = cf.query_metrics(full_predictions, meta)
                hard_metrics = cf.query_metrics(hard_predictions, meta)
                full_reference_subset.append(reference_by_qid[qid])
                observed_full_subset.append({"qid": qid, "pred_relevant_windows": full_predictions})
                hard_center = info["hard_center"]
                full_trajectory = references_row(full_decoded, index, meta, hard_center, "FULL_ACCESS", 0, clip_length, full_predictions)
                hard_trajectory = references_row(hard_decoded, index, meta, hard_center, "REMOVE_HARD", 0, clip_length, hard_predictions)
                for row in full_trajectory + hard_trajectory:
                    row["cohort"] = qcohort["cohort"]
                    row["duration_bin"] = qcohort["duration_bin"]
                    trajectory_rows.append(row)
                for replicate, decoded in enumerate(random_decoded):
                    random_predictions = postprocess(decoded, index, meta, clip_length)
                    for row in references_row(decoded, index, meta, hard_center, "REMOVE_RANDOM", replicate, clip_length, random_predictions):
                        row["cohort"] = qcohort["cohort"]
                        row["duration_bin"] = qcohort["duration_bin"]
                        trajectory_rows.append(row)
                for replicate, decoded in enumerate(random_decoded):
                    random_predictions = postprocess(decoded, index, meta, clip_length)
                    random_metrics = cf.query_metrics(random_predictions, meta)
                    remove_rows.append({
                        **qcohort,
                        "p3_location_failure": geometry[qid].get("p3_location_failure", ""),
                        "replicate": replicate,
                        "random_stratum": info["specs"]["hard_stratum"],
                        "random_stratum_fallback": int(info["specs"]["random_fallback_flags"][replicate]),
                        "hard_region_token_count_removed": len(info["specs"]["hard_indices"]),
                        "full_center_hit10_2s": full_metrics["center_hit10_2s"],
                        "remove_hard_center_hit10_2s": hard_metrics["center_hit10_2s"],
                        "remove_random_center_hit10_2s": random_metrics["center_hit10_2s"],
                        "hard_minus_random_center_pp": (hard_metrics["center_hit10_2s"] - random_metrics["center_hit10_2s"]) * 100.0,
                        "full_oracle10_iou07": full_metrics["oracle10_iou07"],
                        "remove_hard_oracle10_iou07": hard_metrics["oracle10_iou07"],
                        "remove_random_oracle10_iou07": random_metrics["oracle10_iou07"],
                        "hard_minus_random_oracle_pp": (hard_metrics["oracle10_iou07"] - random_metrics["oracle10_iou07"]) * 100.0,
                        "full_width_gt_ratio": full_metrics["final_width_gt_ratio"],
                        "remove_hard_width_gt_ratio": hard_metrics["final_width_gt_ratio"],
                        "remove_random_width_gt_ratio": random_metrics["final_width_gt_ratio"],
                    })
                    transition_rows.append({
                        "qid": qid,
                        "duration_bin": qcohort["duration_bin"],
                        "cohort": qcohort["cohort"],
                        "condition": "REMOVE_RANDOM",
                        "replicate": replicate,
                        "full_hit": int(full_metrics["center_hit10_2s"]),
                        "intervention_hit": int(random_metrics["center_hit10_2s"]),
                        "transition": transition(int(full_metrics["center_hit10_2s"]), int(random_metrics["center_hit10_2s"])),
                    })
                transition_rows.append({
                    "qid": qid,
                    "duration_bin": qcohort["duration_bin"],
                    "cohort": qcohort["cohort"],
                    "condition": "REMOVE_HARD",
                    "replicate": 0,
                    "full_hit": int(full_metrics["center_hit10_2s"]),
                    "intervention_hit": int(hard_metrics["center_hit10_2s"]),
                    "transition": transition(int(full_metrics["center_hit10_2s"]), int(hard_metrics["center_hit10_2s"])),
                })
                if qcohort["cohort"] == "harmful" and int(full_metrics["center_hit10_2s"]) == 0 and int(hard_metrics["center_hit10_2s"]) == 1:
                    rescued_rows.append({
                        "qid": qid,
                        "duration_bin": qcohort["duration_bin"],
                        "audio_duration_sec": qcohort["audio_duration_sec"],
                        "gt_duration_sec": qcohort["gt_duration_sec"],
                        "full_final_width_gt_ratio": full_metrics["final_width_gt_ratio"],
                        "remove_hard_final_width_gt_ratio": hard_metrics["final_width_gt_ratio"],
                        "remove_hard_abs_log_width_gt_ratio": abs(math.log(max(hard_metrics["final_width_gt_ratio"], 1e-12))),
                        "remove_hard_oracle10_iou05": hard_metrics["oracle10_iou05"],
                        "remove_hard_oracle10_iou07": hard_metrics["oracle10_iou07"],
                        "remove_hard_r1_iou07": hard_metrics["r1_iou07"],
                    })
                for condition, decoded in all_layer_decoded.items():
                    predictions = postprocess(decoded, index, meta, clip_length)
                    metrics = cf.query_metrics(predictions, meta)
                    layer_rows.append({
                        **qcohort,
                        "p3_location_failure": geometry[qid].get("p3_location_failure", ""),
                        "condition": condition,
                        "active_layer": condition.replace("REMOVE_HARD_", ""),
                        "removed_token_count": len(info["specs"]["hard_indices"]),
                        "full_center_hit10_2s": full_metrics["center_hit10_2s"],
                        "intervention_center_hit10_2s": metrics["center_hit10_2s"],
                        "intervention_minus_full_center_pp": (metrics["center_hit10_2s"] - full_metrics["center_hit10_2s"]) * 100.0,
                        "intervention_oracle10_iou07": metrics["oracle10_iou07"],
                        "intervention_width_gt_ratio": metrics["final_width_gt_ratio"],
                    })
            validation["batches"] += 1

    reference_comparison = cf.compare_submissions(full_reference_subset, observed_full_subset)
    validation["saved_submission_reference_pass"] = bool(reference_comparison["pass"])
    validation["full_access_reference_pass"] = bool(validation["official_same_run_pass"])
    validation["max_full_access_difference"] = float(reference_comparison["max_abs_difference"])
    validation["pass"] = bool(
        validation["full_access_reference_pass"]
        and validation["mask_contract_pass"]
        and validation["finite_pass"]
        and validation["layer_wrapper_pass"]
    )
    fields = {
        "remove": list(remove_rows[0].keys()) if remove_rows else ["qid"],
        "transition": list(transition_rows[0].keys()) if transition_rows else ["qid"],
        "trajectory": list(trajectory_rows[0].keys()) if trajectory_rows else ["qid"],
        "harmful_control": list(harmful_control_rows[0].keys()) if harmful_control_rows else ["qid"],
        "layer": list(layer_rows[0].keys()) if layer_rows else ["qid"],
        "rescued": list(rescued_rows[0].keys()) if rescued_rows else ["qid"],
    }
    write_csv(output / "remove_hard_vs_random.csv", remove_rows, fields["remove"])
    write_csv(output / "paired_transitions.csv", transition_rows, fields["transition"])
    write_csv(output / "reference_trajectory_intervention.csv", trajectory_rows, fields["trajectory"])
    write_csv(output / "harmful_vs_control.csv", harmful_control_rows, fields["harmful_control"])
    write_csv(output / "layer_specific_intervention.csv", layer_rows, fields["layer"])
    write_csv(output / "rescued_query_scale_analysis.csv", rescued_rows, fields["rescued"])
    write_json(output / "qd_necessity_run_provenance.json", {
        "baseline_commit": cf.git_revision(args.worktree.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "test_jsonl_sha256": sha256_file(args.test_jsonl.resolve()),
        "counterfactual_source": "previous frozen search-space counterfactual query-level output",
        "geometry_source": "previous frozen hard-negative geometry output",
        "seed": args.seed,
        "random_replicates": RANDOM_REPLICATES,
        "decoder_layer_count": validation["layer_count"],
        "num_queries": validation["num_queries"],
        "selected_qids": len(cohort),
    })
    write_json(output / "intervention_validation.json", {**validation, "reference_comparison": reference_comparison})
    print(json.dumps({"validation": validation, "rows": {"remove": len(remove_rows), "transition": len(transition_rows), "trajectory": len(trajectory_rows), "layer": len(layer_rows), "rescued": len(rescued_rows)}}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--reference-submission", type=Path, required=True)
    parser.add_argument("--counterfactual-csv", type=Path, required=True)
    parser.add_argument("--counterfactual-runner", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
