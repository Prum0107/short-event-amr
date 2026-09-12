#!/usr/bin/env python3
"""Run the registered coordinate-preserving decoder-access counterfactual.

The runner executes the unchanged QD-DETR encoder once per batch and calls the
unchanged decoder with different ``memory_key_padding_mask`` values. It does
not modify model parameters, temporal coordinates, encoder masks, or input
features. The script is intentionally self-contained so that the run can be
audited independently of the public release's omitted model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader


CONDITIONS = (
    "GT_ONLY",
    "RANDOM_25",
    "RANDOM_50",
    "RANDOM_100",
    "HARD_25",
    "HARD_50",
    "HARD_100",
    "FULL_ACCESS",
)
RANDOM_REPLICATES = 10
PRIMARY_BINS = ("0-2s", "2-5s")
ALL_BINS = ("0-2s", "2-5s", "5-10s", "10-20s", "20s+")
FRACTIONS = {"25": 0.25, "50": 0.50, "100": 1.00}


class InvalidCounterfactual(RuntimeError):
    """Raised when a required counterfactual invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise InvalidCounterfactual(f"non-finite tensor: {name}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def git_revision(worktree: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def inverse_sigmoid(value: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    value = value.clamp(min=0, max=1)
    x1 = value.clamp(min=eps)
    x2 = (1 - value).clamp(min=eps)
    return torch.log(x1 / x2)


def duration_bin(gt_duration: float) -> str:
    if gt_duration < 2:
        return "0-2s"
    if gt_duration < 5:
        return "2-5s"
    if gt_duration < 10:
        return "5-10s"
    if gt_duration < 20:
        return "10-20s"
    return "20s+"


def longest_gt(meta: Mapping[str, Any]) -> float:
    return max(float(end) - float(start) for start, end in meta["relevant_windows"])


def qid_integer(qid: Any) -> int:
    try:
        return int(qid)
    except (TypeError, ValueError):
        return sum((index + 1) * ord(char) for index, char in enumerate(str(qid)))


def derived_seed(base_seed: int, qid: Any, replicate: int) -> int:
    return int(base_seed + qid_integer(qid) * 1_000_003 + replicate)


def gt_token_mask(
    meta: Mapping[str, Any],
    valid: np.ndarray,
    clip_length: float,
) -> np.ndarray:
    """Return valid tokens whose half-open temporal support overlaps any GT.

    For token ``i``, the support is ``[i * clip_length, (i + 1) *
    clip_length)``. This is the existing one-second temporal-grid convention
    used by the duration audit and retains the union across multiple GT
    intervals.
    """

    positions = np.arange(len(valid), dtype=np.float64) * clip_length
    ends = positions + clip_length
    mask = np.zeros(len(valid), dtype=bool)
    for start, end in meta["relevant_windows"]:
        start_f, end_f = float(start), float(end)
        mask |= (positions < end_f) & (ends > start_f)
    mask &= valid
    if not mask.any() and valid.any():
        # The official label path clamps an endpoint-only interval to the
        # final valid token; keep that existing grid convention without adding
        # a tunable temporal margin.
        first_start = float(meta["relevant_windows"][0][0])
        fallback = min(max(int(first_start / clip_length), 0), int(valid.sum()) - 1)
        mask[fallback] = True
    return mask


def postprocess_prediction(
    pred_spans: torch.Tensor,
    pred_logits: torch.Tensor,
    duration: float,
    clip_length: float,
) -> list[list[float]]:
    """Apply the official evaluation conversion without changing its rules.

    The official evaluator serializes raw span rows to Python floats rounded
    to four decimals before ``PostProcessorDETR`` clips and rounds timestamps.
    Reproduce that ordering so half-precision edge cases match the reference
    submission exactly.
    """

    from span_utils import span_cxw_to_xx

    spans = span_cxw_to_xx(pred_spans.detach().cpu()) * float(duration)
    scores = F.softmax(pred_logits.detach().cpu(), -1)[..., 0]
    rows = torch.cat([spans, scores[:, None]], dim=1).tolist()
    rows = sorted(rows, key=lambda row: row[2], reverse=True)
    rows = [[float(f"{value:.4f}") for value in row] for row in rows]
    windows_and_scores = torch.tensor(rows)
    windows = windows_and_scores[:, :2]
    windows = torch.clamp(windows, min=0, max=300)
    windows = torch.round(windows / clip_length) * clip_length
    processed = torch.cat([windows, windows_and_scores[:, 2:3]], dim=1).tolist()
    return [row[:2] + [float(f"{row[2]:.4f}")] for row in processed]


def temporal_iou(first: Sequence[float], second: Sequence[float]) -> float:
    start = max(float(first[0]), float(second[0]))
    end = min(float(first[1]), float(second[1]))
    intersection = max(0.0, end - start)
    union = max(0.0, float(first[1]) - float(first[0])) + max(0.0, float(second[1]) - float(second[0])) - intersection
    return intersection / union if union > 0 else 0.0


def query_metrics(predictions: Sequence[Sequence[float]], meta: Mapping[str, Any]) -> dict[str, float]:
    gt_windows = [[float(start), float(end)] for start, end in meta["relevant_windows"]]
    gt_centers = [(start + end) / 2.0 for start, end in gt_windows]
    ious = [max(temporal_iou(prediction, gt) for gt in gt_windows) for prediction in predictions]
    centers = [(float(prediction[0]) + float(prediction[1])) / 2.0 for prediction in predictions]
    center_errors = [min(abs(center - gt_center) for gt_center in gt_centers) for center in centers]
    gt_duration = longest_gt(meta)
    top10 = list(predictions[:10])
    top10_ious = ious[:10]
    return {
        "center_hit10_1s": float(any(error <= 1.0 for error in center_errors[:10])),
        "center_hit10_2s": float(any(error <= 2.0 for error in center_errors[:10])),
        "positive_overlap_candidate_top10": float(any(iou > 0.0 for iou in top10_ious)),
        "oracle10_iou05": float(max(top10_ious, default=0.0) >= 0.5),
        "oracle10_iou07": float(max(top10_ious, default=0.0) >= 0.7),
        "r1_iou05": float(ious[0] >= 0.5),
        "r1_iou07": float(ious[0] >= 0.7),
        "top1_iou07_success": float(ious[0] >= 0.7),
        "final_center_error_sec": float(center_errors[0]),
        "final_width_gt_ratio": float(max(0.0, float(predictions[0][1]) - float(predictions[0][0])) / gt_duration),
        "top1_iou": float(ious[0]),
        "oracle10_best_iou": float(max(top10_ious, default=0.0)),
    }


def load_options(config_path: Path, worktree: Path, output: Path, device: str) -> EasyDict:
    from config import BaseOptions

    manager = BaseOptions(str(config_path))
    manager.parse()
    option = EasyDict(dict(manager.option))
    option.results_dir = str(output / "runtime")
    option.eval_split_name = "test"
    option.test_path = str(worktree / "data/castella_test_release.jsonl")
    option.a_feat_dir = str(worktree / "features/castella/clap")
    option.t_feat_dir = str(worktree / "features/castella/clap_text")
    option.device = device
    return option


def build_dataset(option: EasyDict):
    from dataset import StartEndDataset

    return StartEndDataset(
        data_path=option.test_path,
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


def encode_full_audio(model: torch.nn.Module, model_inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Execute the official QD-DETR path through the frozen second encoder."""

    src_txt = model.input_txt_proj(model_inputs["src_txt"])
    src_aud = model.input_aud_proj(model_inputs["src_aud"])
    src = torch.cat([src_aud, src_txt], dim=1)
    valid_mask = torch.cat([model_inputs["src_aud_mask"], model_inputs["src_txt_mask"]], dim=1).bool()

    pos_aud = model.position_embed(src_aud, model_inputs["src_aud_mask"])
    pos_txt = torch.zeros_like(src_txt)
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
    src = model.transformer.t2v_encoder(
        src,
        src_key_padding_mask=source_padding,
        pos=pos,
        audio_length=audio_length,
    )
    src = src[: audio_length + 1]
    source_padding = source_padding[:, : audio_length + 1]
    pos = pos[: audio_length + 1]
    memory = model.transformer.encoder(src, src_key_padding_mask=source_padding, pos=pos)
    memory_global = memory[0]
    memory_local = memory[1:]
    position_local = pos[1:]
    decoder_padding_full = source_padding[:, 1:]
    return {
        "memory_global": memory_global,
        "memory_local": memory_local,
        "position_local": position_local,
        "decoder_padding_full": decoder_padding_full,
        "valid_audio": ~decoder_padding_full,
    }


def decode_with_access_mask(
    model: torch.nn.Module,
    encoded: Mapping[str, torch.Tensor],
    decoder_padding: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run only the official decoder with a supplied key-padding mask."""

    memory_local = encoded["memory_local"]
    batch_size = memory_local.shape[1]
    query_embed = model.query_embed.weight
    refpoint_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
    target = torch.zeros(
        refpoint_embed.shape[0], batch_size, memory_local.shape[2], device=memory_local.device
    )
    hidden_states, references = model.transformer.decoder(
        target,
        memory_local,
        memory_key_padding_mask=decoder_padding,
        pos=encoded["position_local"],
        refpoints_unsigmoid=refpoint_embed,
    )
    outputs_class = model.class_embed(hidden_states)
    coordinates = model.span_embed(hidden_states) + inverse_sigmoid(references)
    if model.span_loss_type == "l1":
        coordinates = coordinates.sigmoid()
    return {
        "pred_logits": outputs_class[-1],
        "pred_spans": coordinates[-1],
        "initial_references": refpoint_embed.sigmoid(),
    }


def condition_specs_for_batch(
    metas: Sequence[Mapping[str, Any]],
    valid_audio: torch.Tensor,
    saliency: torch.Tensor,
    clip_length: float,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Create all registered masks before any reduced-access result is read."""

    valid_cpu = valid_audio.detach().cpu().numpy().astype(bool)
    saliency_cpu = saliency.detach().cpu().numpy()
    masks: list[dict[str, Any]] = []
    for batch_index, meta in enumerate(metas):
        valid = valid_cpu[batch_index]
        gt = gt_token_mask(meta, valid, clip_length)
        non_gt_indices = np.flatnonzero(valid & ~gt)
        if not gt.any():
            raise InvalidCounterfactual(f"no valid GT token for qid={meta['qid']}")

        def make_mask(keep_indices: Iterable[int]) -> torch.Tensor:
            keep = np.zeros(len(valid), dtype=bool)
            keep[gt] = True
            keep[list(keep_indices)] = True
            if not keep.any():
                raise InvalidCounterfactual(f"all decoder keys masked for qid={meta['qid']}")
            return torch.from_numpy(~keep)

        base = {
            "qid": meta["qid"],
            "gt_token_count": int(gt.sum()),
            "valid_token_count": int(valid.sum()),
            "non_gt_token_count": int(len(non_gt_indices)),
        }
        masks.append({**base, "condition": "GT_ONLY", "replicate": 0, "random_seed": None, "mask": make_mask([])})
        for condition in ("RANDOM_25", "RANDOM_50"):
            fraction = FRACTIONS[condition.rsplit("_", 1)[1]]
            for replicate in range(RANDOM_REPLICATES):
                seed = derived_seed(base_seed, meta["qid"], replicate)
                permutation = np.random.default_rng(seed).permutation(non_gt_indices)
                count = min(len(non_gt_indices), int(math.ceil(fraction * len(non_gt_indices))))
                selected = permutation[:count]
                masks.append({
                    **base,
                    "condition": condition,
                    "replicate": replicate,
                    "random_seed": seed,
                    "mask": make_mask(selected),
                })
        masks.append({
            **base,
            "condition": "RANDOM_100",
            "replicate": 0,
            "random_seed": None,
            "mask": make_mask(non_gt_indices),
        })
        hard_order = sorted(
            (int(index) for index in non_gt_indices),
            key=lambda index: (-float(saliency_cpu[batch_index, index]), index),
        )
        for condition in ("HARD_25", "HARD_50", "HARD_100"):
            fraction = FRACTIONS[condition.rsplit("_", 1)[1]]
            count = min(len(non_gt_indices), int(math.ceil(fraction * len(non_gt_indices))))
            masks.append({
                **base,
                "condition": condition,
                "replicate": 0,
                "random_seed": None,
                "mask": make_mask(hard_order[:count]),
            })
        masks.append({
            **base,
            "condition": "FULL_ACCESS",
            "replicate": 0,
            "random_seed": None,
            "mask": torch.from_numpy(~valid),
        })

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in masks:
        grouped[(str(item["condition"]), int(item["replicate"]))].append(item)
    specs: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        replicates = range(RANDOM_REPLICATES) if condition in {"RANDOM_25", "RANDOM_50"} else range(1)
        for replicate in replicates:
            entries = grouped[(condition, replicate)]
            mask = torch.stack([entry["mask"] for entry in entries], dim=0).to(valid_audio.device)
            specs.append({
                "condition": condition,
                "replicate": replicate,
                "random_seed": entries[0]["random_seed"],
                "mask": mask,
                "entries": entries,
            })
    return specs


def assert_mask_contract(
    specs: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, torch.Tensor],
    gt_by_batch: Sequence[np.ndarray],
) -> None:
    full_padding = encoded["decoder_padding_full"]
    valid = encoded["valid_audio"]
    for spec in specs:
        padding = spec["mask"]
        if padding.shape != full_padding.shape:
            raise InvalidCounterfactual("decoder mask shape changed")
        extra_mask = padding & full_padding.logical_not()
        if torch.any(extra_mask & valid.logical_not()):
            raise InvalidCounterfactual("a non-valid position was treated as a new distractor")
        if torch.any((~padding) & (~valid)):
            raise InvalidCounterfactual("a padded token became accessible")
        for batch_index, gt in enumerate(gt_by_batch):
            if torch.any(padding[batch_index].cpu() & torch.from_numpy(gt)):
                raise InvalidCounterfactual("a GT token was masked")


def compare_submissions(reference: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ref_by_qid = {str(row["qid"]): row for row in reference}
    obs_by_qid = {str(row["qid"]): row for row in observed}
    missing = sorted(set(ref_by_qid) - set(obs_by_qid))
    extra = sorted(set(obs_by_qid) - set(ref_by_qid))
    mismatches = 0
    max_abs = 0.0
    mismatch_qids: list[str] = []
    mismatch_examples: dict[str, Any] = {}
    for qid in sorted(set(ref_by_qid) & set(obs_by_qid)):
        ref_windows = np.asarray(ref_by_qid[qid]["pred_relevant_windows"], dtype=np.float64)
        obs_windows = np.asarray(obs_by_qid[qid]["pred_relevant_windows"], dtype=np.float64)
        if ref_windows.shape != obs_windows.shape:
            mismatches += 1
            mismatch_qids.append(qid)
            mismatch_examples[qid] = {"reference": ref_by_qid[qid]["pred_relevant_windows"], "observed": obs_by_qid[qid]["pred_relevant_windows"]}
            continue
        if ref_windows.size:
            max_abs = max(max_abs, float(np.max(np.abs(ref_windows - obs_windows))))
        if not np.allclose(ref_windows, obs_windows, atol=1e-4, rtol=0.0):
            mismatches += 1
            mismatch_qids.append(qid)
            mismatch_examples[qid] = {"reference": ref_by_qid[qid]["pred_relevant_windows"], "observed": obs_by_qid[qid]["pred_relevant_windows"]}
    return {
        "pass": not missing and not extra and mismatches == 0,
        "reference_count": len(reference),
        "observed_count": len(observed),
        "missing_qids": missing[:20],
        "extra_qids": extra[:20],
        "mismatch_count": mismatches,
        "mismatch_qids": mismatch_qids[:20],
        "mismatch_examples": {qid: mismatch_examples[qid] for qid in mismatch_qids[:2]},
        "max_abs_difference": max_abs,
    }


METRIC_FIELDS = (
    "center_hit10_1s",
    "center_hit10_2s",
    "positive_overlap_candidate_top10",
    "oracle10_iou05",
    "oracle10_iou07",
    "r1_iou05",
    "r1_iou07",
    "top1_iou07_success",
    "final_center_error_sec",
    "final_width_gt_ratio",
)


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    result: dict[str, Any] = {
        "N_rows": len(rows),
        "N_queries": len({str(row["qid"]) for row in rows}),
        "replicate_count": len({int(row["replicate"]) for row in rows}),
    }
    for field in METRIC_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result[f"mean_{field}_pct"] = float(np.mean(values) * 100.0) if field not in {"final_center_error_sec", "final_width_gt_ratio"} else None
        if field in {"final_center_error_sec", "final_width_gt_ratio"}:
            result[f"median_{field}"] = float(np.median(values))
    for field in ("effective_accessible_tokens", "effective_distractor_count", "gt_token_count", "valid_token_count"):
        result[f"mean_{field}"] = float(np.mean([float(row[field]) for row in rows]))
    return result


def metric_percent(aggregate: Mapping[str, Any], field: str) -> float:
    return float(aggregate.get(f"mean_{field}_pct", float("nan")))


def group_rows(rows: Sequence[Mapping[str, Any]], condition: str, bin_name: str, replicate: int | None = None) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row["condition"] == condition
        and row["duration_bin"] == bin_name
        and (replicate is None or int(row["replicate"]) == replicate)
    ]


def condition_metric_rows(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "condition", "duration_bin", "N_queries", "N_rows", "replicate_count",
        "mean_center_hit10_1s_pct", "mean_center_hit10_2s_pct",
        "mean_positive_overlap_candidate_top10_pct", "mean_oracle10_iou05_pct",
        "mean_oracle10_iou07_pct", "mean_r1_iou05_pct", "mean_r1_iou07_pct",
        "median_final_center_error_sec", "median_final_width_gt_ratio",
        "mean_effective_accessible_tokens", "mean_effective_distractor_count",
        "mean_gt_token_count", "mean_valid_token_count",
    ]
    output: list[dict[str, Any]] = []
    for bin_name in ALL_BINS:
        for condition in CONDITIONS:
            selected = group_rows(all_rows, condition, bin_name)
            if not selected:
                continue
            aggregate = aggregate_rows(selected)
            record = {field: aggregate.get(field, "") for field in fields if field not in {"condition", "duration_bin"}}
            record.update({"condition": condition, "duration_bin": bin_name})
            output.append(record)
    return output


def random_variability_rows(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in ("RANDOM_25", "RANDOM_50"):
        for bin_name in ALL_BINS:
            for replicate in range(RANDOM_REPLICATES):
                selected = group_rows(all_rows, condition, bin_name, replicate)
                if not selected:
                    continue
                aggregate = aggregate_rows(selected)
                output.append({
                    "condition": condition,
                    "duration_bin": bin_name,
                    "replicate": replicate,
                    "random_seed": selected[0]["random_seed"],
                    "N_queries": aggregate["N_queries"],
                    "center_hit10_1s_pct": metric_percent(aggregate, "center_hit10_1s"),
                    "center_hit10_2s_pct": metric_percent(aggregate, "center_hit10_2s"),
                    "positive_overlap_candidate_top10_pct": metric_percent(aggregate, "positive_overlap_candidate_top10"),
                    "oracle10_iou05_pct": metric_percent(aggregate, "oracle10_iou05"),
                    "oracle10_iou07_pct": metric_percent(aggregate, "oracle10_iou07"),
                    "r1_iou05_pct": metric_percent(aggregate, "r1_iou05"),
                    "r1_iou07_pct": metric_percent(aggregate, "r1_iou07"),
                    "median_final_center_error_sec": aggregate["median_final_center_error_sec"],
                    "median_final_width_gt_ratio": aggregate["median_final_width_gt_ratio"],
                })
    return output


def query_mean_metric(rows: Sequence[Mapping[str, Any]], condition: str, bin_name: str, field: str) -> dict[str, float]:
    selected = group_rows(rows, condition, bin_name)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        grouped[str(row["qid"])].append(float(row[field]))
    return {qid: float(np.mean(values)) for qid, values in grouped.items()}


def random_vs_hard_rows(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for bin_name in PRIMARY_BINS:
        for level in ("25", "50", "100"):
            random_condition = f"RANDOM_{level}"
            hard_condition = f"HARD_{level}"
            random_center = query_mean_metric(all_rows, random_condition, bin_name, "center_hit10_2s")
            hard_center = query_mean_metric(all_rows, hard_condition, bin_name, "center_hit10_2s")
            random_oracle = query_mean_metric(all_rows, random_condition, bin_name, "oracle10_iou07")
            hard_oracle = query_mean_metric(all_rows, hard_condition, bin_name, "oracle10_iou07")
            qids = sorted(set(random_center) & set(hard_center))
            if not qids:
                continue
            output.append({
                "duration_bin": bin_name,
                "level": level,
                "random_condition": random_condition,
                "hard_condition": hard_condition,
                "N_queries": len(qids),
                "random_center_hit10_2s_pct": float(np.mean([random_center[qid] for qid in qids]) * 100),
                "hard_center_hit10_2s_pct": float(np.mean([hard_center[qid] for qid in qids]) * 100),
                "hard_minus_random_center_pp": float(np.mean([hard_center[qid] - random_center[qid] for qid in qids]) * 100),
                "random_oracle10_iou07_pct": float(np.mean([random_oracle[qid] for qid in qids]) * 100),
                "hard_oracle10_iou07_pct": float(np.mean([hard_oracle[qid] for qid in qids]) * 100),
                "hard_minus_random_oracle_pp": float(np.mean([hard_oracle[qid] - random_oracle[qid] for qid in qids]) * 100),
            })
    return output


def dose_response_rows(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    levels = (
        ("GT_ONLY", "0", "RESTRICTED"),
        ("RANDOM_25", "25", "RANDOM"),
        ("RANDOM_50", "50", "RANDOM"),
        ("RANDOM_100", "100", "RANDOM"),
        ("HARD_25", "25", "HARD"),
        ("HARD_50", "50", "HARD"),
        ("HARD_100", "100", "HARD"),
        ("FULL_ACCESS", "FULL", "FULL"),
    )
    for bin_name in PRIMARY_BINS:
        full = aggregate_rows(group_rows(all_rows, "FULL_ACCESS", bin_name))
        for condition, level, family in levels:
            selected = group_rows(all_rows, condition, bin_name)
            if not selected:
                continue
            aggregate = aggregate_rows(selected)
            output.append({
                "duration_bin": bin_name,
                "family": family,
                "level": level,
                "condition": condition,
                "N_rows": aggregate["N_rows"],
                "N_queries": aggregate["N_queries"],
                "mean_effective_distractor_count": aggregate["mean_effective_distractor_count"],
                "mean_center_hit10_2s_pct": metric_percent(aggregate, "center_hit10_2s"),
                "delta_vs_full_center_pp": metric_percent(aggregate, "center_hit10_2s") - metric_percent(full, "center_hit10_2s"),
                "mean_oracle10_iou07_pct": metric_percent(aggregate, "oracle10_iou07"),
                "delta_vs_full_oracle_pp": metric_percent(aggregate, "oracle10_iou07") - metric_percent(full, "oracle10_iou07"),
            })
    return output


def transition_rows(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    conditions = tuple(condition for condition in CONDITIONS if condition != "FULL_ACCESS")
    for bin_name in PRIMARY_BINS:
        full_rows = {str(row["qid"]): row for row in group_rows(all_rows, "FULL_ACCESS", bin_name)}
        for condition in conditions:
            selected = group_rows(all_rows, condition, bin_name)
            for metric, label in (("center_hit10_2s", "CenterHit10<=2s"), ("top1_iou07_success", "Top1IoU>=0.7")):
                transitions = Counter()
                for row in selected:
                    qid = str(row["qid"])
                    if qid in full_rows:
                        transitions[(int(full_rows[qid][metric]), int(row[metric]))] += 1
                total = sum(transitions.values())
                for (from_state, to_state), count in sorted(transitions.items()):
                    output.append({
                        "duration_bin": bin_name,
                        "condition": condition,
                        "metric": label,
                        "from_state": from_state,
                        "to_state": to_state,
                        "N_rows": count,
                        "pct_of_condition_rows": float(count / total * 100) if total else 0.0,
                    })
    return output


def duration_rescue_rows(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for bin_name in PRIMARY_BINS:
        primary = [row for row in all_rows if row["duration_bin"] == bin_name and int(row["replicate"]) == 0]
        durations = [float(row["audio_duration_sec"]) for row in primary if row["condition"] == "FULL_ACCESS"]
        if not durations:
            continue
        threshold = float(np.median(durations))
        group_by_qid = {
            str(row["qid"]): "shorter_or_equal_audio" if float(row["audio_duration_sec"]) <= threshold else "longer_audio"
            for row in primary if row["condition"] == "FULL_ACCESS"
        }
        for condition in CONDITIONS:
            for recording_group in ("shorter_or_equal_audio", "longer_audio"):
                selected = [
                    row for row in all_rows
                    if row["duration_bin"] == bin_name
                    and row["condition"] == condition
                    and group_by_qid.get(str(row["qid"])) == recording_group
                ]
                full = [
                    row for row in all_rows
                    if row["duration_bin"] == bin_name
                    and row["condition"] == "FULL_ACCESS"
                    and group_by_qid.get(str(row["qid"])) == recording_group
                ]
                if not selected or not full:
                    continue
                selected_aggregate = aggregate_rows(selected)
                full_aggregate = aggregate_rows(full)
                output.append({
                    "duration_bin": bin_name,
                    "recording_duration_group": recording_group,
                    "recording_duration_median_split_sec": threshold,
                    "condition": condition,
                    "N_queries": selected_aggregate["N_queries"],
                    "full_center_hit10_2s_pct": metric_percent(full_aggregate, "center_hit10_2s"),
                    "condition_center_hit10_2s_pct": metric_percent(selected_aggregate, "center_hit10_2s"),
                    "delta_center_pp": metric_percent(selected_aggregate, "center_hit10_2s") - metric_percent(full_aggregate, "center_hit10_2s"),
                    "full_oracle10_iou07_pct": metric_percent(full_aggregate, "oracle10_iou07"),
                    "condition_oracle10_iou07_pct": metric_percent(selected_aggregate, "oracle10_iou07"),
                    "delta_oracle_pp": metric_percent(selected_aggregate, "oracle10_iou07") - metric_percent(full_aggregate, "oracle10_iou07"),
                })
    return output


def assess_hypotheses(
    all_rows: Sequence[Mapping[str, Any]],
    hard_random: Sequence[Mapping[str, Any]],
    dose: Sequence[Mapping[str, Any]],
    rescue: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def mean_metric(condition: str, bin_name: str, field: str) -> float:
        selected = group_rows(all_rows, condition, bin_name)
        return float(np.mean([float(row[field]) for row in selected])) if selected else float("nan")

    h1_values = []
    h5_values = []
    for bin_name in PRIMARY_BINS:
        full_center = mean_metric("FULL_ACCESS", bin_name, "center_hit10_2s")
        gt_center = mean_metric("GT_ONLY", bin_name, "center_hit10_2s")
        full_positive = mean_metric("FULL_ACCESS", bin_name, "positive_overlap_candidate_top10")
        gt_positive = mean_metric("GT_ONLY", bin_name, "positive_overlap_candidate_top10")
        h1_values.append(gt_center > full_center and gt_positive >= full_positive)
        gt_width = np.median([float(row["final_width_gt_ratio"]) for row in group_rows(all_rows, "GT_ONLY", bin_name)])
        gt_r1 = mean_metric("GT_ONLY", bin_name, "r1_iou07")
        h5_values.append(gt_width > 2.0 and gt_r1 < 0.5)

    hard_random_values = []
    for bin_name in PRIMARY_BINS:
        # At level 100 both conditions expose the complete valid non-GT set,
        # so RANDOM_100 and HARD_100 are a structural identity control and
        # cannot support a strict HARD<RANDOM comparison. The informative
        # matched-count contrasts are 25 and 50.
        selected = [
            row for row in hard_random
            if row["duration_bin"] == bin_name and str(row["level"]) in {"25", "50"}
        ]
        hard_random_values.append(bool(selected) and all(float(row["hard_minus_random_center_pp"]) < 0 for row in selected))

    random_dose_values = []
    for bin_name in PRIMARY_BINS:
        selected = [row for row in dose if row["duration_bin"] == bin_name and row["family"] == "RANDOM"]
        by_level = {str(row["level"]): float(row["mean_center_hit10_2s_pct"]) for row in selected}
        ordered = [by_level[level] for level in ("25", "50", "100") if level in by_level]
        gt = mean_metric("GT_ONLY", bin_name, "center_hit10_2s") * 100
        full = mean_metric("FULL_ACCESS", bin_name, "center_hit10_2s") * 100
        random_dose_values.append(len(ordered) == 3 and gt >= ordered[0] >= ordered[1] >= ordered[2] >= full)

    rescue_values = []
    for bin_name in PRIMARY_BINS:
        selected = [row for row in rescue if row["duration_bin"] == bin_name and row["condition"] == "GT_ONLY"]
        by_group = {row["recording_duration_group"]: float(row["delta_center_pp"]) for row in selected}
        rescue_values.append(
            "longer_audio" in by_group and "shorter_or_equal_audio" in by_group
            and by_group["longer_audio"] > by_group["shorter_or_equal_audio"]
            and by_group["longer_audio"] > 0
        )

    def status(values: Sequence[bool]) -> str:
        if values and all(values):
            return "SUPPORTED"
        if any(values):
            return "PARTIALLY_SUPPORTED"
        return "NOT_SUPPORTED"

    statuses = {
        "H1_GLOBAL_COMPETITION_CAUSAL_COMPONENT": status(h1_values),
        "H2_HARD_DISTRACTOR_EFFECT": status(hard_random_values),
        "H3_SEARCH_SPACE_DOSE_RESPONSE": status(random_dose_values),
        "H4_LONG_RECORDING_RESCUE": status(rescue_values),
        "H5_SCALE_ERROR_REMAINS_AFTER_COMPETITION_RESCUE": status(h5_values),
    }
    if statuses["H2_HARD_DISTRACTOR_EFFECT"] == "SUPPORTED":
        decision = "HARD_NEGATIVE_COMPETITION_SUPPORTED"
    elif statuses["H1_GLOBAL_COMPETITION_CAUSAL_COMPONENT"] == "SUPPORTED" and statuses["H3_SEARCH_SPACE_DOSE_RESPONSE"] == "SUPPORTED":
        decision = "GLOBAL_SEARCH_COMPETITION_SUPPORTED"
    elif statuses["H1_GLOBAL_COMPETITION_CAUSAL_COMPONENT"] in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
        decision = "GENERAL_SEARCH_SPACE_SUPPORTED"
    elif statuses["H1_GLOBAL_COMPETITION_CAUSAL_COMPONENT"] == "NOT_SUPPORTED" and statuses["H3_SEARCH_SPACE_DOSE_RESPONSE"] == "NOT_SUPPORTED":
        decision = "GLOBAL_COMPETITION_NOT_SUPPORTED"
    else:
        decision = "INCONCLUSIVE"
    return {"statuses": statuses, "decision": decision, "component_values": {"H1": h1_values, "H2": hard_random_values, "H3": random_dose_values, "H4": rescue_values, "H5": h5_values}}


def write_hypothesis_report(
    path: Path,
    assessment: Mapping[str, Any],
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    statuses = assessment["statuses"]
    lines = [
        "# Hypothesis assessment",
        "",
        f"Scientific decision: **{assessment['decision']}**",
        "",
        "The runner used the pre-registered decoder-only memory-key mask. The full encoder, full-coordinate memory, positional embeddings, query embeddings, initial references, model parameters, query count, and post-processing remained fixed.",
        "",
        "| Hypothesis | Status |",
        "|---|---|",
    ]
    for name, status_value in statuses.items():
        lines.append(f"| `{name}` | **{status_value}** |")
    lines.extend([
        "",
        "## Predeclared assessment rules",
        "",
        "- H1 requires GT_ONLY to improve both CenterHit@10≤2s and positive-overlap Top-10 availability in both primary duration bins.",
        "- H2 requires HARD to be lower than matched RANDOM at the informative 25 and 50 levels in both primary bins for CenterHit@10≤2s; level 100 is a structural identity control because both masks expose the same complete non-GT set.",
        "- H3 requires the RANDOM 25→50→100 means to be non-increasing between GT_ONLY and FULL_ACCESS in both primary bins.",
        "- H4 requires the GT_ONLY rescue delta to be positive and larger for the long-recording half in both primary bins.",
        "- H5 records persistent scale error when GT_ONLY has median width/GT > 2 and R1@0.7 < 50% in a primary bin.",
        "",
        "These are diagnostic decision rules, not claims that the intervention is deployable. GT_ONLY uses ground-truth access and is a lower-access endpoint only.",
        "",
        "## Validation",
        "",
        f"- Counterfactual validation: **{'PASS' if validation['pass'] else 'FAIL'}**",
        f"- Full-access reference reproduction: **{'PASS' if validation['reference_comparison']['pass'] else 'FAIL'}**",
        f"- Memory/position/reference invariants: **{'PASS' if validation['invariants_pass'] else 'FAIL'}**",
        "",
        "## Claim boundary",
        "",
        "A positive result supports only a decoder-level full-coordinate temporal competition mechanism. It does not show that global competition is the only cause, that the encoder representation is perfect, that normalized coordinates are irrelevant, or that GT-based masking is a deployable method.",
        "",
        f"Baseline source revision: `{provenance['baseline_commit']}`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(path: Path, validation: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    lines = [
        "# Counterfactual validation report",
        "",
        f"Status: **{'PASS' if validation['pass'] else 'FAIL'}**",
        "",
        "## Fixed path",
        "",
        "The original full-audio QD-DETR encoder and positional path was executed once per batch. The decoder was then called with the original memory and position tensors plus a condition-specific `memory_key_padding_mask`. No crop, re-indexing, synthetic feature, parameter update, or coordinate recomputation was used.",
        "",
        "## Checks",
        "",
        f"- Full-access prediction reproduction: `{'PASS' if validation['reference_comparison']['pass'] else 'FAIL'}`; max absolute serialized difference `{validation['reference_comparison']['max_abs_difference']}`.",
        f"- First-batch direct original-forward comparison: `{'PASS' if validation['direct_forward_pass'] else 'FAIL'}`; max logit difference `{validation['direct_logits_max_abs']}`, max span difference `{validation['direct_spans_max_abs']}`.",
        f"- Encoder memory unchanged after every decoder call: `{'PASS' if validation['memory_pass'] else 'FAIL'}`.",
        f"- Positional embeddings unchanged after every decoder call: `{'PASS' if validation['position_pass'] else 'FAIL'}`.",
        f"- Initial references/query embeddings unchanged: `{'PASS' if validation['reference_pass'] else 'FAIL'}`.",
        f"- Mask contract and original padding preservation: `{'PASS' if validation['mask_pass'] else 'FAIL'}`.",
        f"- Finite tensors and outputs: `{'PASS' if validation['finite_pass'] else 'FAIL'}`.",
        f"- Model query count: `{provenance['num_queries']}`.",
        "",
        "## Token rule",
        "",
        "A one-second token `i` covers `[i, i+1)`. The GT set is the union of valid tokens whose support has positive overlap with any annotated GT interval. All valid non-GT tokens are eligible distractors. No margin is added.",
        "",
        "## Random schedule",
        "",
        f"The fixed project audit seed is `{provenance['base_seed']}`. For query `qid` and replicate `r`, the derived seed is `base_seed + qid * 1,000,003 + r`; the same permutation prefix is used for RANDOM_25 and RANDOM_50. Counts are `ceil(fraction * available_non_gt_tokens)`, capped at the available count; effective counts are recorded in the output table.",
        "",
        "## Provenance",
        "",
        f"- Baseline commit: `{provenance['baseline_commit']}`",
        f"- Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`",
        f"- Config SHA-256: `{provenance['config_sha256']}`",
        f"- Test metadata SHA-256: `{provenance['test_jsonl_sha256']}`",
        f"- Queries processed: `{provenance['query_count']}`",
        f"- Device: `{provenance['device']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    worktree = args.worktree.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(worktree / "src"))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise InvalidCounterfactual("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    option = load_options(args.config.resolve(), worktree, output, args.device)
    from dataset import prepare_batch_inputs, start_end_collate
    from evaluate import setup_model

    checkpoint = args.checkpoint.resolve()
    dataset = build_dataset(option)
    loader = DataLoader(
        dataset,
        collate_fn=start_end_collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    model, criterion, _, _ = setup_model(option)
    del criterion
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    reference_submission = load_jsonl(args.reference_submission.resolve())
    reference_metrics = json.loads(args.reference_metrics.resolve().read_text(encoding="utf-8")) if args.reference_metrics else None
    provenance = {
        "baseline_commit": git_revision(worktree),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "test_jsonl": str(worktree / "data/castella_test_release.jsonl"),
        "test_jsonl_sha256": sha256_file(worktree / "data/castella_test_release.jsonl"),
        "query_count": len(dataset),
        "num_queries": int(option.num_queries),
        "base_seed": args.seed,
        "clip_length": float(option.clip_length),
        "device": args.device,
        "batch_size": args.batch_size,
        "random_replicates": RANDOM_REPLICATES,
        "gt_token_rule": "valid one-second token support with positive overlap with any GT interval",
        "random_count_rule": "ceil(fraction * available non-GT tokens), capped at available count",
        "reference_submission": str(args.reference_submission.resolve()),
        "reference_metrics": str(args.reference_metrics.resolve()) if args.reference_metrics else None,
        "reference_metrics_payload": reference_metrics,
    }
    if provenance["checkpoint_sha256"] != "9cdc18a14e906689484f1dde055b42cdcc4b77f0d850f6a0174ff9ef42063d35":
        raise InvalidCounterfactual("checkpoint SHA-256 does not match the registered checkpoint")

    all_rows: list[dict[str, Any]] = []
    full_submission: list[dict[str, Any]] = []
    validation = {
        "memory_pass": True,
        "position_pass": True,
        "reference_pass": True,
        "mask_pass": True,
        "finite_pass": True,
        "direct_forward_pass": True,
        "direct_logits_max_abs": 0.0,
        "direct_spans_max_abs": 0.0,
        "memory_digest_first_batch": None,
        "position_digest_first_batch": None,
        "initial_reference_digest": tensor_digest(model.query_embed.weight),
        "batches": 0,
    }

    with torch.no_grad():
        for batch_index, (metas, batched) in enumerate(loader):
            model_inputs, _targets = prepare_batch_inputs(batched, option.device)
            encoded = encode_full_audio(model, model_inputs)
            memory_snapshot = encoded["memory_local"].detach().clone()
            position_snapshot = encoded["position_local"].detach().clone()
            reference_snapshot = model.query_embed.weight.detach().clone()
            finite_or_raise("encoder memory", encoded["memory_local"])
            finite_or_raise("position embeddings", encoded["position_local"])
            valid_audio = encoded["valid_audio"]
            memory_batch = encoded["memory_local"].transpose(0, 1)
            saliency = (
                torch.sum(
                    model.saliency_proj1(memory_batch)
                    * model.saliency_proj2(encoded["memory_global"]).unsqueeze(1),
                    dim=-1,
                ) / math.sqrt(model.hidden_dim)
            )
            finite_or_raise("full-access saliency", saliency)
            specs = condition_specs_for_batch(metas, valid_audio, saliency, float(option.clip_length), args.seed)
            gt_by_batch = [
                gt_token_mask(
                    meta,
                    valid_audio[index].detach().cpu().numpy().astype(bool),
                    float(option.clip_length),
                )
                for index, meta in enumerate(metas)
            ]
            assert_mask_contract(specs, encoded, gt_by_batch)
            validation["batches"] += 1
            if batch_index == 0:
                validation["memory_digest_first_batch"] = tensor_digest(encoded["memory_local"])
                validation["position_digest_first_batch"] = tensor_digest(encoded["position_local"])
                original = model(**model_inputs)
                full_spec = next(spec for spec in specs if spec["condition"] == "FULL_ACCESS")
                direct = decode_with_access_mask(model, encoded, full_spec["mask"])
                logits_error = torch.max(torch.abs(original["pred_logits"] - direct["pred_logits"])).item()
                spans_error = torch.max(torch.abs(original["pred_spans"] - direct["pred_spans"])).item()
                saliency_error = torch.max(torch.abs(original["saliency_scores"] - saliency)).item()
                validation["direct_logits_max_abs"] = float(logits_error)
                validation["direct_spans_max_abs"] = float(spans_error)
                validation["direct_saliency_max_abs"] = float(saliency_error)
                validation["direct_forward_pass"] = bool(logits_error <= 1e-5 and spans_error <= 1e-5 and saliency_error <= 1e-5)
                if not validation["direct_forward_pass"]:
                    raise InvalidCounterfactual("custom FULL_ACCESS path differs from original model.forward")

            for spec in specs:
                before_memory = encoded["memory_local"].detach().clone()
                before_position = encoded["position_local"].detach().clone()
                before_reference = model.query_embed.weight.detach().clone()
                decoded = decode_with_access_mask(model, encoded, spec["mask"])
                finite_or_raise(f"{spec['condition']} logits", decoded["pred_logits"])
                finite_or_raise(f"{spec['condition']} spans", decoded["pred_spans"])
                if not torch.equal(encoded["memory_local"], before_memory):
                    validation["memory_pass"] = False
                if not torch.equal(encoded["position_local"], before_position):
                    validation["position_pass"] = False
                if not torch.equal(model.query_embed.weight, before_reference):
                    validation["reference_pass"] = False
                if not torch.equal(encoded["memory_local"], memory_snapshot):
                    validation["memory_pass"] = False
                if not torch.equal(encoded["position_local"], position_snapshot):
                    validation["position_pass"] = False
                if not torch.equal(model.query_embed.weight, reference_snapshot):
                    validation["reference_pass"] = False
                predictions_by_batch = [
                    postprocess_prediction(
                        decoded["pred_spans"][index],
                        decoded["pred_logits"][index],
                        float(meta["duration"]),
                        float(option.clip_length),
                    )
                    for index, meta in enumerate(metas)
                ]
                for index, (meta, predictions, mask_entry) in enumerate(zip(metas, predictions_by_batch, spec["entries"])):
                    metrics = query_metrics(predictions, meta)
                    duration_value = float(meta["duration"])
                    row = {
                        "qid": meta["qid"],
                        "vid": meta["vid"],
                        "duration_sec": duration_value,
                        "audio_duration_sec": duration_value,
                        "gt_duration_sec": longest_gt(meta),
                        "duration_bin": duration_bin(longest_gt(meta)),
                        "condition": spec["condition"],
                        "replicate": int(spec["replicate"]),
                        "random_seed": spec["random_seed"],
                        "gt_token_count": mask_entry["gt_token_count"],
                        "valid_token_count": mask_entry["valid_token_count"],
                        "effective_accessible_tokens": int((~spec["mask"][index]).sum().item()),
                        "effective_distractor_count": int((~spec["mask"][index]).sum().item() - mask_entry["gt_token_count"]),
                    }
                    row.update(metrics)
                    all_rows.append(row)
                    if spec["condition"] == "FULL_ACCESS":
                        full_submission.append({
                            "qid": meta["qid"],
                            "query": meta["query"],
                            "vid": meta["vid"],
                            "pred_relevant_windows": predictions,
                        })

            for spec in specs:
                if not torch.equal(encoded["memory_local"], memory_snapshot):
                    validation["memory_pass"] = False
                if not torch.equal(encoded["position_local"], position_snapshot):
                    validation["position_pass"] = False

    reference_comparison = compare_submissions(reference_submission, full_submission)
    validation["reference_comparison"] = reference_comparison
    validation["finite_pass"] = bool(validation["finite_pass"])
    validation["invariants_pass"] = bool(
        validation["memory_pass"]
        and validation["position_pass"]
        and validation["reference_pass"]
        and validation["mask_pass"]
        and validation["finite_pass"]
    )
    validation["pass"] = bool(validation["invariants_pass"] and validation["direct_forward_pass"] and reference_comparison["pass"])
    if not validation["pass"]:
        raise InvalidCounterfactual(
            "counterfactual validation failed: "
            + json.dumps(
                {
                    "invariants_pass": validation["invariants_pass"],
                    "direct_forward_pass": validation["direct_forward_pass"],
                    "reference_comparison": reference_comparison,
                    "memory_pass": validation["memory_pass"],
                    "position_pass": validation["position_pass"],
                    "reference_pass": validation["reference_pass"],
                    "mask_pass": validation["mask_pass"],
                    "finite_pass": validation["finite_pass"],
                    "direct_logits_max_abs": validation["direct_logits_max_abs"],
                    "direct_spans_max_abs": validation["direct_spans_max_abs"],
                    "direct_saliency_max_abs": validation.get("direct_saliency_max_abs", 0.0),
                },
                sort_keys=True,
            )
        )

    condition_rows = condition_metric_rows(all_rows)
    variability_rows = random_variability_rows(all_rows)
    hard_random_rows = random_vs_hard_rows(all_rows)
    dose_rows = dose_response_rows(all_rows)
    transition_output_rows = transition_rows(all_rows)
    rescue_rows = duration_rescue_rows(all_rows)
    assessment = assess_hypotheses(all_rows, hard_random_rows, dose_rows, rescue_rows)

    query_fields = [
        "qid", "vid", "duration_sec", "audio_duration_sec", "gt_duration_sec", "duration_bin",
        "condition", "replicate", "random_seed", "gt_token_count", "valid_token_count",
        "effective_accessible_tokens", "effective_distractor_count", *METRIC_FIELDS,
    ]
    write_csv(output / "query_level_counterfactual.csv", all_rows, query_fields)
    write_csv(output / "condition_metrics_by_duration.csv", condition_rows, list(condition_rows[0].keys()) if condition_rows else ["condition"])
    write_csv(output / "random_seed_variability.csv", variability_rows, list(variability_rows[0].keys()) if variability_rows else ["condition"])
    write_csv(output / "random_vs_hard.csv", hard_random_rows, list(hard_random_rows[0].keys()) if hard_random_rows else ["duration_bin"])
    write_csv(output / "dose_response.csv", dose_rows, list(dose_rows[0].keys()) if dose_rows else ["duration_bin"])
    write_csv(output / "paired_transitions.csv", transition_output_rows, list(transition_output_rows[0].keys()) if transition_output_rows else ["duration_bin"])
    write_csv(output / "audio_duration_rescue.csv", rescue_rows, list(rescue_rows[0].keys()) if rescue_rows else ["duration_bin"])

    summary = {
        "status": "VALID_COUNTERFACTUAL_COMPLETE",
        "experiment": "COORDINATE_PRESERVING_GLOBAL_SEARCH_SPACE_COUNTERFACTUAL",
        "validation": validation,
        "provenance": provenance,
        "conditions": list(CONDITIONS),
        "primary_bins": list(PRIMARY_BINS),
        "secondary_bins": list(bin_name for bin_name in ALL_BINS if bin_name not in PRIMARY_BINS),
        "hypothesis_assessment": assessment,
        "output_files": [
            "validation_report.md", "condition_metrics_by_duration.csv", "random_seed_variability.csv",
            "random_vs_hard.csv", "dose_response.csv", "paired_transitions.csv", "audio_duration_rescue.csv",
            "hypothesis_assessment.md", "scientific_decision.md", "summary.json", "query_level_counterfactual.csv",
        ],
        "row_count": len(all_rows),
        "full_access_submission_count": len(full_submission),
    }
    write_validation_report(output / "validation_report.md", validation, provenance)
    write_hypothesis_report(output / "hypothesis_assessment.md", assessment, validation, provenance)
    (output / "scientific_decision.md").write_text(
        "# Scientific decision\n\n"
        f"**{assessment['decision']}**\n\n"
        "This decision is restricted to the frozen encoder plus decoder cross-attention key-access intervention. It does not establish a deployable GT mask, eliminate scale error, or establish that the encoder is perfect.\n",
        encoding="utf-8",
    )
    write_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-submission", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    try:
        result = run(arguments)
    except InvalidCounterfactual as error:
        print(f"INVALID_COUNTERFACTUAL: {error}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"status": result["status"], "decision": result["hypothesis_assessment"]["decision"]}, indent=2))
