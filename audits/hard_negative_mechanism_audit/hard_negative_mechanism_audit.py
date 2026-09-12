#!/usr/bin/env python3
"""Run the bounded GT-vs-hard-negative mechanism audit.

This script is inference-only.  It reuses the verified coordinate-preserving
counterfactual helpers, keeps the frozen QD-DETR encoder and temporal
coordinates unchanged, and records quantities that are either selection
conditioned or independently measured.  It intentionally does not train,
modify checkpoint weights, or compute unsupported audio/text similarities.
"""

from __future__ import annotations

import argparse
import csv
import html
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
HARMFUL_RULE = "HARD_25 center_hit10_2s < mean(RANDOM_25 replicate 0..9 center_hit10_2s)"
ADJACENT_SEC = 1.0
MODERATELY_NEAR_SEC = 5.0
EXPANDED_NEIGHBORHOOD_SEC = 2.0
FINAL_CENTER_TOLERANCE_SEC = 2.0


def load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    return float(value)


def cohort_from_counterfactual(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    by_qid: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["duration_bin"] in PRIMARY_BINS:
            by_qid[row["qid"]][row["condition"]].append(row)
    cohort: dict[str, dict[str, Any]] = {}
    for qid, conditions in by_qid.items():
        hard = conditions.get("HARD_25", [])
        random_rows = conditions.get("RANDOM_25", [])
        if len(hard) != 1 or len(random_rows) != 10:
            raise RuntimeError(f"counterfactual replicate contract failed for qid={qid}")
        random_mean = float(np.mean([safe_float(row["center_hit10_2s"]) for row in random_rows]))
        hard_value = safe_float(hard[0]["center_hit10_2s"])
        delta = hard_value - random_mean
        cohort[qid] = {
            "qid": qid,
            "vid": hard[0]["vid"],
            "duration_bin": hard[0]["duration_bin"],
            "gt_duration_sec": safe_float(hard[0]["gt_duration_sec"]),
            "audio_duration_sec": safe_float(hard[0]["audio_duration_sec"]),
            "hard_center_hit10_2s": hard_value,
            "random25_center_hit10_2s_mean": random_mean,
            "hard_minus_random_center": delta,
            "cohort": "harmful" if delta < 0 else "control",
        }
    return cohort


def nearest_gt_gap(start: float, end: float, windows: Sequence[Sequence[float]]) -> float:
    gaps = []
    for gt_start, gt_end in windows:
        if end < gt_start:
            gaps.append(gt_start - end)
        elif start > gt_end:
            gaps.append(start - gt_end)
        else:
            gaps.append(0.0)
    return min(gaps)


def nearest_gt_center_distance(center: float, windows: Sequence[Sequence[float]]) -> float:
    return min(abs(center - (float(start) + float(end)) / 2.0) for start, end in windows)


def interval_overlaps(start: float, end: float, windows: Sequence[Sequence[float]], margin: float = 0.0) -> bool:
    return any(start < float(gt_end) + margin and end > float(gt_start) - margin for gt_start, gt_end in windows)


def components(indices: Sequence[int]) -> list[list[int]]:
    if not indices:
        return []
    values = sorted(int(index) for index in indices)
    output: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value == output[-1][-1] + 1:
            output[-1].append(value)
        else:
            output.append([value])
    return output


def strongest_peak(
    selected: Sequence[int],
    saliency: np.ndarray,
    clip_length: float,
    duration: float,
) -> tuple[list[int], int, float, float, float]:
    if not selected:
        raise RuntimeError("empty hard-negative selection")
    top_token = max(selected, key=lambda index: (float(saliency[index]), -int(index)))
    selected_components = components(selected)
    component = next(item for item in selected_components if top_token in item)
    start = float(component[0] * clip_length)
    end = min(float(duration), float((component[-1] + 1) * clip_length))
    return component, top_token, start, end, (start + end) / 2.0


def bool_center_rank(predictions: Sequence[Sequence[float]], center: float, tolerance: float) -> int:
    for index, prediction in enumerate(predictions, start=1):
        candidate_center = (float(prediction[0]) + float(prediction[1])) / 2.0
        if abs(candidate_center - center) <= tolerance:
            return index
    return len(predictions) + 1


def gt_center_rank(predictions: Sequence[Sequence[float]], windows: Sequence[Sequence[float]], tolerance: float) -> int:
    centers = [(float(start) + float(end)) / 2.0 for start, end in windows]
    for index, prediction in enumerate(predictions, start=1):
        candidate_center = (float(prediction[0]) + float(prediction[1])) / 2.0
        if min(abs(candidate_center - center) for center in centers) <= tolerance:
            return index
    return len(predictions) + 1


def interval_width_gt_ratio(predictions: Sequence[Sequence[float]], windows: Sequence[Sequence[float]]) -> float:
    predicted_width = max(0.0, float(predictions[0][1]) - float(predictions[0][0]))
    gt_width = max(float(end) - float(start) for start, end in windows)
    return predicted_width / gt_width


def load_options(cf: types.ModuleType, config: Path, worktree: Path, output: Path, device: str):
    return cf.load_options(config, worktree, output, device)


def encode_full_audio(cf: types.ModuleType, model: torch.nn.Module, model_inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return cf.encode_full_audio(model, model_inputs)


def decode_trace(
    model: torch.nn.Module,
    encoded: Mapping[str, torch.Tensor],
    decoder_padding: torch.Tensor,
    capture_attention: bool,
) -> dict[str, Any]:
    """Call the unchanged decoder and optionally retain its returned weights."""
    memory_local = encoded["memory_local"]
    batch_size = memory_local.shape[1]
    query_embed = model.query_embed.weight
    refpoint_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
    target = torch.zeros(
        refpoint_embed.shape[0], batch_size, memory_local.shape[2], device=memory_local.device
    )
    captured: dict[int, torch.Tensor] = {}
    originals: list[Any] = []
    if capture_attention:
        for layer_index, layer in enumerate(model.transformer.decoder.layers):
            original = layer.cross_attn.forward
            originals.append(original)

            def wrapped(*args: Any, _original=original, _layer_index=layer_index, **kwargs: Any):
                result = _original(*args, **kwargs)
                if result[1] is not None:
                    captured[_layer_index] = result[1].detach()
                return result

            layer.cross_attn.forward = wrapped
    try:
        hidden_states, references = model.transformer.decoder(
            target,
            memory_local,
            memory_key_padding_mask=decoder_padding,
            pos=encoded["position_local"],
            refpoints_unsigmoid=refpoint_embed,
        )
    finally:
        if capture_attention:
            for layer, original in zip(model.transformer.decoder.layers, originals):
                layer.cross_attn.forward = original
    outputs_class = model.class_embed(hidden_states)
    coordinates = model.span_embed(hidden_states) + _CF.inverse_sigmoid(references)
    if model.span_loss_type == "l1":
        coordinates = coordinates.sigmoid()
    return {
        "pred_logits": outputs_class[-1],
        "pred_spans": coordinates[-1],
        "initial_references": refpoint_embed.sigmoid(),
        "hidden_states": hidden_states,
        "references": references,
        "attention": captured,
    }


def make_mask(valid: np.ndarray, gt: np.ndarray, selected: Sequence[int]) -> torch.Tensor:
    keep = np.zeros(len(valid), dtype=bool)
    keep[gt] = True
    keep[list(selected)] = True
    return torch.from_numpy(~keep)


def p3_failure(qid: str, submission: Mapping[str, Mapping[str, Any]], meta: Mapping[str, Any]) -> bool:
    prediction = submission.get(qid)
    if prediction is None:
        return True
    gt_centers = [(float(start) + float(end)) / 2.0 for start, end in meta["relevant_windows"]]
    for candidate in prediction["pred_relevant_windows"][:10]:
        center = (float(candidate[0]) + float(candidate[1])) / 2.0
        if min(abs(center - gt_center) for gt_center in gt_centers) <= 2.0:
            return False
    return True


def condition_summary(
    cf: types.ModuleType,
    decoded: Mapping[str, Any],
    batch_index: int,
    meta: Mapping[str, Any],
    hard_center: float,
    hard_component: Sequence[int],
    gt_mask: np.ndarray,
    clip_length: float,
    duration: float,
) -> dict[str, Any]:
    predictions = cf.postprocess_prediction(
        decoded["pred_spans"][batch_index],
        decoded["pred_logits"][batch_index],
        duration,
        clip_length,
    )
    centers = [(float(item[0]) + float(item[1])) / 2.0 for item in predictions]
    gt_windows = meta["relevant_windows"]
    rank_gt = gt_center_rank(predictions, gt_windows, FINAL_CENTER_TOLERANCE_SEC)
    rank_hard = bool_center_rank(predictions, hard_center, FINAL_CENTER_TOLERANCE_SEC)
    density_gt = sum(
        min(abs(center - (float(start) + float(end)) / 2.0) for start, end in gt_windows) <= FINAL_CENTER_TOLERANCE_SEC
        for center in centers[:10]
    )
    density_hard = sum(abs(center - hard_center) <= FINAL_CENTER_TOLERANCE_SEC for center in centers[:10])
    return {
        "predictions": predictions,
        "top1_center_sec": centers[0],
        "top1_width_gt_ratio": interval_width_gt_ratio(predictions, gt_windows),
        "best_gt_center_rank": rank_gt,
        "best_hard_center_rank": rank_hard,
        "gt_center_density_top10": int(density_gt),
        "hard_center_density_top10": int(density_hard),
        "attention": decoded["attention"],
        "gt_mask": gt_mask,
        "hard_component": hard_component,
    }


def attention_masses(
    condition: Mapping[str, Any],
    batch_index: int,
    top1_query: int,
    valid: np.ndarray,
) -> dict[str, float]:
    output: dict[str, float] = {}
    gt_mask = condition["gt_mask"]
    hard_mask = np.zeros(len(valid), dtype=bool)
    hard_mask[list(condition["hard_component"])] = True
    for layer_index, weights in sorted(condition["attention"].items()):
        array = weights[batch_index, top1_query].detach().float().cpu().numpy()
        finite = np.isfinite(array)
        array = np.where(finite, array, 0.0)
        total = float(array.sum())
        output[f"attn_l{layer_index}_total"] = total
        output[f"attn_l{layer_index}_gt_mass"] = float(array[gt_mask].sum())
        output[f"attn_l{layer_index}_hard_peak_mass"] = float(array[hard_mask].sum())
        output[f"attn_l{layer_index}_other_mass"] = float(array[~gt_mask & ~hard_mask].sum())
    return output


def decoder_row(
    decoded: Mapping[str, Any],
    batch_index: int,
    meta: Mapping[str, Any],
    hard_center: float,
    duration: float,
) -> dict[str, Any]:
    predictions = cf_postprocess_prediction(decoded, batch_index, meta, duration)
    scores = torch.softmax(decoded["pred_logits"][batch_index].detach().float(), dim=-1)[:, 0]
    top1_query = int(torch.argmax(scores).item())
    refs = decoded["references"][:, batch_index, :, :].detach().float().cpu().numpy()
    initial = decoded["initial_references"][..., :].detach().float().cpu().numpy()[:, batch_index, :]
    # references[0] is the same initial reference returned by the decoder.
    centers = refs[:, top1_query, 0] * duration
    initial_center = float(initial[top1_query, 0] * duration)
    gt_centers = [(float(start) + float(end)) / 2.0 for start, end in meta["relevant_windows"]]
    init_gt = min(abs(initial_center - value) for value in gt_centers)
    init_hard = abs(initial_center - hard_center)
    final_center = float((predictions[0][0] + predictions[0][1]) / 2.0)
    final_gt = min(abs(final_center - value) for value in gt_centers)
    final_hard = abs(final_center - hard_center)
    if init_gt < init_hard and final_hard < final_gt:
        classification = "begin_gt_move_hard"
    elif init_hard <= init_gt and final_hard <= final_gt:
        classification = "begin_hard_and_remain"
    elif final_gt < init_gt:
        classification = "approach_gt"
    else:
        classification = "approach_neither"
    return {
        "top1_query_index": top1_query,
        "initial_gt_distance_sec": init_gt,
        "initial_hard_distance_sec": init_hard,
        "final_gt_distance_sec": final_gt,
        "final_hard_distance_sec": final_hard,
        "hard_distance_change_sec": final_hard - init_hard,
        "gt_distance_change_sec": final_gt - init_gt,
        "reference_attraction_flag": bool(final_hard < init_hard and final_hard < final_gt),
        "classification": classification,
        "reference_centers_sec": json.dumps([float(value) for value in centers.tolist()]),
        "final_center_sec": final_center,
        "final_width_gt_ratio": interval_width_gt_ratio(predictions, meta["relevant_windows"]),
    }


def cf_postprocess_prediction(decoded: Mapping[str, Any], batch_index: int, meta: Mapping[str, Any], duration: float) -> list[list[float]]:
    # The module is injected in run() to keep this helper readable.
    return _CF.postprocess_prediction(decoded["pred_spans"][batch_index], decoded["pred_logits"][batch_index], duration, _CF_CLIP_LENGTH)


_CF: types.ModuleType
_CF_CLIP_LENGTH: float


def write_cohort_definition(path: Path, cohort: Mapping[str, Mapping[str, Any]], p3: Mapping[str, bool], provenance: Mapping[str, Any]) -> None:
    counts = {bin_name: {label: 0 for label in ("harmful", "control")} for bin_name in PRIMARY_BINS}
    p3_counts = {bin_name: 0 for bin_name in PRIMARY_BINS}
    for qid, row in cohort.items():
        counts[row["duration_bin"]][row["cohort"]] += 1
        p3_counts[row["duration_bin"]] += int(p3.get(qid, False))
    lines = [
        "# GT vs hard-negative mechanism audit: cohort definition",
        "",
        "The cohort was frozen from the saved coordinate-preserving counterfactual output before mechanism measurements.",
        "",
        f"- Harmful-hard rule: `{HARMFUL_RULE}`.",
        "- Control rule: the paired HARD_25 value is greater than or equal to the query's 10-replicate RANDOM_25 mean.",
        "- Primary duration bins: `0-2s` and `2-5s`; no manual query selection was used.",
        "- P3 location failure: no baseline Top-10 candidate center within ±2 s of any GT center.",
        f"- Geometry bins: adjacent = interval gap ≤ {ADJACENT_SEC:g} s; moderately near = > {ADJACENT_SEC:g} and ≤ {MODERATELY_NEAR_SEC:g} s; remote = > {MODERATELY_NEAR_SEC:g} s.",
        f"- Expanded GT neighborhood: any strongest HARD_25 peak overlap with a GT interval expanded by ±{EXPANDED_NEIGHBORHOOD_SEC:g} s.",
        "- The strongest hard peak is the contiguous temporal-token component containing the highest-saliency selected HARD_25 token. Saliency is recorded only as the selection variable; it is not treated as independent explanatory evidence.",
        "- Native audio-text comparison, when available, uses equal absolute window duration equal to max(1 s, the longest annotated GT interval), with the GT window centered on the longest GT interval and the hard window centered on the strongest hard peak.",
        "- Matched replacement is one pre-specified token-level replacement in RANDOM_25 replicate 0: remove the first deterministic random token and add the strongest selected hard token not already present. A peak-level replacement is not claimed.",
        "",
        "## Cohort counts",
        "",
        "| Duration bin | Harmful | Control | P3 failure total |",
        "|---|---:|---:|---:|",
    ]
    for bin_name in PRIMARY_BINS:
        lines.append(f"| {bin_name} | {counts[bin_name]['harmful']} | {counts[bin_name]['control']} | {p3_counts[bin_name]} |")
    lines += [
        "",
        "## Measurement boundaries",
        "",
        "QD-DETR quantities are measured in the official model's internal temporal path. Decoder cross-attention weights are returned by the existing attention implementation and captured without changing its inputs or parameters. Query-conditioned memory response is BLOCKED because the audit has no independently justified common query-memory scoring rule.",
        "",
        "Native MS-CLAP is scored only for qids whose original WAV is present. Missing WAVs are retained as BLOCKED rows; stored QD temporal features are never substituted.",
        "",
        f"Counterfactual source: `{provenance['counterfactual_csv']}`; baseline checkpoint SHA-256 `{provenance['checkpoint_sha256']}`; QD baseline commit `{provenance['baseline_commit']}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_human_review(
    directory: Path,
    cohort: Mapping[str, Mapping[str, Any]],
    meta_by_qid: Mapping[str, Mapping[str, Any]],
    geometry_by_qid: Mapping[str, Mapping[str, Any]],
    raw_audio_dir: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for bin_name in PRIMARY_BINS:
        selected = [
            row for row in cohort.values()
            if row["duration_bin"] == bin_name and row["cohort"] == "harmful"
            and (raw_audio_dir / f"{row['vid']}.wav").exists()
        ]
        selected.sort(key=lambda row: (float(row["hard_minus_random_center"]), str(row["qid"])))
        rows.extend(selected[:15])
    manifest: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row["qid"])
        meta = meta_by_qid[qid]
        geometry = geometry_by_qid.get(qid, {})
        manifest.append({
            "qid": qid,
            "vid": row["vid"],
            "duration_bin": row["duration_bin"],
            "cohort": row["cohort"],
            "query": meta.get("query", ""),
            "gt_windows": json.dumps(meta["relevant_windows"], ensure_ascii=False),
            "hard_peak_start_sec": geometry.get("hard_peak_start_sec", ""),
            "hard_peak_end_sec": geometry.get("hard_peak_end_sec", ""),
            "hard_peak_center_sec": geometry.get("hard_peak_center_sec", ""),
            "audio_relpath": f"../../../review_ui/audio/{row['vid']}.wav",
            "human_label": "",
            "review_notes": "",
        })
    fields = list(manifest[0].keys()) if manifest else ["qid", "vid", "duration_bin", "human_label", "review_notes"]
    write_csv(directory / "review_manifest.csv", manifest, fields)
    lines = [
        "<!doctype html><meta charset='utf-8'><title>Hard-negative mechanism review</title>",
        "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto}article{border:1px solid #ccc;padding:1rem;margin:1rem 0}audio{width:100%}code{white-space:pre-wrap}</style>",
        "<h1>GT vs hard-negative mechanism review</h1>",
        "<p>Labels are intentionally blank. Review each case using the query, the GT interval, and the strongest HARD_25 peak. Suggested labels: A target event, B semantically similar, C unrelated, D ambiguous, E possible annotation miss.</p>",
    ]
    for row in manifest:
        lines.extend([
            "<article>",
            f"<h2>{html.escape(row['qid'])} · {html.escape(row['duration_bin'])}</h2>",
            f"<p><b>Query:</b> {html.escape(row['query'])}</p>",
            f"<p><b>GT:</b> {html.escape(row['gt_windows'])}<br><b>Hard peak:</b> {row['hard_peak_start_sec']}–{row['hard_peak_end_sec']} s (center {row['hard_peak_center_sec']} s)</p>",
            f"<audio controls preload='none' src='{html.escape(row['audio_relpath'])}'></audio>",
            "<p>Human label: ____ &nbsp; Notes: __________________________________________</p>",
            "</article>",
        ])
    (directory / "review.html").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "README.md").write_text(
        "# Human review\n\n"
        "This is a preselected, listening-only review set. It contains up to 15 harmful cases per primary duration bin with available original WAVs, ranked before listening by the paired HARD_25 minus RANDOM_25 center-hit difference. No labels were auto-filled.\n\n"
        "Use `review.html` or fill `review_manifest.csv`. The hard-negative classification remains unresolved until a human supplies labels.\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    global _CF, _CF_CLIP_LENGTH
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cf = load_module(args.counterfactual_runner.resolve(), "counterfactual_helpers")
    _CF = cf
    cohort = cohort_from_counterfactual(args.counterfactual_csv.resolve())
    all_meta = read_jsonl(args.test_jsonl.resolve())
    meta_by_qid = {str(item["qid"]): item for item in all_meta}
    submission = {str(item["qid"]): item for item in read_jsonl(args.reference_submission.resolve())}
    p3_by_qid = {qid: p3_failure(qid, submission, meta_by_qid[qid]) for qid in cohort}
    provenance = {
        "counterfactual_csv": "saved search-space counterfactual output/query_level_counterfactual.csv",
        "baseline_commit": cf.git_revision(args.worktree.resolve()),
        "checkpoint_sha256": cf.sha256_file(args.checkpoint.resolve()),
        "test_jsonl_sha256": cf.sha256_file(args.test_jsonl.resolve()),
        "qd_device": args.device,
        "harmful_rule": HARMFUL_RULE,
        "selected_primary_qids": len(cohort),
    }
    write_cohort_definition(output / "cohort_definition.md", cohort, p3_by_qid, provenance)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sys.path.insert(0, str(args.worktree.resolve() / "src"))
    from dataset import prepare_batch_inputs, start_end_collate
    from evaluate import setup_model

    option = load_options(cf, args.config.resolve(), args.worktree.resolve(), output, args.device)
    _CF_CLIP_LENGTH = float(option.clip_length)
    dataset = cf.build_dataset(option)
    selected_qids = set(cohort)
    selected_indices = [index for index, item in enumerate(dataset.data) if str(item["qid"]) in selected_qids]
    if len(selected_indices) != len(selected_qids):
        raise RuntimeError(f"dataset cohort mismatch {len(selected_indices)} != {len(selected_qids)}")
    loader = DataLoader(
        Subset(dataset, selected_indices),
        collate_fn=start_end_collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    model, criterion, _, _ = setup_model(option)
    del criterion
    state = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    geometry_rows: list[dict[str, Any]] = []
    qd_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    geometry_by_qid: dict[str, dict[str, Any]] = {}
    batches = 0
    with torch.no_grad():
        for metas, batched in loader:
            model_inputs, _targets = prepare_batch_inputs(batched, option.device)
            encoded = encode_full_audio(cf, model, model_inputs)
            valid_audio = encoded["valid_audio"]
            memory_batch = encoded["memory_local"].transpose(0, 1)
            saliency = (
                torch.sum(
                    model.saliency_proj1(memory_batch)
                    * model.saliency_proj2(encoded["memory_global"]).unsqueeze(1),
                    dim=-1,
                ) / math.sqrt(model.hidden_dim)
            )
            valid_np = valid_audio.detach().cpu().numpy().astype(bool)
            saliency_np = saliency.detach().cpu().numpy()
            hard_masks: list[torch.Tensor] = []
            full_masks: list[torch.Tensor] = []
            random_masks: list[torch.Tensor] = []
            replacement_masks: list[torch.Tensor] = []
            batch_info: list[dict[str, Any]] = []
            for index, meta in enumerate(metas):
                qid = str(meta["qid"])
                valid = valid_np[index]
                gt = cf.gt_token_mask(meta, valid, float(option.clip_length))
                non_gt = np.flatnonzero(valid & ~gt)
                hard_count = min(len(non_gt), int(math.ceil(0.25 * len(non_gt))))
                hard_order = sorted((int(item) for item in non_gt), key=lambda item: (-float(saliency_np[index, item]), item))
                hard_selected = hard_order[:hard_count]
                component, hard_token, peak_start, peak_end, peak_center = strongest_peak(
                    hard_selected, saliency_np[index], float(option.clip_length), float(meta["duration"])
                )
                windows = [[float(start), float(end)] for start, end in meta["relevant_windows"]]
                gt_gap = nearest_gt_gap(peak_start, peak_end, windows)
                gt_center_distance = nearest_gt_center_distance(peak_center, windows)
                if gt_gap <= ADJACENT_SEC:
                    geometry_class = "adjacent"
                elif gt_gap <= MODERATELY_NEAR_SEC:
                    geometry_class = "moderately_near"
                else:
                    geometry_class = "remote"
                expanded = interval_overlaps(peak_start, peak_end, windows, EXPANDED_NEIGHBORHOOD_SEC)
                gt_centers = [(start + end) / 2.0 for start, end in windows]
                gt_center = gt_centers[int(np.argmax([end - start for start, end in windows]))]
                duration = float(meta["duration"])
                gt_mask = gt
                hard_mask = make_mask(valid, gt, hard_selected)
                full_mask = torch.from_numpy(~valid)
                seed = cf.derived_seed(args.seed, qid, 0)
                random_order = np.random.default_rng(seed).permutation(non_gt).tolist()
                random_count = min(len(non_gt), int(math.ceil(0.25 * len(non_gt))))
                random_selected = [int(item) for item in random_order[:random_count]]
                random_mask = make_mask(valid, gt, random_selected)
                if qid in cohort and cohort[qid]["cohort"] == "harmful" and random_selected:
                    replacement_token = next((item for item in hard_order if item not in random_selected), None)
                    removed_token = int(random_selected[0])
                    if replacement_token is None:
                        replacement_mask = full_mask
                        replacement_status = "BLOCKED_no_clean_hard_token"
                    else:
                        replacement_selected = list(random_selected)
                        replacement_selected.remove(removed_token)
                        replacement_selected.append(int(replacement_token))
                        replacement_mask = make_mask(valid, gt, replacement_selected)
                        replacement_status = "VALID_TOKEN_LEVEL"
                else:
                    replacement_token = None
                    removed_token = None
                    replacement_mask = full_mask
                    replacement_status = "NOT_IN_HARMFUL_COHORT"
                hard_masks.append(hard_mask)
                full_masks.append(full_mask)
                random_masks.append(random_mask)
                replacement_masks.append(replacement_mask)
                batch_info.append({
                    "qid": qid,
                    "meta": meta,
                    "valid": valid,
                    "gt": gt,
                    "non_gt": non_gt,
                    "hard_selected": hard_selected,
                    "hard_order": hard_order,
                    "hard_token": hard_token,
                    "component": component,
                    "peak_start": peak_start,
                    "peak_end": peak_end,
                    "peak_center": peak_center,
                    "gt_gap": gt_gap,
                    "gt_center_distance": gt_center_distance,
                    "geometry_class": geometry_class,
                    "expanded": expanded,
                    "gt_center": gt_center,
                    "gt_mask": gt_mask,
                    "hard_mask": hard_mask,
                    "full_mask": full_mask,
                    "random_mask": random_mask,
                    "replacement_mask": replacement_mask,
                    "replacement_token": replacement_token,
                    "removed_token": removed_token,
                    "replacement_status": replacement_status,
                    "random_count": random_count,
                })

            hard_decoded = decode_trace(model, encoded, torch.stack(hard_masks).to(encoded["memory_local"].device), True)
            full_decoded = decode_trace(model, encoded, torch.stack(full_masks).to(encoded["memory_local"].device), True)
            random_decoded = decode_trace(model, encoded, torch.stack(random_masks).to(encoded["memory_local"].device), False)
            replacement_decoded = decode_trace(model, encoded, torch.stack(replacement_masks).to(encoded["memory_local"].device), False)

            # The same full-access top-1 decoder query is the fixed reference for
            # the layerwise attraction trace.
            for index, info in enumerate(batch_info):
                qid = info["qid"]
                row_cohort = cohort[qid]
                meta = info["meta"]
                duration = float(meta["duration"])
                windows = [[float(start), float(end)] for start, end in meta["relevant_windows"]]
                geometry_row = {
                    **row_cohort,
                    "query": meta.get("query", ""),
                    "p3_location_failure": int(p3_by_qid[qid]),
                    "gt_windows": json.dumps(windows),
                    "hard_selected_token_count": len(info["hard_selected"]),
                    "hard_peak_token": info["hard_token"],
                    "hard_peak_component_size": len(info["component"]),
                    "hard_peak_token_indices": json.dumps(info["component"]),
                    "hard_peak_start_sec": info["peak_start"],
                    "hard_peak_end_sec": info["peak_end"],
                    "hard_peak_center_sec": info["peak_center"],
                    "hard_peak_to_gt_interval_gap_sec": info["gt_gap"],
                    "hard_peak_to_gt_center_sec": info["gt_center_distance"],
                    "hard_peak_geometry_class": info["geometry_class"],
                    "hard_peak_overlaps_gt_plus_minus_2s": int(info["expanded"]),
                    "hard_peak_saliency": float(saliency_np[index, info["hard_token"]]),
                    "gt_max_saliency": float(np.max(saliency_np[index][info["gt"]])),
                    "selection_conditioned_note": "HARD_25 selected by full-access saliency; saliency is not independent evidence",
                }
                geometry_rows.append(geometry_row)
                geometry_by_qid[qid] = geometry_row

                full_summary = condition_summary(cf, full_decoded, index, meta, info["peak_center"], info["component"], info["gt"], float(option.clip_length), duration)
                hard_summary = condition_summary(cf, hard_decoded, index, meta, info["peak_center"], info["component"], info["gt"], float(option.clip_length), duration)
                top1_query = int(torch.argmax(torch.softmax(full_decoded["pred_logits"][index].detach().float(), dim=-1)[:, 0]).item())
                full_attn = attention_masses(full_summary, index, top1_query, info["valid"])
                hard_attn = attention_masses(hard_summary, index, top1_query, info["valid"])
                qd_row: dict[str, Any] = {
                    **row_cohort,
                    "p3_location_failure": int(p3_by_qid[qid]),
                    "query_conditioned_memory_score_status": "BLOCKED",
                    "query_conditioned_memory_score_reason": "No official common QD query-memory score; stored 768-D audio/text features are not compared.",
                    "full_best_gt_center_rank": full_summary["best_gt_center_rank"],
                    "full_best_hard_center_rank": full_summary["best_hard_center_rank"],
                    "full_gt_center_density_top10": full_summary["gt_center_density_top10"],
                    "full_hard_center_density_top10": full_summary["hard_center_density_top10"],
                    "full_top1_center_sec": full_summary["top1_center_sec"],
                    "full_top1_width_gt_ratio": full_summary["top1_width_gt_ratio"],
                    "hard25_best_gt_center_rank": hard_summary["best_gt_center_rank"],
                    "hard25_best_hard_center_rank": hard_summary["best_hard_center_rank"],
                    "hard25_gt_center_density_top10": hard_summary["gt_center_density_top10"],
                    "hard25_hard_center_density_top10": hard_summary["hard_center_density_top10"],
                    "hard25_top1_center_sec": hard_summary["top1_center_sec"],
                    "hard25_top1_width_gt_ratio": hard_summary["top1_width_gt_ratio"],
                }
                qd_row.update({f"full_{key}": value for key, value in full_attn.items()})
                qd_row.update({f"hard25_{key}": value for key, value in hard_attn.items()})
                qd_rows.append(qd_row)

                attraction = decoder_row(full_decoded, index, meta, info["peak_center"], duration)
                attraction.update({
                    "qid": qid,
                    "vid": meta["vid"],
                    "duration_bin": meta.get("duration_bin", row_cohort["duration_bin"]),
                    "cohort": row_cohort["cohort"],
                    "p3_location_failure": int(p3_by_qid[qid]),
                    "hard_peak_center_sec": info["peak_center"],
                })
                decoder_rows.append(attraction)

                if row_cohort["cohort"] == "harmful":
                    base_summary = condition_summary(cf, random_decoded, index, meta, info["peak_center"], info["component"], info["gt"], float(option.clip_length), duration)
                    repl_summary = condition_summary(cf, replacement_decoded, index, meta, info["peak_center"], info["component"], info["gt"], float(option.clip_length), duration)
                    replacement_rows.append({
                        **row_cohort,
                        "p3_location_failure": int(p3_by_qid[qid]),
                        "replacement_status": info["replacement_status"],
                        "replacement_unit": "single_token; peak-level replacement not claimed",
                        "removed_random_token": info["removed_token"] if info["removed_token"] is not None else "",
                        "added_hard_token": info["replacement_token"] if info["replacement_token"] is not None else "",
                        "hard_peak_component_size": len(info["component"]),
                        "random25_top1_center_sec": base_summary["top1_center_sec"],
                        "replacement_top1_center_sec": repl_summary["top1_center_sec"],
                        "random25_top1_width_gt_ratio": base_summary["top1_width_gt_ratio"],
                        "replacement_top1_width_gt_ratio": repl_summary["top1_width_gt_ratio"],
                        "random25_best_gt_center_rank": base_summary["best_gt_center_rank"],
                        "replacement_best_gt_center_rank": repl_summary["best_gt_center_rank"],
                        "random25_gt_center_density_top10": base_summary["gt_center_density_top10"],
                        "replacement_gt_center_density_top10": repl_summary["gt_center_density_top10"],
                    })
            batches += 1

    geometry_fields = list(geometry_rows[0].keys()) if geometry_rows else ["qid"]
    qd_fields = list(qd_rows[0].keys()) if qd_rows else ["qid"]
    decoder_fields = list(decoder_rows[0].keys()) if decoder_rows else ["qid"]
    replacement_fields = list(replacement_rows[0].keys()) if replacement_rows else ["qid"]
    write_csv(output / "hard_negative_geometry.csv", geometry_rows, geometry_fields)
    write_csv(output / "qd_internal_gt_vs_hard.csv", qd_rows, qd_fields)
    write_csv(output / "decoder_attraction.csv", decoder_rows, decoder_fields)
    write_csv(output / "matched_replacement_probe.csv", replacement_rows, replacement_fields)
    write_human_review(output / "human_review", cohort, meta_by_qid, geometry_by_qid, args.raw_audio_dir.resolve())
    write_json(output / "qd_run_provenance.json", {**provenance, "batches": batches, "geometry_rows": len(geometry_rows), "replacement_rows": len(replacement_rows)})
    print(json.dumps({"batches": batches, "geometry_rows": len(geometry_rows), "replacement_rows": len(replacement_rows), "output": str(output)}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--reference-submission", type=Path, required=True)
    parser.add_argument("--counterfactual-csv", type=Path, required=True)
    parser.add_argument("--counterfactual-runner", type=Path, required=True)
    parser.add_argument("--raw-audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
