"""Execute the matched R1 refinement-geometry causal pilot."""

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
from torch.utils.data import DataLoader

from basic_utils import mkdirp
from config import BaseOptions
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from evaluate import eval_epoch, setup_model
from postprocessing import PostProcessorDETR


SEED = 2023
R1_EPS = 1e-3
VARIANTS = ("baseline", "r1")
GT_BINS = ("0-2s", "2-5s", "5-10s", "10-20s", "20s+", "overall")
STAGES = ("initial", "decoder_layer1", "decoder_layer2", "final_span")


def set_seed(seed: int) -> None:
    """Set the random sources used by model construction and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_options(base_config: Path, variant: str, out_dir: Path) -> EasyDict:
    """Load the official config and bind verified shared assets."""
    manager = BaseOptions(str(base_config))
    manager.parse()
    opt = EasyDict(dict(manager.option))
    opt.seed = SEED
    opt.r1_enabled = variant == "r1"
    opt.r1_eps = R1_EPS
    opt.results_dir = str(out_dir / variant)
    opt.ckpt_filepath = str(out_dir / variant / "best_checkpoint.pth")
    opt.train_log_filepath = str(out_dir / variant / "train.log")
    opt.eval_log_filepath = str(out_dir / variant / "val.log")
    opt.eval_split_name = "val"
    opt.train_path = "/private/research-artifact"
    opt.eval_path = "/private/research-artifact"
    opt.val_path = opt.eval_path
    opt.test_path = "/private/research-artifact"
    opt.a_feat_dir = "/private/research-artifact"
    opt.t_feat_dir = "/private/research-artifact"
    return opt


def build_dataset(opt: EasyDict, split: str) -> StartEndDataset:
    """Build one official CASTELLA split using the unchanged loader."""
    data_path = getattr(opt, f"{split}_path")
    return StartEndDataset(
        data_path=data_path,
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


def iou(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Return temporal intersection over union for two intervals."""
    left = max(float(pred[0]), float(gt[0]))
    right = min(float(pred[1]), float(gt[1]))
    intersection = max(0.0, right - left)
    union = max(float(pred[1]), float(gt[1])) - min(float(pred[0]), float(gt[0]))
    return intersection / union if union > 0 else 0.0


def gt_len(gt: Sequence[Sequence[float]]) -> float:
    """Return the longest annotated GT duration used for binning."""
    return max(float(span[1]) - float(span[0]) for span in gt)


def gt_centers(gt: Sequence[Sequence[float]]) -> list[float]:
    """Return all annotated GT centers in seconds."""
    return [(float(span[0]) + float(span[1])) / 2.0 for span in gt]


def gt_bin(gt: Sequence[Sequence[float]]) -> str:
    """Assign the predeclared GT-duration bin."""
    length = gt_len(gt)
    if length < 2:
        return "0-2s"
    if length < 5:
        return "2-5s"
    if length < 10:
        return "5-10s"
    if length < 20:
        return "10-20s"
    return "20s+"


def finite_median(values: Iterable[float]) -> float | None:
    """Return a finite median, or None for an empty set."""
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


def mean(values: Iterable[float]) -> float | None:
    """Return a mean, or None for an empty set."""
    values = list(values)
    return float(np.mean(values)) if values else None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic CSV rows."""
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for field in row.keys():
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tensor_dict(value: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Keep only tensor outputs for structural comparison."""
    return {key: item.detach().cpu() for key, item in value.items() if torch.is_tensor(item)}


def structural_validation(base_config: Path, out_dir: Path) -> dict[str, Any]:
    """Verify source isolation and compare disabled R1 to untouched baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    structural_config = load_options(base_config, "baseline", out_dir)
    dataset = build_dataset(structural_config, "val")
    batch = start_end_collate([dataset[index] for index in range(4)])
    model_inputs, _targets = prepare_batch_inputs(batch[1], structural_config.device)
    checkpoint_path = Path("/private/research-artifact")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    set_seed(SEED)
    disabled, _criterion, _optimizer, _scheduler = setup_model(structural_config)
    disabled.load_state_dict(checkpoint["model"], strict=True)
    disabled.eval()
    disabled.transformer.decoder.posthoc_trace_references = True
    with torch.no_grad():
        disabled_output = disabled(**model_inputs)
    disabled_trace = [item.detach().cpu() for item in disabled.transformer.decoder.posthoc_reference_trace]

    r1_config = load_options(base_config, "r1", out_dir)
    set_seed(SEED)
    r1, _criterion, _optimizer, _scheduler = setup_model(r1_config)
    r1.load_state_dict(checkpoint["model"], strict=True)
    r1.eval()
    r1.transformer.decoder.posthoc_trace_references = True
    with torch.no_grad():
        r1_output = r1(**model_inputs)
    r1_trace = [item.detach().cpu() for item in r1.transformer.decoder.posthoc_reference_trace]

    input_path = out_dir / "structural_inputs.pt"
    output_path = out_dir / "structural_disabled_output.pt"
    input_path_t = str(input_path)
    output_path_t = str(output_path)
    torch.save({key: item.detach().cpu() for key, item in model_inputs.items()}, input_path)
    official_src = "/private/research-artifact"
    official_script = f"""
import sys
import torch
sys.path.insert(0, {official_src!r})
from config import BaseOptions
from evaluate import setup_model
manager = BaseOptions('/private/research-artifact')
manager.parse()
opt = manager.option
model, _criterion, _optimizer, _scheduler = setup_model(opt)
checkpoint = torch.load({str(checkpoint_path)!r}, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
inputs = torch.load({input_path_t!r}, map_location='cpu', weights_only=False)
inputs = {{key: value.to(opt.device) for key, value in inputs.items()}}
with torch.no_grad():
    output = model(**inputs)
torch.save({{key: value.detach().cpu() for key, value in output.items() if torch.is_tensor(value)}}, {output_path_t!r})
"""
    subprocess.run([sys.executable, "-c", official_script], check=True)
    official_output = torch.load(output_path, map_location="cpu", weights_only=False)
    disabled_tensors = tensor_dict(disabled_output)
    official_equal = sorted(official_output) == sorted(disabled_tensors) and all(
        torch.equal(official_output[key], disabled_tensors[key]) for key in official_output
    )

    disabled_param_names = sorted(name for name, _ in disabled.named_parameters())
    r1_param_names = sorted(name for name, _ in r1.named_parameters())
    disabled_parameter_count = sum(parameter.numel() for parameter in disabled.parameters())
    r1_parameter_count = sum(parameter.numel() for parameter in r1.parameters())
    initial_diff = (disabled_trace[0] - r1_trace[0]).abs().max().item()
    first_center_diff = (disabled_trace[1][..., 0] - r1_trace[1][..., 0]).abs().max().item()
    first_width_diff = (disabled_trace[1][..., 1] - r1_trace[1][..., 1]).abs().max().item()
    output_shapes_same = {
        "pred_logits": tuple(disabled_output["pred_logits"].shape) == tuple(r1_output["pred_logits"].shape),
        "pred_spans": tuple(disabled_output["pred_spans"].shape) == tuple(r1_output["pred_spans"].shape),
    }
    finite_r1 = all(torch.isfinite(value).all().item() for value in tensor_dict(r1_output).values())
    initial_widths = r1_trace[0][..., 1]
    expected_widths = r1.query_embed.weight[:, 1].detach().cpu().sigmoid().unsqueeze(1).expand_as(initial_widths)
    initial_widths_same = torch.equal(initial_widths, expected_widths)

    buffer = out_dir / "structural_r1_state.pth"
    torch.save(r1.state_dict(), buffer)
    reloaded, _criterion, _optimizer, _scheduler = setup_model(r1_config)
    reloaded.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
    checkpoint_save_load = True
    for path in [input_path, output_path, buffer]:
        path.unlink(missing_ok=True)

    return {
        "status": "PASS" if all([
            official_equal,
            disabled_param_names == r1_param_names,
            disabled_parameter_count == r1_parameter_count,
            initial_diff == 0.0,
            first_center_diff <= 1e-7,
            output_shapes_same["pred_logits"],
            output_shapes_same["pred_spans"],
            finite_r1,
            initial_widths_same,
            checkpoint_save_load,
        ]) else "FAIL",
        "source_baseline_commit": "45ef471ee47ea75a2141d75bd9cfdb8c45dfc101",
        "untouched_baseline_transformer_sha256": sha256(Path("/private/research-artifact")),
        "official_baseline_checkpoint": str(checkpoint_path),
        "official_baseline_checkpoint_sha256": sha256(checkpoint_path),
        "r1_eps": R1_EPS,
        "disabled_r1_matches_untouched_baseline_forward": official_equal,
        "parameter_names_identical": disabled_param_names == r1_param_names,
        "baseline_parameter_count": disabled_parameter_count,
        "r1_parameter_count": r1_parameter_count,
        "parameter_count_delta": r1_parameter_count - disabled_parameter_count,
        "initial_reference_max_abs_delta": initial_diff,
        "first_update_center_max_abs_delta": first_center_diff,
        "first_update_width_max_abs_delta": first_width_diff,
        "decoder_output_shapes_unchanged": output_shapes_same,
        "attention_reference_shape": list(r1_trace[0].shape),
        "attention_width_convention": "normalized width is the second reference coordinate; sine/modulation code unchanged",
        "r1_outputs_finite": finite_r1,
        "initial_widths_exactly_sigmoid_query_width": initial_widths_same,
        "checkpoint_save_load": checkpoint_save_load,
        "batch_size": len(batch[0]),
        "validation_audio_lengths": [int(mask.sum()) for mask in batch[1]["audio_feat"][1]],
    }


def train_variant(
    opt: EasyDict,
    variant: str,
    train_dataset: StartEndDataset,
    val_dataset: StartEndDataset,
    initial_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Train one arm from the matched shared initialization."""
    out_dir = Path(opt.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model, criterion, optimizer, scheduler = setup_model(opt)
    model.load_state_dict(initial_state, strict=True)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        generator=generator,
    )
    best_score = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(opt.n_epoch):
        model.train()
        criterion.train()
        losses_seen: list[float] = []
        for batch in loader:
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
            outputs = model(**model_inputs)
            loss_dict = criterion(outputs, targets)
            total_loss = sum(
                loss_dict[key] * criterion.weight_dict[key]
                for key in loss_dict
                if key in criterion.weight_dict
            )
            optimizer.zero_grad()
            total_loss.backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            losses_seen.append(float(total_loss.detach().cpu()))
        scheduler.step()
        opt.eval_split_name = "val"
        with torch.no_grad():
            metrics, _meters, _paths = eval_epoch(
                model, val_dataset, opt, f"latest_{variant}_val_preds.jsonl", criterion
            )
        score = float(metrics["brief"]["MR-full-R1@0.7"])
        row = {"epoch": epoch + 1, "train_loss": mean(losses_seen), "val_R1@0.7": score}
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({
                "model": best_state,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "variant": variant,
                "seed": SEED,
                "training_from_scratch": True,
            }, out_dir / "best_checkpoint.pth")
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == opt.n_epoch:
            print(json.dumps({"variant": variant, **row}), flush=True)
    assert best_state is not None
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"best_epoch": best_epoch, "best_val_R1@0.7": best_score, "history": history}


def load_trained_model(opt: EasyDict, checkpoint_path: Path) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load one trained model and criterion."""
    model, criterion, _optimizer, _scheduler = setup_model(opt)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, criterion


def collect_trajectories(
    model: torch.nn.Module,
    dataset: StartEndDataset,
    opt: EasyDict,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Collect all proposal trajectories and final query-level mechanism rows."""
    model.eval()
    model.transformer.decoder.posthoc_trace_references = True
    loader = DataLoader(dataset, collate_fn=start_end_collate, batch_size=opt.eval_bsz, num_workers=0, shuffle=False)
    trajectory_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    prediction_rows: dict[str, Mapping[str, Any]] = {}
    post_processor = PostProcessorDETR(
        clip_length=opt.clip_length,
        min_ts_val=0,
        max_ts_val=300,
        min_w_l=1,
        max_w_l=300,
        move_window_method="left",
        process_func_names=("clip_ts", "round_multiple"),
    )
    with torch.no_grad():
        for batch in loader:
            holder: dict[str, Any] = {}
            original_forward = model.transformer.forward

            def capture_first(*args: Any, **kwargs: Any) -> Any:
                result = original_forward(*args, **kwargs)
                if "trace" not in holder:
                    holder["trace"] = [item.detach().cpu().clone() for item in model.transformer.decoder.posthoc_reference_trace]
                return result

            model.transformer.forward = capture_first
            model_inputs, _targets = prepare_batch_inputs(batch[1], opt.device)
            outputs = model(**model_inputs)
            model.transformer.forward = original_forward
            trace = torch.stack(holder["trace"], dim=0).permute(0, 2, 1, 3).numpy()
            final_spans = outputs["pred_spans"].detach().cpu().numpy()
            scores = F.softmax(outputs["pred_logits"], dim=-1)[..., 0].detach().cpu().numpy()
            raw_predictions: list[dict[str, Any]] = []
            for batch_index, meta in enumerate(batch[0]):
                duration = float(meta["duration"])
                gt = meta["relevant_windows"]
                raw_windows = []
                for slot in range(final_spans.shape[1]):
                    center = float(final_spans[batch_index, slot, 0])
                    width = float(final_spans[batch_index, slot, 1])
                    raw_windows.append([duration * (center - width / 2), duration * (center + width / 2), float(scores[batch_index, slot])])
                raw_windows.sort(key=lambda row: row[2], reverse=True)
                raw_predictions.append({
                    "qid": meta["qid"],
                    "query": meta["query"],
                    "vid": meta["vid"],
                    "pred_relevant_windows": [[float(f"{item:.4f}") for item in row] for row in raw_windows],
                })
            processed = post_processor(raw_predictions)
            for row in processed:
                prediction_rows[str(row["qid"])] = row

            for batch_index, meta in enumerate(batch[0]):
                qid = str(meta["qid"])
                duration = float(meta["duration"])
                gt = meta["relevant_windows"]
                gt_duration = gt_len(gt)
                centers = gt_centers(gt)
                initial_rows: list[dict[str, Any]] = []
                for stage_index, stage in enumerate(STAGES[:3]):
                    spans = np.column_stack([
                        trace[stage_index, batch_index, :, 0] * duration - trace[stage_index, batch_index, :, 1] * duration / 2.0,
                        trace[stage_index, batch_index, :, 0] * duration + trace[stage_index, batch_index, :, 1] * duration / 2.0,
                    ])
                    for slot, span in enumerate(spans):
                        center = float((span[0] + span[1]) / 2.0)
                        width = max(0.0, float(span[1] - span[0]))
                        center_error = min(abs(center - target) for target in centers)
                        best_iou = max(iou(span, target) for target in gt)
                        row = {
                            "variant": variant,
                            "qid": qid,
                            "vid": str(meta["vid"]),
                            "gt_bin": gt_bin(gt),
                            "gt_duration_sec": gt_duration,
                            "audio_duration_sec": duration,
                            "slot": slot,
                            "proposal_rank": int(np.argsort(-scores[batch_index], kind="stable").tolist().index(slot) + 1),
                            "stage": stage,
                            "stage_order": stage_index,
                            "center_sec": center,
                            "width_sec": width,
                            "width_gt_duration_ratio": width / max(gt_duration, 1e-8),
                            "abs_log_width_gt_ratio": abs(math.log(max(width, R1_EPS) / max(gt_duration, R1_EPS))),
                            "nearest_gt_center_distance_sec": center_error,
                            "best_gt_iou": best_iou,
                        }
                        trajectory_rows.append(row)
                        if stage == "initial":
                            initial_rows.append(row)

                final_spans_sec = np.column_stack([
                    final_spans[batch_index, :, 0] * duration - final_spans[batch_index, :, 1] * duration / 2.0,
                    final_spans[batch_index, :, 0] * duration + final_spans[batch_index, :, 1] * duration / 2.0,
                ])
                final_rows: list[dict[str, Any]] = []
                for slot, span in enumerate(final_spans_sec):
                    center = float((span[0] + span[1]) / 2.0)
                    width = max(0.0, float(span[1] - span[0]))
                    center_error = min(abs(center - target) for target in centers)
                    best_iou = max(iou(span, target) for target in gt)
                    row = {
                        "variant": variant,
                        "qid": qid,
                        "vid": str(meta["vid"]),
                        "gt_bin": gt_bin(gt),
                        "gt_duration_sec": gt_duration,
                        "audio_duration_sec": duration,
                        "slot": slot,
                        "proposal_rank": int(np.argsort(-scores[batch_index], kind="stable").tolist().index(slot) + 1),
                        "stage": "final_span",
                        "stage_order": 3,
                        "center_sec": center,
                        "width_sec": width,
                        "width_gt_duration_ratio": width / max(gt_duration, 1e-8),
                        "abs_log_width_gt_ratio": abs(math.log(max(width, R1_EPS) / max(gt_duration, R1_EPS))),
                        "nearest_gt_center_distance_sec": center_error,
                        "best_gt_iou": best_iou,
                    }
                    trajectory_rows.append(row)
                    final_rows.append(row)

                initial_errors = [float(row["nearest_gt_center_distance_sec"]) for row in initial_rows]
                top10 = list(prediction_rows[qid]["pred_relevant_windows"][:10])
                top1 = top10[0][:2]
                top10_iou = max(iou(window[:2], target) for window in top10 for target in gt)
                top1_iou = max(iou(top1, target) for target in gt)
                query_rows.append({
                    "variant": variant,
                    "qid": qid,
                    "vid": str(meta["vid"]),
                    "gt_bin": gt_bin(gt),
                    "gt_duration_sec": gt_duration,
                    "audio_duration_sec": duration,
                    "initial_well_centered_count_le1s": sum(error <= 1.0 for error in initial_errors),
                    "initial_well_centered_count_le2s": sum(error <= 2.0 for error in initial_errors),
                    "oracle10_iou05": float(top10_iou >= 0.5),
                    "oracle10_iou07": float(top10_iou >= 0.7),
                    "center_hit10_le2s": float(any(abs((window[0] + window[1]) / 2.0 - target_center) <= 2.0 for window in top10 for target_center in centers)),
                    "top1_iou_zero": float(top1_iou == 0.0),
                    "r1_iou05": float(top1_iou >= 0.5),
                    "r1_iou07": float(top1_iou >= 0.7),
                    "top1_width_gt_ratio": (float(top1[1]) - float(top1[0])) / max(gt_duration, 1e-8),
                })
    return trajectory_rows, query_rows, prediction_rows


def aggregate_trajectory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate proposal trajectories by stage, duration bin, and center subset."""
    initial_errors = {
        (row["variant"], row["qid"], row["slot"]): float(row["nearest_gt_center_distance_sec"])
        for row in rows
        if row["stage"] == "initial"
    }
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        for bin_name in GT_BINS:
            bin_rows = variant_rows if bin_name == "overall" else [row for row in variant_rows if row["gt_bin"] == bin_name]
            for stage in STAGES:
                stage_rows = [row for row in bin_rows if row["stage"] == stage]
                if not stage_rows:
                    continue
                for subset_name, subset_rows in [
                    ("all", stage_rows),
                    ("initial_center_le1s", [row for row in stage_rows if initial_errors[(row["variant"], row["qid"], row["slot"])] <= 1.0]),
                    ("initial_center_le2s", [row for row in stage_rows if initial_errors[(row["variant"], row["qid"], row["slot"])] <= 2.0]),
                ]:
                    if not subset_rows:
                        continue
                    output.append({
                        "variant": variant,
                        "gt_bin": bin_name,
                        "stage": stage,
                        "subset": subset_name,
                        "N_proposals": len(subset_rows),
                        "median_abs_log_width_gt_ratio": finite_median(row["abs_log_width_gt_ratio"] for row in subset_rows),
                        "median_width_gt_ratio": finite_median(row["width_gt_duration_ratio"] for row in subset_rows),
                        "median_width_sec": finite_median(row["width_sec"] for row in subset_rows),
                        "median_nearest_gt_center_distance_sec": finite_median(row["nearest_gt_center_distance_sec"] for row in subset_rows),
                        "median_best_gt_iou": finite_median(row["best_gt_iou"] for row in subset_rows),
                        "median_audio_duration_sec": finite_median(row["audio_duration_sec"] for row in subset_rows),
                    })
    return output


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate postprocessed retrieval and mechanism metrics."""
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        for bin_name in GT_BINS:
            selected = variant_rows if bin_name == "overall" else [row for row in variant_rows if row["gt_bin"] == bin_name]
            if not selected:
                continue
            output.append({
                "variant": variant,
                "gt_bin": bin_name,
                "N_queries": len(selected),
                "Oracle10_IoU>=0.5": mean(row["oracle10_iou05"] for row in selected),
                "Oracle10_IoU>=0.7": mean(row["oracle10_iou07"] for row in selected),
                "Center_Hit10<=2s": mean(row["center_hit10_le2s"] for row in selected),
                "Top1_IoU=0": mean(row["top1_iou_zero"] for row in selected),
                "R1@0.5": mean(row["r1_iou05"] for row in selected),
                "R1@0.7": mean(row["r1_iou07"] for row in selected),
                "median_top1_width_gt_ratio": finite_median(row["top1_width_gt_ratio"] for row in selected),
            })
    return output


def audio_stratification(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report exact recording-duration strata for initially well-centered proposals."""
    initial_errors = {
        (row["variant"], row["qid"], row["slot"]): float(row["nearest_gt_center_distance_sec"])
        for row in rows
        if row["stage"] == "initial"
    }
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for bin_name in GT_BINS[:-1]:
            selected = [
                row for row in rows
                if row["variant"] == variant and row["gt_bin"] == bin_name
                and row["stage"] == "final_span"
                and initial_errors[(row["variant"], row["qid"], row["slot"])] <= 1.0
            ]
            for duration in sorted({float(row["audio_duration_sec"]) for row in selected}):
                duration_rows = [row for row in selected if float(row["audio_duration_sec"]) == duration]
                output.append({
                    "variant": variant,
                    "gt_bin": bin_name,
                    "audio_duration_sec": duration,
                    "N_proposals": len(duration_rows),
                    "median_final_abs_log_width_gt_ratio": finite_median(row["abs_log_width_gt_ratio"] for row in duration_rows),
                    "median_final_width_gt_ratio": finite_median(row["width_gt_duration_ratio"] for row in duration_rows),
                    "median_final_width_sec": finite_median(row["width_sec"] for row in duration_rows),
                })
    return output


def duration_slopes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute descriptive slopes of final log-width residual versus recording duration."""
    initial_errors = {
        (row["variant"], row["qid"], row["slot"]): float(row["nearest_gt_center_distance_sec"])
        for row in rows
        if row["stage"] == "initial"
    }
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for bin_name in GT_BINS[:-1]:
            grouped: dict[float, list[float]] = defaultdict(list)
            for row in rows:
                if row["variant"] != variant or row["gt_bin"] != bin_name or row["stage"] != "final_span":
                    continue
                if initial_errors[(row["variant"], row["qid"], row["slot"])] > 1.0:
                    continue
                grouped[float(row["audio_duration_sec"])].append(float(row["abs_log_width_gt_ratio"]))
            pairs = [(duration, float(np.median(values))) for duration, values in sorted(grouped.items())]
            slope = None if len(pairs) < 2 else float(np.polyfit([pair[0] for pair in pairs], [pair[1] for pair in pairs], 1)[0])
            output.append({
                "variant": variant,
                "gt_bin": bin_name,
                "N_audio_duration_strata": len(pairs),
                "audio_duration_slope_abs_log_ratio_per_sec": slope,
                "audio_duration_slope_abs_log_ratio_per_100sec": None if slope is None else 100.0 * slope,
            })
    return output


def write_reports(
    out_dir: Path,
    structural: Mapping[str, Any],
    train_results: Mapping[str, Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    duration_rows: Sequence[Mapping[str, Any]],
    slopes: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the R1 output files and return the final summary."""
    mkdirp(str(out_dir))
    trajectory_aggregate = aggregate_trajectory(trajectory_rows)
    metric_aggregate = aggregate_metrics(query_rows)
    write_csv(out_dir / "width_trajectory_by_duration.csv", trajectory_aggregate)
    initial_errors = {
        (row["variant"], row["qid"], row["slot"]): float(row["nearest_gt_center_distance_sec"])
        for row in trajectory_rows
        if row["stage"] == "initial"
    }
    well_centered_rows = [
        row for row in trajectory_rows
        if initial_errors[(row["variant"], row["qid"], row["slot"])] <= 1.0
    ]
    write_csv(out_dir / "well_centered_width_trajectory.csv", well_centered_rows)
    audio_rows = list(duration_rows) + list(slopes)
    write_csv(out_dir / "audio_duration_stratification.csv", audio_rows)
    write_csv(out_dir / "baseline_vs_r1_metrics.csv", metric_aggregate)
    (out_dir / "structural_validation.json").write_text(json.dumps(structural, indent=2) + "\n")

    config_payload = {
        "experiment": "R1_REFINEMENT_GEOMETRY_CAUSAL_PILOT",
        "baseline_commit": provenance["baseline_commit"],
        "source_worktree": provenance["source_worktree"],
        "seed": SEED,
        "train_path": "/private/research-artifact",
        "val_path": "/private/research-artifact",
        "test_path": "/private/research-artifact",
        "audio_features": "/private/research-artifact",
        "text_features": "/private/research-artifact",
        "shared": {
            "num_queries": 10,
            "hidden_dim": 256,
            "enc_layers": 2,
            "dec_layers": 2,
            "batch_size": 32,
            "eval_batch_size": 100,
            "epochs": 200,
            "optimizer": "AdamW",
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "lr_drop": 400,
            "grad_clip": 0.1,
            "max_windows": 5,
            "clip_length": 1,
            "span_loss_type": "l1",
            "set_cost_span": 10,
            "set_cost_giou": 1,
            "set_cost_class": 4,
            "span_loss_coef": 10,
            "giou_loss_coef": 1,
            "label_loss_coef": 4,
            "eos_coef": 0.1,
            "lw_saliency": 1,
            "evaluator": "official standalone_eval through evaluate.eval_epoch",
            "checkpoint_selection": "validation MR-full-R1@0.7",
            "data_loader_generator_seed": SEED,
        },
        "baseline": {"r1_enabled": False, "r1_eps": R1_EPS},
        "r1": {
            "r1_enabled": True,
            "r1_eps": R1_EPS,
            "update": "v_l=log(clamp(w_l,eps,1-eps)); v_next=v_l+delta_w_l; w_next=clamp(exp(v_next),eps,1-eps)",
        },
        "unchanged": [
            "center initialization and refinement code",
            "width initialization",
            "attention semantics and tensor conventions",
            "final normalized (center,width) output",
            "L1/GIoU/classification/saliency losses",
            "Hungarian matcher",
            "ranking and postprocessing",
            "number of trainable parameters",
        ],
    }
    (out_dir / "matched_training_config.yaml").write_text(yaml.safe_dump(config_payload, sort_keys=False))

    by_key = {(row["variant"], row["gt_bin"]): row for row in metric_aggregate}
    traj_key = {(row["variant"], row["gt_bin"], row["stage"], row["subset"]): row for row in trajectory_aggregate}
    primary_bins = ["0-2s", "2-5s"]
    primary_comparisons = {}
    for bin_name in primary_bins:
        base = traj_key.get(("baseline", bin_name, "final_span", "initial_center_le1s"))
        r1 = traj_key.get(("r1", bin_name, "final_span", "initial_center_le1s"))
        primary_comparisons[bin_name] = {
            "baseline_median_abs_log": None if base is None else base["median_abs_log_width_gt_ratio"],
            "r1_median_abs_log": None if r1 is None else r1["median_abs_log_width_gt_ratio"],
            "baseline_median_ratio": None if base is None else base["median_width_gt_ratio"],
            "r1_median_ratio": None if r1 is None else r1["median_width_gt_ratio"],
        }
    primary_residual_improved = all(
        primary_comparisons[bin_name]["r1_median_abs_log"] is not None
        and primary_comparisons[bin_name]["baseline_median_abs_log"] is not None
        and primary_comparisons[bin_name]["r1_median_abs_log"] < primary_comparisons[bin_name]["baseline_median_abs_log"]
        and primary_comparisons[bin_name]["r1_median_ratio"] < primary_comparisons[bin_name]["baseline_median_ratio"]
        for bin_name in primary_bins
    )
    short_localization_deltas = []
    for bin_name in primary_bins:
        base = by_key.get(("baseline", bin_name))
        r1 = by_key.get(("r1", bin_name))
        if base and r1:
            for metric in ["Oracle10_IoU>=0.5", "Oracle10_IoU>=0.7", "R1@0.5", "R1@0.7"]:
                short_localization_deltas.append(float(r1[metric]) - float(base[metric]))
    localization_not_worse = bool(short_localization_deltas) and max(short_localization_deltas) >= 0.0
    long_base = by_key.get(("baseline", "20s+"))
    long_r1 = by_key.get(("r1", "20s+"))
    long_r1_drop = None if not long_base or not long_r1 else float(long_r1["R1@0.7"]) - float(long_base["R1@0.7"])
    long_safety_failure = long_r1_drop is not None and long_r1_drop <= -0.05
    if primary_residual_improved and localization_not_worse and not long_safety_failure:
        decision = "SHORT_SCALE_REFINEMENT_SUPPORTED"
    elif primary_residual_improved and long_safety_failure:
        decision = "TRADEOFF_ONLY"
    elif not primary_residual_improved:
        decision = "REFINEMENT_NOT_SUPPORTED"
    else:
        decision = "INCONCLUSIVE"

    report_lines = [
        "# R1 refinement geometry causal pilot",
        "",
        "This is a single-seed exploratory causal pilot, not a novelty claim.",
        "",
        "## Implementation and protocol",
        "",
        f"- Structural validation: **{structural['status']}**.",
        f"- Training arms: baseline and R1, both from scratch with seed {SEED}.",
        f"- Best checkpoint selection: validation MR-full-R1@0.7; test was used only after selection.",
        f"- R1 epsilon: `{R1_EPS}`; it was fixed before training and not tuned.",
        f"- Parameter count delta: `{structural['parameter_count_delta']}`.",
        "- Only the decoder layer-to-layer normalized width update was changed; center, initialization, attention convention, final geometry, loss, matcher, ranking, and postprocessing were retained.",
        "",
        "## Training outcome",
        "",
    ]
    for variant in VARIANTS:
        result = train_results[variant]
        report_lines.append(f"- `{variant}` best epoch `{result['best_epoch']}`, validation R1@0.7 `{result['best_val_R1@0.7']:.3f}%`.")
    report_lines += [
        "",
        "## Primary mechanism observations",
        "",
        "The primary subset is the initially well-centered proposal population (initial center error <=1 s). Values below are medians over proposal rows, not aggregate means over audio files.",
        "",
    ]
    for bin_name in primary_bins:
        comparison = primary_comparisons[bin_name]
        report_lines.append(
            f"- `{bin_name}` final median |log(width/GT)|: baseline `{comparison['baseline_median_abs_log']}`, R1 `{comparison['r1_median_abs_log']}`; final median width/GT: baseline `{comparison['baseline_median_ratio']}`, R1 `{comparison['r1_median_ratio']}`."
        )
    report_lines += [
        "",
        "## Decision rule and boundary",
        "",
        "The predeclared pilot rule requires both short bins to improve in both primary residual summaries. Localization metrics must not show a uniformly negative short-bin direction, and a >=5 percentage-point 20 s+ R1@0.7 drop without compelling short-scale improvement is a safety failure.",
        f"- Primary residual direction passed: `{primary_residual_improved}`.",
        f"- At least one short-bin localization metric was non-worse: `{localization_not_worse}`.",
        f"- 20 s+ R1@0.7 delta: `{long_r1_drop}`.",
        f"- Decision: **{decision}**.",
        "",
        "The strongest supported claim, if any, is restricted to this implementation, split, one seed, and the measured proposal trajectory. It does not establish a generally useful method, novelty, or that refinement geometry is the only cause of short-event failure.",
    ]
    (out_dir / "implementation_report.md").write_text("\n".join(report_lines) + "\n")

    long_lines = [
        "# Long-event safety (20 s+)",
        "",
        "All values are postprocessed test-set query metrics unless explicitly marked as trajectory statistics.",
        "",
        "| variant | N | Oracle@10 IoU>=0.5 | Oracle@10 IoU>=0.7 | Center Hit@10<=2s | Top1 IoU=0 | R1@0.5 | R1@0.7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = by_key.get((variant, "20s+"))
        if row:
            long_lines.append(
                f"| {variant} | {row['N_queries']} | {row['Oracle10_IoU>=0.5']:.6f} | {row['Oracle10_IoU>=0.7']:.6f} | {row['Center_Hit10<=2s']:.6f} | {row['Top1_IoU=0']:.6f} | {row['R1@0.5']:.6f} | {row['R1@0.7']:.6f} |"
            )
    for stage in STAGES:
        base = traj_key.get(("baseline", "20s+", stage, "initial_center_le1s"))
        r1 = traj_key.get(("r1", "20s+", stage, "initial_center_le1s"))
        if base and r1:
            long_lines.append(
                f"\n- `{stage}` well-centered median |log(width/GT)|: baseline `{base['median_abs_log_width_gt_ratio']}`, R1 `{r1['median_abs_log_width_gt_ratio']}`; median width/GT: baseline `{base['median_width_gt_ratio']}`, R1 `{r1['median_width_gt_ratio']}`."
            )
    long_lines += [
        "",
        f"The predeclared safety flag is `{long_safety_failure}` because the R1@0.7 delta is `{long_r1_drop}` and the threshold is -0.05.",
    ]
    (out_dir / "long_event_safety.md").write_text("\n".join(long_lines) + "\n")

    decision_lines = [
        "# Mechanism decision",
        "",
        f"Decision: **{decision}**.",
        "",
        "## Observations",
        "",
        *[f"- `{bin_name}`: {primary_comparisons[bin_name]}" for bin_name in primary_bins],
        f"- 20 s+ R1@0.7 delta: `{long_r1_drop}`.",
        "",
        "## Interpretation",
        "",
        "This result is an exploratory one-seed matched pilot. A supported result would mean that the changed width update law is a causal component of the measured scale residual under this baseline; it would not mean that the representation, initialization, loss, or attention semantics were independently validated.",
        "",
        "No rescue module, extra seed, post-hoc threshold, or next method was added.",
    ]
    (out_dir / "mechanism_decision.md").write_text("\n".join(decision_lines) + "\n")

    summary = {
        "experiment": "R1_REFINEMENT_GEOMETRY_CAUSAL_PILOT",
        "status": "COMPLETE" if structural["status"] == "PASS" else "INVALID",
        "decision": decision if structural["status"] == "PASS" else "INVALID",
        "implementation": "PASS" if structural["status"] == "PASS" else "FAIL",
        "structural_isolation": structural["status"],
        "parameter_count_changed": structural["parameter_count_delta"] != 0,
        "parameter_count_delta": structural["parameter_count_delta"],
        "primary_mechanism": {
            "subset": "initial center error <=1s",
            "bins": primary_comparisons,
            "metric": "median abs(log(width_l/GT_width)) and median width_l/GT_width by decoder stage",
        },
        "metrics": metric_aggregate,
        "audio_duration_dependence": slopes,
        "long_event_r1_at_0_7_delta": long_r1_drop,
        "long_event_safety_failure": long_safety_failure,
        "train_results": {variant: {key: value for key, value in result.items() if key != "history"} for variant, result in train_results.items()},
        "provenance": dict(provenance),
        "claim_boundary": "Exploratory evidence for refinement geometry as one possible causal component in this matched baseline; not a novel-method or generalization claim.",
        "next_scale_mechanism_authorized": decision == "SHORT_SCALE_REFINEMENT_SUPPORTED",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Run structural validation, matched training, and mechanism reporting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    base_config = Path(args.base_config)
    out_dir = Path(args.output_dir)
    structural = structural_validation(base_config, out_dir)
    print(json.dumps({"structural_validation": structural}, indent=2), flush=True)
    if args.structural_only:
        (out_dir / "structural_validation.json").write_text(json.dumps(structural, indent=2) + "\n")
        return
    if structural["status"] != "PASS":
        (out_dir / "summary.json").write_text(json.dumps({"status": "INVALID", "structural_validation": structural}, indent=2) + "\n")
        raise SystemExit("R1 structural validation failed; training was not started")

    if args.analysis_only:
        test_dataset = build_dataset(load_options(base_config, "baseline", out_dir), "test")
        train_results: dict[str, Mapping[str, Any]] = {}
        trajectory_rows: list[dict[str, Any]] = []
        query_rows: list[dict[str, Any]] = []
        for variant in VARIANTS:
            opt = load_options(base_config, variant, out_dir)
            history_path = Path(opt.results_dir) / "training_history.json"
            history = json.loads(history_path.read_text(encoding="utf-8"))
            best = max(history, key=lambda row: float(row["val_R1@0.7"]))
            train_results[variant] = {
                "best_epoch": int(best["epoch"]),
                "best_val_R1@0.7": float(best["val_R1@0.7"]),
            }
            model, _criterion = load_trained_model(opt, Path(opt.results_dir) / "best_checkpoint.pth")
            variant_trajectory, variant_queries, _predictions = collect_trajectories(model, test_dataset, opt, variant)
            trajectory_rows.extend(variant_trajectory)
            query_rows.extend(variant_queries)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        duration_rows = audio_stratification(trajectory_rows)
        slopes = duration_slopes(trajectory_rows)
        provenance = {
            "baseline_commit": "45ef471ee47ea75a2141d75bd9cfdb8c45dfc101",
            "source_worktree": "/private/research-artifact",
            "base_config": str(base_config),
            "test_data_sha256": sha256(Path("/private/research-artifact")),
            "feature_audio_dir": "/private/research-artifact",
            "feature_text_dir": "/private/research-artifact",
            "seed": SEED,
            "training_from_scratch": True,
            "test_queries": len(test_dataset),
            "analysis_only_recovery": True,
        }
        summary = write_reports(out_dir, structural, train_results, trajectory_rows, query_rows, duration_rows, slopes, provenance)
        print(json.dumps({"summary": summary}, indent=2), flush=True)
        return

    train_dataset = build_dataset(load_options(base_config, "baseline", out_dir), "train")
    val_dataset = build_dataset(load_options(base_config, "baseline", out_dir), "val")
    test_dataset = build_dataset(load_options(base_config, "baseline", out_dir), "test")

    set_seed(SEED)
    base_opt = load_options(base_config, "baseline", out_dir)
    base_model, _criterion, _optimizer, _scheduler = setup_model(base_opt)
    initial_state = {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_results: dict[str, Mapping[str, Any]] = {}
    for variant in VARIANTS:
        opt = load_options(base_config, variant, out_dir)
        train_results[variant] = train_variant(opt, variant, train_dataset, val_dataset, initial_state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trajectory_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        opt = load_options(base_config, variant, out_dir)
        model, _criterion = load_trained_model(opt, Path(opt.results_dir) / "best_checkpoint.pth")
        variant_trajectory, variant_queries, _predictions = collect_trajectories(model, test_dataset, opt, variant)
        trajectory_rows.extend(variant_trajectory)
        query_rows.extend(variant_queries)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    duration_rows = audio_stratification(trajectory_rows)
    slopes = duration_slopes(trajectory_rows)
    provenance = {
        "baseline_commit": "45ef471ee47ea75a2141d75bd9cfdb8c45dfc101",
        "source_worktree": "/private/research-artifact",
        "base_config": str(base_config),
        "test_data_sha256": sha256(Path("/private/research-artifact")),
        "feature_audio_dir": "/private/research-artifact",
        "feature_text_dir": "/private/research-artifact",
        "seed": SEED,
        "training_from_scratch": True,
        "test_queries": len(test_dataset),
    }
    summary = write_reports(out_dir, structural, train_results, trajectory_rows, query_rows, duration_rows, slopes, provenance)
    print(json.dumps({"summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
