"""Inference-only audio-duration mechanism decomposition for the official QD-DETR baseline."""

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


FINE_BINS = [
    ("0-1s", 0.0, 1.0), ("1-2s", 1.0, 2.0), ("2-3s", 2.0, 3.0),
    ("3-5s", 3.0, 5.0), ("5-10s", 5.0, 10.0), ("10-20s", 10.0, 20.0),
    ("20s+", 20.0, None),
]
BROAD_BINS = [("0-2s", 0.0, 2.0), ("2-5s", 2.0, 5.0), ("5-10s", 5.0, 10.0), ("10s+", 10.0, None)]
SHORT_BINS = {"0-2s", "2-5s"}
SEED = 20260912


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def finite(values: Iterable[float]) -> np.ndarray:
    return np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)


def median(values: Iterable[float]) -> float | None:
    array = finite(values)
    return None if len(array) == 0 else float(np.median(array))


def mean(values: Iterable[float]) -> float | None:
    array = finite(values)
    return None if len(array) == 0 else float(np.mean(array))


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    a, b = finite(x), finite(y)
    if len(a) != len(b) or len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def rank_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.argsort(array, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    a, b = finite(x), finite(y)
    if len(a) != len(b) or len(a) < 3:
        return None
    return pearson(rank_array(a), rank_array(b))


def bin_for(value: float, bins: Sequence[tuple[str, float, float | None]]) -> str:
    for name, lower, upper in bins:
        if value >= lower and (upper is None or value < upper):
            return name
    raise ValueError(f"unhandled value {value}")


def query_words(query: str) -> int:
    return len(str(query).split())


def word_bin(n_words: int) -> str:
    if n_words <= 5:
        return "0-5w"
    if n_words <= 10:
        return "6-10w"
    if n_words <= 15:
        return "11-15w"
    return "16w+"


def gt_longest(row: Mapping[str, Any]) -> float:
    return max(float(end) - float(start) for start, end in row["relevant_windows"])


def gt_union_duration(row: Mapping[str, Any]) -> float:
    intervals = sorted((float(start), float(end)) for start, end in row["relevant_windows"])
    total = 0.0
    left = right = None
    for start, end in intervals:
        if right is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (0.0 if right is None else right - left)


def gt_centers(row: Mapping[str, Any]) -> list[float]:
    return [(float(start) + float(end)) / 2.0 for start, end in row["relevant_windows"]]


def temporal_iou(pred: Sequence[float], target: Sequence[float]) -> float:
    left = max(float(pred[0]), float(target[0]))
    right = min(float(pred[1]), float(target[1]))
    intersection = max(0.0, right - left)
    union = max(float(pred[1]), float(target[1])) - min(float(pred[0]), float(target[0]))
    return intersection / union if union > 0 else 0.0


def interval_distance(point: float, start: float, end: float) -> float:
    if start <= point <= end:
        return 0.0
    return min(abs(point - start), abs(point - end))


def prediction_metrics(row: Mapping[str, Any], gt_windows: Sequence[Sequence[float]]) -> dict[str, float]:
    predictions = row["pred_relevant_windows"]
    ious = [max(temporal_iou(pred[:2], gt) for gt in gt_windows) for pred in predictions]
    top1_center = (float(predictions[0][0]) + float(predictions[0][1])) / 2.0
    gt_center_error = min(abs(top1_center - center) for center in [
        (float(start) + float(end)) / 2.0 for start, end in gt_windows
    ])
    return {
        "r1_iou05": float(ious[0] >= 0.5),
        "r1_iou07": float(ious[0] >= 0.7),
        "oracle10_iou05": float(max(ious[:10]) >= 0.5),
        "oracle10_iou07": float(max(ious[:10]) >= 0.7),
        "final_width_sec": float(predictions[0][1]) - float(predictions[0][0]),
        "final_center_sec": top1_center,
        "center_error_sec": gt_center_error,
        "oracle10_best_iou": float(max(ious[:10])),
    }


def load_options(config_path: Path, worktree: Path, output: Path) -> EasyDict:
    sys.path.insert(0, str(worktree / "src"))
    from config import BaseOptions  # type: ignore

    manager = BaseOptions(str(config_path))
    manager.parse()
    opt = EasyDict(dict(manager.option))
    opt.results_dir = str(output / "runtime")
    opt.eval_split_name = "test"
    opt.ckpt_filepath = str(worktree / "results/best_checkpoint.pth")
    opt.test_path = str(worktree / "data/castella_test_release.jsonl")
    opt.a_feat_dir = str(worktree / "features/castella/clap")
    opt.t_feat_dir = str(worktree / "features/castella/clap_text")
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    return opt


def build_dataset(opt: EasyDict):
    from dataset import StartEndDataset  # type: ignore

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


def official_predictions(outputs: Mapping[str, torch.Tensor], metas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from span_utils import span_cxw_to_xx  # type: ignore

    spans = span_cxw_to_xx(outputs["pred_spans"].detach().cpu())
    scores = outputs["pred_logits"].detach().cpu().softmax(-1)[..., 0]
    rows: list[dict[str, Any]] = []
    for index, meta in enumerate(metas):
        duration = float(meta["duration"])
        cur = torch.cat([spans[index] * duration, scores[index, :, None]], dim=1).numpy()
        cur = sorted(cur.tolist(), key=lambda item: item[2], reverse=True)
        processed = []
        for start, end, score in cur:
            start = float(np.clip(np.round(start), 0.0, duration))
            end = float(np.clip(np.round(end), 0.0, duration))
            processed.append([start, end, float(f"{score:.4f}")])
        rows.append({"qid": meta["qid"], "query": meta["query"], "vid": meta["vid"], "pred_relevant_windows": processed})
    return rows


def token_evidence(
    scores: Sequence[float],
    valid_mask: Sequence[bool],
    row: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, float]:
    score = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    duration = float(row["duration"])
    token_count = int(valid.sum())
    centers = np.arange(len(score), dtype=np.float64) + 0.5
    gt_mask = np.zeros(len(score), dtype=bool)
    for start, end in row["relevant_windows"]:
        start, end = float(start), float(end)
        gt_mask |= valid & (centers + 0.5 > start) & (centers - 0.5 < end)
    false_mask = valid & ~gt_mask
    if not gt_mask.any():
        gt_mask[min(max(int(float(row["relevant_windows"][0][0])), 0), len(score) - 1)] = True
    gt_scores = score[gt_mask]
    false_scores = score[false_mask]
    best_gt = float(np.max(gt_scores))
    best_false = float(np.max(false_scores)) if len(false_scores) else float("nan")
    gt_rank = 1 + int(np.sum(score[valid] > best_gt))
    gt_percentile = float(np.mean(score[valid] <= best_gt)) if token_count else float("nan")
    false_over_gt = int(np.sum(false_scores > best_gt)) if len(false_scores) else 0
    gt_indices = np.flatnonzero(gt_mask)
    nearby_mask = false_mask.copy()
    for index in np.flatnonzero(false_mask):
        if min(interval_distance(float(centers[index]), float(start), float(end)) for start, end in row["relevant_windows"]) > 5.0:
            nearby_mask[index] = False
    nearby_scores = score[nearby_mask]
    random_best: list[float] = []
    false_indices = np.flatnonzero(false_mask)
    sample_size = min(len(gt_indices), len(false_indices))
    if sample_size:
        for _ in range(100):
            sampled = rng.choice(false_indices, size=sample_size, replace=False)
            random_best.append(float(np.max(score[sampled])))

    high_threshold = float(np.quantile(score[valid], 0.90)) if token_count else float("nan")
    false_peak_count = 0
    high_peak_count = 0
    for index in np.flatnonzero(false_mask):
        left = score[index - 1] if index > 0 and valid[index - 1] else -np.inf
        right = score[index + 1] if index + 1 < len(score) and valid[index + 1] else -np.inf
        if score[index] >= left and score[index] >= right:
            false_peak_count += 1
            if score[index] >= high_threshold:
                high_peak_count += 1
    return {
        "valid_tokens": float(token_count),
        "gt_token_count": float(len(gt_indices)),
        "gt_token_fraction": float(len(gt_indices) / token_count) if token_count else float("nan"),
        "best_gt_saliency": best_gt,
        "mean_gt_saliency": float(np.mean(gt_scores)),
        "best_false_saliency": best_false,
        "gt_vs_false_margin": float(best_gt - best_false) if math.isfinite(best_false) else float("nan"),
        "gt_saliency_percentile": gt_percentile,
        "gt_saliency_rank": float(gt_rank),
        "false_peaks_above_gt": float(false_over_gt),
        "false_peak_count": float(false_peak_count),
        "high_saliency_distractor_peak_count": float(high_peak_count),
        "nearby_best_saliency": float(np.max(nearby_scores)) if len(nearby_scores) else float("nan"),
        "gt_vs_nearby_margin": float(best_gt - np.max(nearby_scores)) if len(nearby_scores) else float("nan"),
        "random_negative_best_mean": mean(random_best),
        "gt_vs_random_negative_margin": float(best_gt - np.mean(random_best)) if random_best else float("nan"),
        "top1_gt_token": float(gt_rank <= 1),
        "top3_gt_token": float(gt_rank <= 3),
        "top5_gt_token": float(gt_rank <= 5),
        "top10_gt_token": float(gt_rank <= 10),
        "gt_token_center_sec": float(np.mean(centers[gt_indices])),
        "audio_duration_sec": duration,
    }


def initial_scale_fields(row: Mapping[str, Any], query_embed: np.ndarray) -> dict[str, float]:
    duration = float(row["duration"])
    gt_width = gt_longest(row)
    widths = 1.0 / (1.0 + np.exp(-query_embed[:, 1]))
    centers = 1.0 / (1.0 + np.exp(-query_embed[:, 0])) * duration
    distances = np.asarray([min(abs(center - target) for target in gt_centers(row)) for center in centers])
    nearest = int(np.argmin(distances))
    return {
        "initial_width_norm_median": float(np.median(widths)),
        "initial_width_sec_median": float(np.median(widths) * duration),
        "initial_width_gt_ratio_median": float(np.median(widths) * duration / gt_width),
        "initial_width_gt_ratio_nearest_center": float(widths[nearest] * duration / gt_width),
        "initial_nearest_center_error_sec": float(distances[nearest]),
    }


def run_baseline(worktree: Path, config_path: Path, output: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    opt = load_options(config_path, worktree, output)
    from dataset import prepare_batch_inputs, start_end_collate  # type: ignore
    from evaluate import setup_model  # type: ignore

    model, _criterion, _optimizer, _scheduler = setup_model(opt)
    checkpoint = worktree / "results/best_checkpoint.pth"
    state = torch.load(checkpoint, map_location="cpu")["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    dataset = build_dataset(opt)
    loader = DataLoader(dataset, collate_fn=start_end_collate, batch_size=int(opt.eval_bsz), num_workers=0, shuffle=False)
    gt_by_qid = {str(row["qid"]): row for row in rows}
    query_embed = state["query_embed.weight"].detach().cpu().numpy().astype(np.float64)
    output_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    with torch.no_grad():
        for batch in loader:
            metas, batched = batch
            model_inputs, _targets = prepare_batch_inputs(batched, opt.device)
            outputs = model(**model_inputs)
            predictions = official_predictions(outputs, metas)
            saliency = outputs["saliency_scores"].detach().cpu().numpy()
            audio_mask = outputs["audio_mask"].detach().cpu().numpy().astype(bool)
            for index, (meta, prediction) in enumerate(zip(metas, predictions)):
                source = gt_by_qid[str(meta["qid"])]
                base = dict(source)
                base.update({
                    "qid": str(source["qid"]),
                    "vid": str(source["vid"]),
                    "gt_duration_sec": gt_longest(source),
                    "gt_union_duration_sec": gt_union_duration(source),
                    "gt_count": len(source["relevant_windows"]),
                    "query_word_count": query_words(str(source["query"])),
                    "query_char_count": len(str(source["query"])),
                    "gt_duration_bin_fine": bin_for(gt_longest(source), FINE_BINS),
                    "gt_duration_bin_broad": bin_for(gt_longest(source), BROAD_BINS),
                    "normalized_gt_width": gt_longest(source) / float(source["duration"]),
                })
                base.update(prediction_metrics(prediction, source["relevant_windows"]))
                base.update(initial_scale_fields(source, query_embed))
                base.update(token_evidence(saliency[index], audio_mask[index], source, rng))
                base["qd_top1_center_hit_le_2s"] = float(base["center_error_sec"] <= 2.0)
                base["short_failure_iou07"] = float(base["r1_iou07"] == 0.0)
                base["final_width_gt_ratio"] = base["final_width_sec"] / base["gt_duration_sec"]
                base["qd_top1_center_sec"] = base["final_center_sec"]
                base["final_center_error_sec"] = base["center_error_sec"]
                output_rows.append(base)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_rows


def add_duration_groups(rows: list[dict[str, Any]]) -> None:
    for broad in sorted(SHORT_BINS):
        selected = [row for row in rows if row["gt_duration_bin_broad"] == broad]
        threshold = float(np.median([float(row["audio_duration_sec"]) for row in selected])) if selected else float("nan")
        for row in selected:
            row["audio_duration_group"] = "shorter_or_equal_audio" if float(row["audio_duration_sec"]) <= threshold else "longer_audio"
            row["audio_duration_group_threshold_sec"] = threshold
    for row in rows:
        if row["gt_duration_bin_broad"] not in SHORT_BINS:
            row["audio_duration_group"] = "not_short_primary"
            row["audio_duration_group_threshold_sec"] = float("nan")
        row["gt_count_bin"] = "1" if int(row["gt_count"]) == 1 else "2" if int(row["gt_count"]) == 2 else "3+"
        row["query_word_bin"] = word_bin(int(row["query_word_count"]))


def matched_duration_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    metrics = ["r1_iou05", "r1_iou07", "oracle10_iou07", "qd_top1_center_hit_le_2s", "final_width_gt_ratio", "center_error_sec", "normalized_gt_width"]
    for broad in ["0-2s", "2-5s"]:
        selected = [row for row in rows if row["gt_duration_bin_broad"] == broad]
        cells: dict[tuple[str, str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: {"shorter_or_equal_audio": [], "longer_audio": []})
        for row in selected:
            cells[(str(row["gt_duration_bin_fine"]), str(row["gt_count_bin"]), str(row["query_word_bin"]))][str(row["audio_duration_group"])].append(row)
        pairs: list[dict[str, Any]] = []
        for cell_key, groups in sorted(cells.items()):
            candidates = [(abs(float(a["gt_duration_sec"]) - float(b["gt_duration_sec"])) + 0.10 * abs(int(a["query_word_count"]) - int(b["query_word_count"])), i, j)
                          for i, a in enumerate(groups["shorter_or_equal_audio"]) for j, b in enumerate(groups["longer_audio"])]
            used_a: set[int] = set()
            used_b: set[int] = set()
            for _cost, i, j in sorted(candidates):
                if i in used_a or j in used_b:
                    continue
                used_a.add(i)
                used_b.add(j)
                pairs.append((groups["shorter_or_equal_audio"][i], groups["longer_audio"][j], cell_key))
        pair_rows: list[dict[str, Any]] = []
        for short, long, cell_key in pairs:
            record: dict[str, Any] = {
                "row_type": "matched_pair",
                "gt_duration_bin_broad": broad,
                "matching_cell": "|".join(cell_key),
                "short_qid": short["qid"], "long_qid": long["qid"],
                "short_vid": short["vid"], "long_vid": long["vid"],
                "short_audio_duration_sec": short["audio_duration_sec"], "long_audio_duration_sec": long["audio_duration_sec"],
                "short_gt_duration_sec": short["gt_duration_sec"], "long_gt_duration_sec": long["gt_duration_sec"],
                "short_gt_count": short["gt_count"], "long_gt_count": long["gt_count"],
                "short_query_word_count": short["query_word_count"], "long_query_word_count": long["query_word_count"],
            }
            for metric in metrics:
                record[f"short_{metric}"] = short[metric]
                record[f"long_{metric}"] = long[metric]
                record[f"long_minus_short_{metric}"] = float(long[metric]) - float(short[metric])
            pair_rows.append(record)
        output.extend(pair_rows)
        summary[broad] = {
            "matched_pairs": len(pair_rows),
            "median_long_minus_short": {metric: median(row[f"long_minus_short_{metric}"] for row in pair_rows) for metric in metrics},
            "mean_long_minus_short": {metric: mean(row[f"long_minus_short_{metric}"] for row in pair_rows) for metric in metrics},
            "matched_audio_duration_sec": {
                "short_median": median(row["short_audio_duration_sec"] for row in pair_rows),
                "long_median": median(row["long_audio_duration_sec"] for row in pair_rows),
            },
        }
    return output, summary


def fit_linear(rows: Sequence[Mapping[str, Any]], outcome: str, predictors: Sequence[str]) -> dict[str, Any]:
    data = [(float(row[outcome]), [float(row[predictor]) for predictor in predictors]) for row in rows
            if all(math.isfinite(float(row.get(key, float("nan")))) for key in [outcome, *predictors])]
    if len(data) < max(20, len(predictors) + 5):
        return {"N": len(data), "r2": None, "standardized_betas": {key: None for key in predictors}}
    y = np.asarray([item[0] for item in data], dtype=np.float64)
    x = np.asarray([item[1] for item in data], dtype=np.float64)
    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(), y.std()
    keep = x_std > 0
    if not np.all(keep) or y_std == 0:
        return {"N": len(data), "r2": None, "standardized_betas": {key: None for key in predictors}}
    xz = (x - x_mean) / x_std
    design = np.column_stack([np.ones(len(xz)), xz])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    ss_total = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / ss_total if ss_total else None
    return {"N": len(data), "r2": r2, "standardized_betas": {key: float(value / y_std) for key, value in zip(predictors, beta[1:])}}


def normalized_width_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    outcomes = ["final_width_gt_ratio", "short_failure_iou07", "r1_iou07", "oracle10_iou07"]
    model_predictors = {
        "AUDIO_DURATION_ONLY": ["audio_duration_sec"],
        "GT_ABSOLUTE_DURATION_ONLY": ["gt_duration_sec"],
        "NORMALIZED_GT_WIDTH_ONLY": ["normalized_gt_width"],
        "INITIAL_WIDTH_GT_ONLY": ["initial_width_gt_ratio_median"],
        "AUDIO_DURATION_PLUS_INITIAL_WIDTH_GT": ["audio_duration_sec", "initial_width_gt_ratio_median"],
        "AUDIO_DURATION_PLUS_NORMALIZED_GT_WIDTH": ["audio_duration_sec", "normalized_gt_width"],
    }
    bands = FINE_BINS
    for band, lower, upper in bands:
        selected = [row for row in rows if float(row["gt_duration_sec"]) >= lower and (upper is None or float(row["gt_duration_sec"]) < upper)]
        for outcome in outcomes:
            for model, predictors in model_predictors.items():
                fit = fit_linear(selected, outcome, predictors)
                output.append({
                    "gt_duration_band": band,
                    "outcome": outcome,
                    "model": model,
                    "N": fit["N"],
                    "r2": fit["r2"],
                    "standardized_betas": json.dumps(fit["standardized_betas"], sort_keys=True),
                    "median_audio_duration_sec": median(row["audio_duration_sec"] for row in selected),
                    "median_normalized_gt_width": median(row["normalized_gt_width"] for row in selected),
                    "median_initial_width_gt_ratio": median(row["initial_width_gt_ratio_median"] for row in selected),
                })
    return output


def mediation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for broad in ["0-2s", "2-5s"]:
        selected = [row for row in rows if row["gt_duration_bin_broad"] == broad]
        for outcome in ["final_width_gt_ratio", "short_failure_iou07", "r1_iou07"]:
            for model, predictors in {
                "A_DURATION_ONLY": ["audio_duration_sec"],
                "B_INITIAL_WIDTH_GT_ONLY": ["initial_width_gt_ratio_median"],
                "C_BOTH": ["audio_duration_sec", "initial_width_gt_ratio_median"],
            }.items():
                fit = fit_linear(selected, outcome, predictors)
                output.append({
                    "gt_duration_bin_broad": broad,
                    "outcome": outcome,
                    "model": model,
                    "N": fit["N"],
                    "r2": fit["r2"],
                    "standardized_betas": json.dumps(fit["standardized_betas"], sort_keys=True),
                    "median_audio_duration_sec": median(row["audio_duration_sec"] for row in selected),
                    "median_initial_width_gt_ratio": median(row["initial_width_gt_ratio_median"] for row in selected),
                })
    return output


def evidence_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}
    metrics = ["gt_token_fraction", "best_gt_saliency", "gt_saliency_percentile", "gt_vs_false_margin", "gt_vs_nearby_margin", "false_peaks_above_gt", "high_saliency_distractor_peak_count", "gt_saliency_rank"]
    for broad in ["0-2s", "2-5s"]:
        for group in ["all", "shorter_or_equal_audio", "longer_audio"]:
            selected = [row for row in rows if row["gt_duration_bin_broad"] == broad and (group == "all" or row["audio_duration_group"] == group)]
            if not selected:
                continue
            record: dict[str, Any] = {"gt_duration_bin_broad": broad, "audio_duration_group": group, "N": len(selected), "median_audio_duration_sec": median(row["audio_duration_sec"] for row in selected)}
            for metric in metrics:
                record[f"median_{metric}"] = median(row[metric] for row in selected)
            output.append(record)
        selected = [row for row in rows if row["gt_duration_bin_broad"] == broad]
        correlations[broad] = {metric: spearman([row["audio_duration_sec"] for row in selected], [row[metric] for row in selected]) for metric in metrics}
    return output, correlations


def classify(rows: Sequence[Mapping[str, Any]], matched_summary: Mapping[str, Any], normalized: Sequence[Mapping[str, Any]], mediation: Sequence[Mapping[str, Any]], evidence_corr: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any], str]:
    short = [row for row in rows if row["gt_duration_bin_broad"] in SHORT_BINS]
    evidence = {
        "matched": matched_summary,
        "evidence_correlations": evidence_corr,
        "mediation": {},
        "normalized": {},
    }
    for broad in ["0-2s", "2-5s"]:
        med = [row for row in mediation if row["gt_duration_bin_broad"] == broad and row["outcome"] == "short_failure_iou07"]
        evidence["mediation"][broad] = {row["model"]: row for row in med}
    for band in ["0-1s", "1-2s", "2-3s", "3-5s"]:
        selected = [row for row in normalized if row["gt_duration_band"] == band and row["outcome"] == "short_failure_iou07"]
        evidence["normalized"][band] = {row["model"]: row for row in selected}

    # H2: global competition rises while local GT percentile does not show a matching decline.
    h2_values = []
    h4_values = []
    for broad in ["0-2s", "2-5s"]:
        corr = evidence_corr.get(broad, {})
        h2_values.append(float(corr.get("false_peaks_above_gt") or 0.0) > 0.10 and float(corr.get("gt_saliency_rank") or 0.0) > 0.10)
        h4_values.append(float(corr.get("gt_saliency_percentile") or 0.0) < -0.10 and float(corr.get("gt_vs_false_margin") or 0.0) < -0.10)
    h2 = "SUPPORTED" if all(h2_values) else "PARTIALLY_SUPPORTED" if any(h2_values) else "NOT_SUPPORTED"
    h4 = "SUPPORTED" if all(h4_values) else "PARTIALLY_SUPPORTED" if any(h4_values) else "NOT_SUPPORTED"

    # H3: adding initial scale improves descriptive fit and reduces the duration coefficient magnitude.
    h3_hits = []
    for broad in ["0-2s", "2-5s"]:
        med = evidence["mediation"][broad]
        a, c = med.get("A_DURATION_ONLY", {}), med.get("C_BOTH", {})
        r2a, r2c = a.get("r2"), c.get("r2")
        beta_a = json.loads(a.get("standardized_betas", "{}")) if a else {}
        beta_c = json.loads(c.get("standardized_betas", "{}")) if c else {}
        beta_a_duration = beta_a.get("audio_duration_sec")
        beta_c_duration = beta_c.get("audio_duration_sec")
        beta_a_value = 0.0 if beta_a_duration is None else float(beta_a_duration)
        beta_c_value = 0.0 if beta_c_duration is None else float(beta_c_duration)
        duration_shrink = 1.0 - abs(beta_c_value) / max(abs(beta_a_value), 1e-8)
        h3_hits.append(r2a is not None and r2c is not None and (float(r2c) - float(r2a) >= 0.02 or duration_shrink >= 0.20))
        evidence["mediation"][broad]["duration_r2_gain_joint"] = None if r2a is None or r2c is None else float(r2c) - float(r2a)
        evidence["mediation"][broad]["duration_beta_shrink_joint"] = duration_shrink
    h3 = "SUPPORTED" if all(h3_hits) else "PARTIALLY_SUPPORTED" if any(h3_hits) else "NOT_SUPPORTED"

    # H1: normalized width adds within-band descriptive explanatory value.
    h1_hits = []
    for band, models in evidence["normalized"].items():
        d = models.get("AUDIO_DURATION_ONLY", {}).get("r2")
        n = models.get("NORMALIZED_GT_WIDTH_ONLY", {}).get("r2")
        joint = models.get("AUDIO_DURATION_PLUS_NORMALIZED_GT_WIDTH", {}).get("r2")
        if d is not None and n is not None and joint is not None:
            h1_hits.append(float(n) > float(d) and float(joint) - float(d) >= 0.02)
    h1 = "SUPPORTED" if len(h1_hits) >= 2 and all(h1_hits) else "PARTIALLY_SUPPORTED" if any(h1_hits) else "NOT_SUPPORTED"

    # H5: compare crude unadjusted long/short difference to the matched-cell difference.
    h5_hits = []
    for broad in ["0-2s", "2-5s"]:
        selected = [row for row in short if row["gt_duration_bin_broad"] == broad]
        long_rows = [row for row in selected if row["audio_duration_group"] == "longer_audio"]
        short_rows = [row for row in selected if row["audio_duration_group"] == "shorter_or_equal_audio"]
        crude = mean(row["short_failure_iou07"] for row in long_rows) - mean(row["short_failure_iou07"] for row in short_rows)
        matched = (matched_summary.get(broad, {}).get("mean_long_minus_short", {}).get("short_failure_iou07"))
        if crude is not None and matched is not None and abs(float(crude)) > 1e-8:
            h5_hits.append(abs(float(matched)) <= 0.5 * abs(float(crude)))
            evidence.setdefault("composition", {})[broad] = {"crude_failure_difference": crude, "matched_failure_difference": matched}
    h5 = "SUPPORTED" if h5_hits and all(h5_hits) else "PARTIALLY_SUPPORTED" if any(h5_hits) else "NOT_SUPPORTED"

    statuses = {
        "H1_NORMALIZED_COORDINATE_EFFECT": h1,
        "H2_SEARCH_SPACE_COMPETITION": h2,
        "H3_INITIAL_SCALE_MEDIATION": h3,
        "H4_GT_LOCAL_EVIDENCE_DEGRADATION": h4,
        "H5_DATASET_COMPOSITION_CONFOUND": h5,
    }
    supported = [name for name, status in statuses.items() if status == "SUPPORTED"]
    if len(supported) >= 2:
        branch = "MULTIPLE_COUPLED_AUDIO_LENGTH_FACTORS"
    elif h2 == "SUPPORTED":
        branch = "GLOBAL_SEARCH_SPACE"
    elif h4 == "SUPPORTED":
        branch = "LOCAL_REPRESENTATION"
    elif h3 == "SUPPORTED":
        branch = "INITIAL_SCALE"
    elif h1 == "SUPPORTED":
        branch = "NORMALIZED_COORDINATE_GEOMETRY"
    elif h5 == "SUPPORTED":
        branch = "INCONCLUSIVE"
    else:
        branch = "INCONCLUSIVE"
    return statuses, evidence, branch


def write_reports(output: Path, rows: Sequence[Mapping[str, Any]], matched_summary: Mapping[str, Any], evidence_corr: Mapping[str, Any], statuses: Mapping[str, str], evidence: Mapping[str, Any], branch: str, provenance: Mapping[str, Any]) -> None:
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.4f}" if isinstance(value, (float, int)) else str(value)

    lines = [
        "# Audio-duration mechanism decomposition",
        "",
        "This is an inference-only diagnostic of the unchanged official QD-DETR baseline. It does not establish causality and does not test a new method.",
        "",
        "## Fixed implementation",
        "",
        "- GT duration bins use the longest annotated interval; local evidence uses the union of all intervals.",
        "- Audio tokens are the official one-second CLAP feature tokens after the unchanged dataset loader and TEF construction.",
        "- Predictions use the official foreground ranking, second-based conversion, clipping, and one-second rounding.",
        "- Matched comparisons control fine GT-duration band, GT-count bin, and query-word-length bin; `vid` is retained as the audio cluster identifier.",
        "",
        "## Duration association after stratification",
        "",
    ]
    for broad in ["0-2s", "2-5s"]:
        summary = matched_summary.get(broad, {})
        lines.append(f"- **{broad}**: {summary.get('matched_pairs', 0)} matched pairs; median long-minus-short R1@0.7 = {fmt(summary.get('median_long_minus_short', {}).get('r1_iou07'))}; median long-minus-short final width/GT = {fmt(summary.get('median_long_minus_short', {}).get('final_width_gt_ratio'))}; median long-minus-short center error = {fmt(summary.get('median_long_minus_short', {}).get('center_error_sec'))} s.")
    lines += ["", "## Search-space versus local evidence", ""]
    for broad in ["0-2s", "2-5s"]:
        corr = evidence_corr.get(broad, {})
        lines.append(f"- **{broad}**: Spearman(audio duration, GT token fraction)={fmt(corr.get('gt_token_fraction'))}; GT rank={fmt(corr.get('gt_saliency_rank'))}; GT percentile={fmt(corr.get('gt_saliency_percentile'))}; false peaks above GT={fmt(corr.get('false_peaks_above_gt'))}; high-saliency distractor peaks={fmt(corr.get('high_saliency_distractor_peak_count'))}.")
    lines += ["", "## Hypothesis status", "", "| Hypothesis | Status |", "|---|---|"]
    for name, status in statuses.items():
        lines.append(f"| {name} | **{status}** |")
    lines += ["", f"## Dominant branch: `{branch}`", "", "The selected branch is a diagnostic priority, not a method proposal. The context-length counterfactual is separately blocked because the stored-feature interface cannot vary context while preserving the model's normalized position coordinate system without synthetic replacement audio.", "", "## Provenance", "", f"```json\n{json.dumps(dict(provenance), indent=2, ensure_ascii=False)}\n```", ""]
    (output / "hypothesis_assessment.md").write_text("\n".join(lines), encoding="utf-8")

    next_lines = [
        "# Next scientific branch",
        "",
        f"Selected branch: **{branch}**.",
        "",
        "This branch is selected from the fixed decomposition summaries. No next experiment or method is implemented here.",
        "",
        "The same-event context-length probe is **BLOCKED** for this stored-feature interface: `PositionEmbeddingSine` computes positions from the cumulative valid-audio mask and normalizes by the last cumulative valid position. Removing a prefix or interior context therefore changes normalized positions; zeroing omitted features would inject an artificial input. A clean local/medium/full counterfactual requires a representation interface that preserves the original coordinate system.",
        "",
        "A further trained causal pilot is not justified by this audit alone; the evidence remains descriptive and the counterfactual mechanism is not identified.",
        "",
    ]
    (output / "next_scientific_branch.md").write_text("\n".join(next_lines), encoding="utf-8")

    context = [
        "# Context-length probe validity",
        "",
        "## Status: BLOCKED",
        "",
        "The official model receives `src_aud_mask` and computes normalized sine positions with `mask.cumsum(1) / mask.cumsum(1)[:, -1:]`. The same mask also determines the transformer padding mask. Consequently, a local or medium context made by changing the mask changes the position coordinate system, including the retained event's normalized positions. Cropping the feature tensor resets positions, and filling removed context with zeros creates synthetic audio. Neither is a same-event context-only counterfactual under the stated validity criterion.",
        "",
        "No context-length probe was run and no counterfactual numbers are reported.",
        "",
    ]
    (output / "context_length_probe.md").write_text("\n".join(context), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.worktree / "data/castella_test_release.jsonl")
    result_rows = run_baseline(args.worktree, args.config, args.output, rows)
    add_duration_groups(result_rows)
    matched, matched_summary = matched_duration_rows(result_rows)
    normalized = normalized_width_rows(result_rows)
    mediation = mediation_rows(result_rows)
    local_global, evidence_corr = evidence_rows(result_rows)
    statuses, evidence, branch = classify(result_rows, matched_summary, normalized, mediation, evidence_corr)

    write_csv(args.output / "search_space_analysis.csv", result_rows)
    write_csv(args.output / "matched_duration_analysis.csv", matched)
    write_csv(args.output / "normalized_width_analysis.csv", normalized)
    write_csv(args.output / "initial_scale_mediation.csv", mediation)
    write_csv(args.output / "local_vs_global_evidence.csv", local_global)
    provenance = {
        "baseline_commit": "45ef471ee47ea75a2141d75bd9cfdb8c45dfc101",
        "baseline_checkpoint": str(args.worktree / "results/best_checkpoint.pth"),
        "baseline_checkpoint_sha256": sha256(args.worktree / "results/best_checkpoint.pth"),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "test_jsonl": str(args.worktree / "data/castella_test_release.jsonl"),
        "test_jsonl_sha256": sha256(args.worktree / "data/castella_test_release.jsonl"),
        "source_qd_detr_sha256": sha256(args.worktree / "src/qd_detr.py"),
        "source_position_encoding_sha256": sha256(args.worktree / "src/position_encoding.py"),
        "source_dataset_sha256": sha256(args.worktree / "src/dataset.py"),
        "sample_count": len(result_rows),
        "random_seed": SEED,
        "clip_length_sec": 1.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    write_reports(args.output, result_rows, matched_summary, evidence_corr, statuses, evidence, branch, provenance)
    summary = {
        "status": "COMPLETE",
        "experiment": "AUDIO_DURATION_MECHANISM_DECOMPOSITION",
        "scope": "official baseline, inference-only, no training or method change",
        "N": len(result_rows),
        "matched_summary": matched_summary,
        "evidence_correlations": evidence_corr,
        "hypothesis_status": statuses,
        "dominant_next_branch": branch,
        "context_length_probe": "BLOCKED",
        "another_trained_causal_pilot_justified": False,
        "provenance": provenance,
        "files": ["experiment_card.md", "search_space_analysis.csv", "matched_duration_analysis.csv", "normalized_width_analysis.csv", "initial_scale_mediation.csv", "local_vs_global_evidence.csv", "context_length_probe.md", "hypothesis_assessment.md", "next_scientific_branch.md", "summary.json"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "audit.log").write_text(json.dumps({"status": "COMPLETE", "N": len(result_rows), "branch": branch, "hypothesis_status": statuses}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "N": len(result_rows), "branch": branch, "hypothesis_status": statuses}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
