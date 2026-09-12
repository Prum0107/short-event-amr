#!/usr/bin/env python3
"""Run the clean seed-2023 EGCG reproduction gate."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import r14_minimal_trainable_egcg as recovered_r14  # noqa: E402
import r15_trainable_egcg as recovered_r15  # noqa: E402
import r17_trainable_evidence_ranker as recovered_r17  # noqa: E402
from evidence import parameter_count as evidence_parameter_count  # noqa: E402
from evidence import train_evidence_head  # noqa: E402
from pipeline import EGCGInferencePipeline  # noqa: E402
from ranker import (  # noqa: E402
    EvidenceRanker,
    FeatureCache,
    examples_from_records,
    parameter_count as ranker_parameter_count,
    rank_examples,
    train_ranker,
)


OUT = Path("/private/research-artifact")
CHECKPOINT_DIR = OUT / "checkpoints"
LOG_DIR = OUT / "logs"
CONFIG_PATH = ROOT / "configs/seed2023.yaml"
BASE_CHECKPOINT = Path("/private/research-artifact")
TOLERANCE_PP = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def metric_value(system: Mapping[str, Any], duration_bin: str | None, metric: str) -> float | None:
    if duration_bin is None:
        return system["official"]["overall"].get(metric)
    return system["official"]["by_gt_duration_bin"][duration_bin]["metrics"].get(metric)


def compare_historical(current: Mapping[str, Any], historical_path: Path) -> Dict[str, Any]:
    historical = json.loads(historical_path.read_text())["per_seed"]["2023"]
    checks: list[Dict[str, Any]] = []
    for system in ["E0_QD", "E1_evidence_head_only", "E2_evidence_candidate_generation", "E3_evidence_ranker_only", "E4_full_EGCG"]:
        for metric in ["MR-full-R1@0.5", "MR-full-R1@0.7", "MR-full-mAP"]:
            current_value = metric_value(current[system], None, metric)
            historical_value = historical[system]["official"]["overall"].get(metric)
            difference = None if current_value is None or historical_value is None else float(current_value) - float(historical_value)
            checks.append({"system": system, "bin": "all", "metric": metric, "current": current_value, "historical": historical_value, "difference_pp": difference, "within_tolerance": difference is not None and abs(difference) <= TOLERANCE_PP})
        for duration_bin in ["0-2s", "2-5s"]:
            metric = "MR-full-R1@0.7"
            current_value = metric_value(current[system], duration_bin, metric)
            historical_value = historical[system]["official"]["by_gt_duration_bin"][duration_bin]["metrics"].get(metric)
            difference = None if current_value is None or historical_value is None else float(current_value) - float(historical_value)
            checks.append({"system": system, "bin": duration_bin, "metric": metric, "current": current_value, "historical": historical_value, "difference_pp": difference, "within_tolerance": difference is not None and abs(difference) <= TOLERANCE_PP})
    failed = [check for check in checks if not check["within_tolerance"]]
    return {"tolerance_percentage_points": TOLERANCE_PP, "checks": checks, "failed_checks": failed, "pass": not failed}


def compact_prediction(candidate: Mapping[str, Any], score_key: str) -> list[float]:
    return [float(candidate["start"]), float(candidate["end"]), float(candidate[score_key])]


def save_predictions(records: Sequence[Mapping[str, Any]]) -> None:
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "qid": record["meta"]["qid"],
                "vid": record["meta"]["vid"],
                "duration_bin": recovered_r14.duration_bin(recovered_r14.gt_duration(record["meta"])),
                "gt_windows": record["meta"]["relevant_windows"],
                "E0_QD_top10": [compact_prediction(c, "qd_score") for c in record["e0_qd"][:10]],
                "E1_head_only_top10": [compact_prediction(c, "qd_score") for c in record["e1_head_only"][:10]],
                "E2_candidate_generation_top10": [compact_prediction(c, "qd_score") for c in record["e2_qd_plus_evidence"][:10]],
                "E3_ranker_only_top10": [compact_prediction(c, "ranker_score") for c in record["e3_ranker_only"][:10]],
                "E4_full_EGCG_top10": [compact_prediction(c, "ranker_score") for c in record["e2_ranked"][:10]],
                "E4_full_EGCG_top100": [compact_prediction(c, "ranker_score") for c in record["e2_ranked"][:100]],
            }
            handle.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")


def qd_inference_seconds(model: Any, dataset: Any, opt: Any) -> float:
    loader = torch.utils.data.DataLoader(dataset, collate_fn=recovered_r14.start_end_collate, batch_size=opt.eval_bsz, num_workers=0, shuffle=False)
    started = time.perf_counter()
    model.qd_model.eval()
    with torch.no_grad():
        for batch in loader:
            model_inputs, _targets = recovered_r14.prepare_batch_inputs(batch[1], opt.device)
            model.qd_model(**model_inputs)
    sync()
    return time.perf_counter() - started


def egcg_inference_seconds(model: Any, dataset: Any, opt: Any, ranker: Any, cache: FeatureCache) -> float:
    started = time.perf_counter()
    records = recovered_r15.enrich(recovered_r14.collect_records(model, dataset, opt))
    examples = examples_from_records(records, cache)
    rank_examples(ranker, examples, opt.device)
    sync()
    return time.perf_counter() - started


def make_reports(metrics: Dict[str, Any], runtime: Dict[str, Any], complexity: Dict[str, Any], gate: Dict[str, Any]) -> None:
    (OUT / "complexity_report.md").write_text("\n".join([
        "# R27 complexity report", "",
        f"- QD-DETR parameters: {complexity['qd_detr_parameters']}",
        f"- Evidence Head parameters: {complexity['evidence_head_parameters']}",
        f"- Ranker parameters: {complexity['ranker_parameters']}",
        f"- Total EGCG parameters: {complexity['total_egcg_parameters']}",
        f"- Additional parameters: {complexity['additional_parameters']}",
        f"- Additional percentage over QD-DETR: {complexity['additional_percentage']:.6f}%", "",
        "Counts are derived from the exact recovered layer definitions; no architecture was changed.", "",
    ]) + "\n")
    (OUT / "runtime_report.md").write_text("\n".join([
        "# R27 runtime report", "",
        f"- Python: {runtime['python']}",
        f"- PyTorch: {runtime['torch']}",
        f"- CUDA runtime: {runtime['cuda']}",
        f"- GPU: {runtime['gpu']}",
        f"- Evidence Head training seconds: {runtime['evidence_training_seconds']:.3f}",
        f"- Ranker training seconds: {runtime['ranker_training_seconds']:.3f}",
        f"- QD-DETR test forward seconds: {runtime['qd_inference_seconds']:.3f}",
        f"- EGCG test inference seconds: {runtime['egcg_inference_seconds']:.3f}",
        f"- Peak GPU memory allocated bytes: {runtime['peak_gpu_memory_allocated']}", "",
        "Test inference timings were measured only after both train-split training phases completed.", "",
    ]) + "\n")
    historical_lines = [
        "# R27 reproduction report", "", "## Status", "",
        f"`{'R27_REPRODUCTION_PASS' if gate['pass'] else 'R27_REPRODUCTION_MISMATCH'}`", "",
        "## Protocol", "",
        "The clean runner uses the recovered R14–R18 implementation choices, frozen QD-DETR, official train split for both learned components, seed 2023, and no test access until Evidence Head and Ranker training completed.", "",
        f"Predeclared material mismatch tolerance: ±{TOLERANCE_PP:.2f} percentage points for overall R1@0.5, R1@0.7, mAP, and short-bin R1@0.7.", "",
        "## Gate comparison", "",
        "| System | Bin | Metric | Current | Historical R18 seed-2023 | Difference (pp) | Within tolerance |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for check in gate["checks"]:
        historical_lines.append(f"| {check['system']} | {check['bin']} | {check['metric']} | {check['current']} | {check['historical']} | {check['difference_pp']} | {check['within_tolerance']} |")
    historical_lines += ["", "## Interpretation", "", "The gate is a reproducibility check, not a method improvement claim. If it fails, strict ablations are not scientifically interpretable and are not run.", ""]
    (OUT / "reproduction_report.md").write_text("\n".join(historical_lines) + "\n")
    if not (OUT / "ablation_report.md").exists():
        (OUT / "ablation_report.md").write_text("# R27 strict ablation\n\nNot run before the reproduction gate.\n")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing R27 output directory: {OUT}")
    OUT.mkdir(parents=True)
    CHECKPOINT_DIR.mkdir()
    LOG_DIR.mkdir()
    config_text = CONFIG_PATH.read_text()
    (OUT / "config.yaml").write_text(config_text)
    (OUT / "training.log").write_text("")

    seed = 2023
    set_seed(seed)
    opt = recovered_r14.load_options()
    train_dataset = recovered_r14.make_dataset(opt, "train")
    model, criterion = recovered_r15.build_model(opt)
    qd_parameters = sum(parameter.numel() for parameter in model.qd_model.parameters())
    model.qd_model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    evidence_history, evidence_optimizer, evidence_elapsed = train_evidence_head(
        model, criterion, train_dataset, opt, 0.5, 3, 16, 1e-3, 1e-4, 0.1,
        CHECKPOINT_DIR / "evidence_head_seed2023.pt",
        CHECKPOINT_DIR / "evidence_head_optimizer_seed2023.pt",
        LOG_DIR / "evidence_head.jsonl",
    )
    train_records = recovered_r15.enrich(recovered_r14.collect_records(model, train_dataset, opt))
    cache = FeatureCache()
    train_examples = examples_from_records(train_records, cache)
    ranker = EvidenceRanker().to(opt.device)
    ranker_history, ranker_optimizer, ranker_elapsed = train_ranker(
        ranker, train_examples, opt.device, 5, 8, 1e-3, 1e-4, 0.1, 0.1,
        CHECKPOINT_DIR / "ranker_seed2023.pt",
        CHECKPOINT_DIR / "ranker_optimizer_seed2023.pt",
        LOG_DIR / "ranker.jsonl",
    )

    # Test is instantiated and accessed only after both train-split optimizers finish.
    test_dataset = recovered_r14.make_dataset(opt, "test")
    qd_seconds = qd_inference_seconds(model, test_dataset, opt)
    test_records = recovered_r15.enrich(recovered_r14.collect_records(model, test_dataset, opt))
    test_examples = examples_from_records(test_records, cache)
    ranked_test = rank_examples(ranker, test_examples, opt.device)
    records = EGCGInferencePipeline.attach_ranked(test_records, ranked_test)
    systems = EGCGInferencePipeline.system_metrics(records)
    gate = compare_historical(systems, Path("/private/research-artifact"))
    egcg_seconds = egcg_inference_seconds(model, test_dataset, opt, ranker, cache)
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    evidence_parameters = evidence_parameter_count(model.evidence_head)
    ranker_parameters = ranker_parameter_count(ranker)
    complexity = {
        "qd_detr_parameters": qd_parameters,
        "evidence_head_parameters": evidence_parameters,
        "ranker_parameters": ranker_parameters,
        "total_egcg_parameters": evidence_parameters + ranker_parameters,
        "additional_parameters": evidence_parameters + ranker_parameters,
        "additional_percentage": 100.0 * (evidence_parameters + ranker_parameters) / qd_parameters,
    }
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "evidence_training_seconds": evidence_elapsed["seconds"],
        "ranker_training_seconds": ranker_elapsed["seconds"],
        "qd_inference_seconds": qd_seconds,
        "egcg_inference_seconds": egcg_seconds,
        "peak_gpu_memory_allocated": peak_memory,
    }
    metrics = {
        "status": "R27_REPRODUCTION_PASS" if gate["pass"] else "R27_REPRODUCTION_MISMATCH",
        "seed": seed,
        "dataset_sizes": {"train": len(train_dataset), "test": len(test_dataset), "train_ranker_examples": len(train_examples)},
        "configuration": {
            "evidence_lambda": 0.5,
            "evidence_epochs": 3,
            "evidence_batch_size": 16,
            "evidence_optimizer": "AdamW",
            "ranker_epochs": 5,
            "ranker_batch_size": 8,
            "ranker_optimizer": "AdamW",
            "ranker_margin": 0.1,
            "ranker_positive_iou": 0.5,
            "test_access_after_training": True,
        },
        "systems": systems,
        "historical_comparison": gate,
        "training": {"evidence_history": evidence_history, "ranker_history": ranker_history},
        "complexity": complexity,
        "runtime": runtime,
        "provenance": {
            "base_checkpoint": str(BASE_CHECKPOINT),
            "base_checkpoint_sha256": sha256(BASE_CHECKPOINT),
            "recovered_source": "/private/research-artifact",
            "clean_source": str(SRC),
            "raw_audio": False,
        },
    }
    write_json(OUT / "metrics.json", metrics)
    save_predictions(records)
    manifest = {
        "status": metrics["status"],
        "artifacts": [],
        "base_checkpoint": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)},
        "evidence_head_parameters": evidence_parameters,
        "ranker_parameters": ranker_parameters,
    }
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            manifest["artifacts"].append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(OUT / "checkpoints_manifest.json", manifest)
    make_reports(metrics, runtime, complexity, gate)
    model.close()
    print(json.dumps({"status": metrics["status"], "output": str(OUT), "failed_gate_checks": len(gate["failed_checks"])}))


if __name__ == "__main__":
    main()
