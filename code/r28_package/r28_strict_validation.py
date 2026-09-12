#!/usr/bin/env python3
"""Run the R28 frozen EGCG ablation on train/validation splits only.

The QD-DETR checkpoint and Evidence Head checkpoint are frozen. Three
independent rankers are trained on train-split candidate IoUs: no-evidence,
evidence-only, and full EGCG. Metrics are computed only on the validation
split. This script deliberately does not construct or read the test split.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import r14_minimal_trainable_egcg as recovered_r14  # noqa: E402
import r15_trainable_egcg as recovered_r15  # noqa: E402
import r17_trainable_evidence_ranker as recovered_r17  # noqa: E402
from evidence import parameter_count as evidence_parameter_count  # noqa: E402
from ranker import (  # noqa: E402
    EvidenceRanker,
    FeatureCache,
    examples_from_records,
    parameter_count as ranker_parameter_count,
    rank_examples,
    train_ranker,
)


OUT = Path("/private/research-artifact")
CHECKPOINT_DIR = ROOT / "checkpoints" / "egcg_v1.0_seed2023"
FROZEN_EVIDENCE = CHECKPOINT_DIR / "evidence_head_seed2023.pt"
BASE_CHECKPOINT = Path("/private/research-artifact")
R27_METRICS = Path("/private/research-artifact")
SEED = 2023
RANKER_EPOCHS = 5
RANKER_BATCH_SIZE = 8
RANKER_LR = 1e-3
RANKER_WD = 1e-4
RANKER_CLIP = 0.1
RANKER_MARGIN = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_frozen_model(opt: Any) -> tuple[Any, Any]:
    model, criterion = recovered_r15.build_model(opt)
    payload = torch.load(FROZEN_EVIDENCE, map_location="cpu", weights_only=False)
    model.evidence_head.load_state_dict(payload["state_dict"], strict=True)
    model.evidence_head.eval()
    model.qd_model.eval()
    return model, criterion


def replace_candidate_features(
    records: Sequence[Mapping[str, Any]], mode: str
) -> list[Dict[str, Any]]:
    """Make an explicit input ablation while preserving candidate intervals."""
    result: list[Dict[str, Any]] = []
    for record in records:
        current = dict(record)
        candidates = []
        for candidate in record["e3_full_egcg"]:
            item = dict(candidate)
            if mode == "no_evidence":
                item["evidence_score"] = 0.0
            elif mode == "evidence_only":
                item["qd_score"] = 0.0
            elif mode != "full":
                raise ValueError(f"unknown ablation mode: {mode}")
            candidates.append(item)
        current["e3_full_egcg"] = candidates
        result.append(current)
    return result


def mask_examples(examples: Sequence[Mapping[str, Any]], mode: str) -> list[Dict[str, Any]]:
    """Zero all ranker inputs except the fields named by each control."""
    output: list[Dict[str, Any]] = []
    for example in examples:
        current = dict(example)
        current["audio"] = example["audio"].clone()
        current["query"] = example["query"].clone()
        current["scalar"] = example["scalar"].clone()
        if mode == "no_evidence":
            current["scalar"][:, 1:] = 0.0
        elif mode == "evidence_only":
            current["audio"].zero_()
            current["query"].zero_()
            current["scalar"].zero_()
            current["scalar"][:, 1] = example["scalar"][:, 1]
        elif mode != "full":
            raise ValueError(f"unknown ablation mode: {mode}")
        output.append(current)
    return output


def train_control(
    name: str,
    records: Sequence[Mapping[str, Any]],
    cache: FeatureCache,
    opt: Any,
) -> tuple[EvidenceRanker, float, int]:
    set_seed(SEED)
    prepared = replace_candidate_features(records, name)
    examples = mask_examples(examples_from_records(prepared, cache), name)
    model = EvidenceRanker().to(opt.device)
    started = time.perf_counter()
    train_ranker(
        model,
        examples,
        opt.device,
        RANKER_EPOCHS,
        RANKER_BATCH_SIZE,
        RANKER_LR,
        RANKER_WD,
        RANKER_CLIP,
        RANKER_MARGIN,
        OUT / "checkpoints" / f"{name}_ranker_seed{SEED}.pt",
        OUT / "checkpoints" / f"{name}_ranker_optimizer_seed{SEED}.pt",
        OUT / "logs" / f"{name}_ranker.jsonl",
    )
    return model, time.perf_counter() - started, len(examples)


def rank_control(
    model: EvidenceRanker,
    records: Sequence[Mapping[str, Any]],
    cache: FeatureCache,
    opt: Any,
    mode: str,
) -> list[Dict[str, Any]]:
    prepared = replace_candidate_features(records, mode)
    examples = mask_examples(examples_from_records(prepared, cache), mode)
    ranked = rank_examples(model, examples, opt.device)
    result = []
    for record, candidates in zip(prepared, ranked):
        current = dict(record)
        current["e2_ranked"] = candidates
        result.append(current)
    return result


def evaluate(records: Sequence[Mapping[str, Any]], key: str, score_key: str) -> Dict[str, Any]:
    candidate = recovered_r17.candidate_metrics(records, key)
    subsets = {
        name: [
            record
            for record in records
            if name == "all"
            or recovered_r14.duration_bin(recovered_r14.gt_duration(record["meta"])) == name
        ]
        for name in ["all", *recovered_r14.BINS.keys()]
    }
    official = {
        name: recovered_r17.official(subset, key, score_key)
        for name, subset in subsets.items()
    }
    return {
        "N": len(records),
        "official": {
            "overall": official["all"],
            "by_gt_duration_bin": {
                name: {"N": len(subsets[name]), "metrics": official[name]}
                for name in recovered_r14.BINS
            },
        },
        "candidate": candidate,
        "definition": f"validation records key={key}, score={score_key}",
    }


def metric(system: Mapping[str, Any], name: str, key: str) -> float | None:
    value = (
        system["official"]["overall"].get(key)
        if name == "all"
        else system["official"]["by_gt_duration_bin"][name]["metrics"].get(key)
    )
    return None if value is None else float(value)


def format_value(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def save_predictions(
    validation_records: Sequence[Mapping[str, Any]],
    system_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    with (OUT / "validation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(validation_records):
            payload: Dict[str, Any] = {
                "qid": record["meta"]["qid"],
                "vid": record["meta"]["vid"],
                "gt_windows": record["meta"]["relevant_windows"],
            }
            for name, rows in system_records.items():
                candidates = rows[index]["e2_ranked"] if name != "A_QD_DETR" else rows[index]["e0_qd"]
                payload[name] = [
                    [float(c["start"]), float(c["end"]), float(c["ranker_score"] if name != "A_QD_DETR" else c["qd_score"])]
                    for c in candidates[:100]
                ]
            handle.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")


def build_reports(metrics: Mapping[str, Any], runtime: Mapping[str, Any], complexity: Mapping[str, Any]) -> None:
    names = ["A_QD_DETR", "B_no_evidence_ranker", "C_evidence_only_ranker", "D_full_EGCG"]
    lines = [
        "# R28 strict frozen EGCG ablation",
        "",
        "## Status",
        "",
        "`R28_FINAL_VALIDATION_COMPLETE`",
        "",
        "## Protocol",
        "",
        "- EGCG v1.0 was frozen before this run; QD-DETR and the Evidence Head checkpoint were not updated.",
        "- Ranker controls B, C, and D were trained for five epochs using train-split candidates and train-split GT only.",
        "- All reported metrics use the official validation split. The test split was not constructed or read by this script.",
        "- A is the original QD-DETR ranking. B retains candidate audio/query features and QD score while zeroing evidence, position, and duration scalars. C retains only the evidence scalar; audio/query/QD/position/duration inputs are zeroed. D retains the full recovered feature set.",
        "",
        "## Overall validation metrics",
        "",
        "| System | R1@0.5 | R1@0.7 | mAP | CandidateRecall@10@0.5 | Oracle@10@0.7 | Top1–Oracle@10 gap@0.7 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        system = metrics["systems"][name]
        c = system["candidate"]["all"]
        lines.append(
            f"| {name} | {format_value(metric(system, 'all', 'MR-full-R1@0.5'))} | {format_value(metric(system, 'all', 'MR-full-R1@0.7'))} | {format_value(metric(system, 'all', 'MR-full-mAP'))} | "
            f"{format_value(100*c['CandidateRecall@10@0.5']['rate'] if c['CandidateRecall@10@0.5']['rate'] is not None else None)}% | "
            f"{format_value(100*c['Oracle@10@0.7']['rate'] if c['Oracle@10@0.7']['rate'] is not None else None)}% | "
            f"{format_value(100*c['Top1_vs_Oracle10_gap@0.7'] if c['Top1_vs_Oracle10_gap@0.7'] is not None else None)}%"
        )
    lines += [
        "",
        "## Short-duration validation",
        "",
        "| System | 0–2s N | 0–2s R1@0.7 | 2–5s N | 2–5s R1@0.7 | 0–2s CandidateRecall@10@0.5 | 2–5s CandidateRecall@10@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        system = metrics["systems"][name]
        b0 = system["official"]["by_gt_duration_bin"]["0-2s"]
        b1 = system["official"]["by_gt_duration_bin"]["2-5s"]
        c0 = system["candidate"]["0-2s"]["CandidateRecall@10@0.5"]["rate"]
        c1 = system["candidate"]["2-5s"]["CandidateRecall@10@0.5"]["rate"]
        lines.append(
            f"| {name} | {b0['N']} | {format_value(float(b0['metrics'].get('MR-full-R1@0.7', 0)))} | {b1['N']} | "
            f"{format_value(float(b1['metrics'].get('MR-full-R1@0.7', 0)))} | {format_value(100*c0 if c0 is not None else None)}% | {format_value(100*c1 if c1 is not None else None)}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The ablation isolates whether the evidence scalar is useful beyond the candidate feature and QD score. Any difference between B and D is the full evidence-aware input effect under the frozen candidate pool and fixed QD-DETR.",
        "",
    ]
    (OUT / "ablation_report.md").write_text("\n".join(lines) + "\n")

    complexity_lines = [
        "# R28 complexity and runtime",
        "",
        f"- QD-DETR parameters: {complexity['qd_detr_parameters']}",
        f"- Evidence Head parameters: {complexity['evidence_head_parameters']}",
        f"- Ranker parameters: {complexity['ranker_parameters']}",
        f"- Total EGCG additional parameters: {complexity['additional_parameters']}",
        f"- Additional parameters over QD-DETR: {complexity['additional_percentage']:.6f}%",
        "",
        "## Training time",
        "",
        f"- Evidence Head frozen checkpoint provenance training time (R27): {runtime['evidence_training_seconds']:.3f} seconds",
        f"- No-evidence ranker (R28): {runtime['ranker_training_seconds']['no_evidence']:.3f} seconds",
        f"- Evidence-only ranker (R28): {runtime['ranker_training_seconds']['evidence_only']:.3f} seconds",
        f"- Full EGCG ranker (R28): {runtime['ranker_training_seconds']['full']:.3f} seconds",
        "",
        "## Validation inference cost",
        "",
        f"- QD-DETR validation forward: {runtime['qd_validation_seconds']:.3f} seconds",
        f"- Full EGCG validation inference: {runtime['egcg_validation_seconds']:.3f} seconds",
        f"- Absolute overhead: {runtime['egcg_validation_seconds'] - runtime['qd_validation_seconds']:.3f} seconds",
        f"- Relative overhead: {runtime['inference_overhead_percentage']:.2f}%",
        f"- QD peak allocated GPU memory: {runtime['qd_peak_gpu_memory_bytes']} bytes",
        f"- EGCG peak allocated GPU memory: {runtime['egcg_peak_gpu_memory_bytes']} bytes",
        "",
        "All timing and memory figures are from the validation-only R28 run on the RTX 5090 runtime.",
        "",
    ]
    (OUT / "complexity_report.md").write_text("\n".join(complexity_lines))


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing R28 output directory: {OUT}")
    if not FROZEN_EVIDENCE.exists():
        raise SystemExit(f"Missing frozen Evidence Head checkpoint: {FROZEN_EVIDENCE}")
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    (OUT / "logs").mkdir()

    set_seed(SEED)
    opt = recovered_r14.load_options()
    train_dataset = recovered_r14.make_dataset(opt, "train")
    validation_dataset = recovered_r14.make_dataset(opt, "val")
    model, _criterion = load_frozen_model(opt)
    cache = FeatureCache()

    train_records = recovered_r15.enrich(recovered_r14.collect_records(model, train_dataset, opt))
    validation_records = recovered_r15.enrich(recovered_r14.collect_records(model, validation_dataset, opt))

    trained: Dict[str, tuple[EvidenceRanker, float, int]] = {}
    for mode in ["no_evidence", "evidence_only", "full"]:
        ranker, elapsed, count = train_control(mode, train_records, cache, opt)
        trained[mode] = (ranker, elapsed, count)

    system_records: Dict[str, Sequence[Mapping[str, Any]]] = {"A_QD_DETR": validation_records}
    for name, mode in [
        ("B_no_evidence_ranker", "no_evidence"),
        ("C_evidence_only_ranker", "evidence_only"),
        ("D_full_EGCG", "full"),
    ]:
        system_records[name] = rank_control(trained[mode][0], validation_records, cache, opt, mode)

    systems = {
        "A_QD_DETR": evaluate(validation_records, "e0_qd", "qd_score"),
        "B_no_evidence_ranker": evaluate(system_records["B_no_evidence_ranker"], "e2_ranked", "ranker_score"),
        "C_evidence_only_ranker": evaluate(system_records["C_evidence_only_ranker"], "e2_ranked", "ranker_score"),
        "D_full_EGCG": evaluate(system_records["D_full_EGCG"], "e2_ranked", "ranker_score"),
    }
    save_predictions(validation_records, system_records)

    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    qd_started = time.perf_counter()
    loader = torch.utils.data.DataLoader(
        validation_dataset,
        collate_fn=recovered_r14.start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=0,
        shuffle=False,
    )
    with torch.no_grad():
        for batch in loader:
            model.qd_model(**recovered_r14.prepare_batch_inputs(batch[1], opt.device)[0])
    sync()
    qd_seconds = time.perf_counter() - qd_started
    qd_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    egcg_started = time.perf_counter()
    timed_records = recovered_r15.enrich(recovered_r14.collect_records(model, validation_dataset, opt))
    timed_examples = examples_from_records(timed_records, cache)
    rank_examples(trained["full"][0], mask_examples(timed_examples, "full"), opt.device)
    sync()
    egcg_seconds = time.perf_counter() - egcg_started
    egcg_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    qd_parameters = sum(parameter.numel() for parameter in model.qd_model.parameters())
    evidence_parameters = evidence_parameter_count(model.evidence_head)
    ranker_parameters = ranker_parameter_count(trained["full"][0])
    complexity = {
        "qd_detr_parameters": qd_parameters,
        "evidence_head_parameters": evidence_parameters,
        "ranker_parameters": ranker_parameters,
        "additional_parameters": evidence_parameters + ranker_parameters,
        "additional_percentage": 100.0 * (evidence_parameters + ranker_parameters) / qd_parameters,
    }
    old_runtime = json.loads(R27_METRICS.read_text()).get("runtime", {}) if R27_METRICS.exists() else {}
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "evidence_training_seconds": float(old_runtime.get("evidence_training_seconds", 0.0)),
        "ranker_training_seconds": {
            "no_evidence": trained["no_evidence"][1],
            "evidence_only": trained["evidence_only"][1],
            "full": trained["full"][1],
        },
        "qd_validation_seconds": qd_seconds,
        "egcg_validation_seconds": egcg_seconds,
        "inference_overhead_percentage": 100.0 * (egcg_seconds - qd_seconds) / qd_seconds if qd_seconds else None,
        "qd_peak_gpu_memory_bytes": qd_peak,
        "egcg_peak_gpu_memory_bytes": egcg_peak,
    }
    metrics = {
        "status": "R28_FINAL_VALIDATION_COMPLETE",
        "seed": SEED,
        "split_protocol": {
            "training_split": "train",
            "evaluation_split": "validation",
            "test_split_constructed_or_read": False,
            "validation_used_for_training_or_selection": False,
        },
        "dataset_sizes": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "train_candidate_examples": {name: value[2] for name, value in trained.items()},
        },
        "systems": systems,
        "configuration": {
            "ranker_epochs": RANKER_EPOCHS,
            "ranker_batch_size": RANKER_BATCH_SIZE,
            "ranker_optimizer": "AdamW",
            "ranker_learning_rate": RANKER_LR,
            "ranker_weight_decay": RANKER_WD,
            "ranker_grad_clip": RANKER_CLIP,
            "ranker_margin": RANKER_MARGIN,
            "ranker_positive_iou": 0.5,
            "controls": {
                "B_no_evidence": "candidate audio/query features + QD score; evidence, position, duration scalars zeroed",
                "C_evidence_only": "evidence score only; audio/query/QD/position/duration zeroed",
                "D_full": "all recovered candidate features",
            },
        },
        "complexity": complexity,
        "runtime": runtime,
        "provenance": {
            "egcg_commit": git_commit(),
            "base_checkpoint": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)},
            "frozen_evidence_checkpoint": {"path": str(FROZEN_EVIDENCE), "sha256": sha256(FROZEN_EVIDENCE)},
            "raw_audio": False,
        },
    }
    write_json(OUT / "metrics.json", metrics)
    build_reports(metrics, runtime, complexity)
    (OUT / "reproduction_note.md").write_text(
        "# R27 reproduction difference and R28 freeze\n\n"
        "Historical R18 seed-2023 full EGCG R1@0.7: 25.61.\n\n"
        "Clean R27 seed-2023 full EGCG R1@0.7: 26.80.\n\n"
        "The frozen QD-DETR baseline reproduced exactly. The remaining numerical difference is attributable to the trainable EGCG components and their recovered runtime/training behavior; it was not corrected by changing implementation choices. The EGCG components were trained only on the official train split, and R28 evaluation used only the validation split. No test split was constructed or read in this R28 script, so there is no test leakage.\n"
    )

    manifest = {
        "status": metrics["status"],
        "egcg_version": "v1.0",
        "egcg_commit": git_commit(),
        "split_protocol": metrics["split_protocol"],
        "environment_files": [str(ROOT / "environment.yml"), str(ROOT / "requirements.txt")],
        "frozen_checkpoints": [],
        "output_artifacts": [],
    }
    for path in sorted(CHECKPOINT_DIR.glob("*")):
        if path.is_file():
            manifest["frozen_checkpoints"].append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "final_manifest.json":
            manifest["output_artifacts"].append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(OUT / "final_manifest.json", manifest)
    model.close()
    print(json.dumps({"status": metrics["status"], "output": str(OUT), "egcg_commit": git_commit()}))


if __name__ == "__main__":
    main()
