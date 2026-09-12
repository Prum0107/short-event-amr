#!/usr/bin/env python3
"""Evaluate frozen EGCG checkpoints on the official test split only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch


ROOT = Path("/private/research-artifact")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import r14_minimal_trainable_egcg as recovered_r14  # noqa: E402
import r15_trainable_egcg as recovered_r15  # noqa: E402
import r17_trainable_evidence_ranker as recovered_r17  # noqa: E402
from ranker import EvidenceRanker, FeatureCache, examples_from_records, rank_examples  # noqa: E402


OUT = Path("/private/research-artifact")
FROZEN_DIR = ROOT / "checkpoints/egcg_v1.0_seed2023"
EVIDENCE_CHECKPOINT = FROZEN_DIR / "evidence_head_seed2023.pt"
FULL_RANKER_CHECKPOINT = FROZEN_DIR / "ranker_seed2023.pt"
NO_EVIDENCE_CHECKPOINT = Path("/tmp/egcg_r30_7mkd2rwh/B_no_evidence_ranker.pt")
BASE_CHECKPOINT = Path("/private/research-artifact")
R27_METRICS = Path("/private/research-artifact")
SEED = 2023
EFFICIENCY_RUNS = 3


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
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


def state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_models(opt: Any) -> tuple[Any, EvidenceRanker, EvidenceRanker]:
    model, _criterion = recovered_r15.build_model(opt)
    evidence_payload = torch.load(EVIDENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    model.evidence_head.load_state_dict(evidence_payload["state_dict"], strict=True)
    model.qd_model.eval()
    model.evidence_head.eval()

    no_evidence = EvidenceRanker().to(opt.device)
    no_evidence_payload = torch.load(NO_EVIDENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    no_evidence.load_state_dict(no_evidence_payload["state_dict"], strict=True)
    no_evidence.eval()

    full = EvidenceRanker().to(opt.device)
    full_payload = torch.load(FULL_RANKER_CHECKPOINT, map_location="cpu", weights_only=False)
    full.load_state_dict(full_payload["state_dict"], strict=True)
    full.eval()

    for module in [model.evidence_head, no_evidence, full]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model, no_evidence, full


def qd_only_records(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    output = []
    for record in records:
        current = dict(record)
        current["e3_full_egcg"] = [dict(candidate) for candidate in record["e0_qd"]]
        output.append(current)
    return output


def no_evidence_examples(examples: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    output = []
    for example in examples:
        current = dict(example)
        current["audio"] = example["audio"].clone()
        current["query"] = example["query"].clone()
        current["scalar"] = example["scalar"].clone()
        current["scalar"][:, 1:] = 0.0
        output.append(current)
    return output


def full_examples(examples: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [
        {
            **example,
            "audio": example["audio"].clone(),
            "query": example["query"].clone(),
            "scalar": example["scalar"].clone(),
        }
        for example in examples
    ]


def ranked_records(
    records: Sequence[Mapping[str, Any]],
    ranker: EvidenceRanker,
    cache: FeatureCache,
    opt: Any,
    pool: str,
    full_inputs: bool,
) -> list[Dict[str, Any]]:
    prepared = qd_only_records(records) if pool == "qd_only" else [dict(record) for record in records]
    examples = examples_from_records(prepared, cache)
    examples = full_examples(examples) if full_inputs else no_evidence_examples(examples)
    ranked = rank_examples(ranker, examples, opt.device)
    return [{**record, "e2_ranked": candidates} for record, candidates in zip(prepared, ranked)]


def evaluate(records: Sequence[Mapping[str, Any]], key: str, score_key: str) -> Dict[str, Any]:
    candidate = recovered_r17.candidate_metrics(records, key)
    groups = {
        name: [
            record
            for record in records
            if name == "all"
            or recovered_r14.duration_bin(recovered_r14.gt_duration(record["meta"])) == name
        ]
        for name in ["all", *recovered_r14.BINS.keys()]
    }
    official = {name: recovered_r17.official(rows, key, score_key) for name, rows in groups.items()}
    return {
        "N": len(records),
        "official": {
            "overall": official["all"],
            "by_gt_duration_bin": {
                name: {"N": len(groups[name]), "metrics": official[name]}
                for name in recovered_r14.BINS
            },
        },
        "candidate": candidate,
        "definition": f"test records key={key}, score={score_key}",
    }


def candidate_aliases(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "CandidateRecall@10@0.5": candidate["CandidateRecall@10@0.5"],
        "CandidateRecall@10@0.7": candidate["CandidateRecall@10@0.7"],
        "CandidateRecall@100@0.5": candidate["Oracle@100@0.5"],
        "CandidateRecall@100@0.7": candidate["Oracle@100@0.7"],
        "Oracle@10@0.5": candidate["Oracle@10@0.5"],
        "Oracle@10@0.7": candidate["Oracle@10@0.7"],
        "Top1-Oracle@10@0.7": candidate["Top1_vs_Oracle10_gap@0.7"],
    }


def save_predictions(
    base_records: Sequence[Mapping[str, Any]],
    systems: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    with (OUT / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(base_records):
            payload: Dict[str, Any] = {
                "qid": record["meta"]["qid"],
                "vid": record["meta"]["vid"],
                "gt_windows": record["meta"]["relevant_windows"],
            }
            for name, rows in systems.items():
                key = "e0_qd" if name == "A_QD_DETR" else "e2_ranked"
                score_key = "qd_score" if name == "A_QD_DETR" else "ranker_score"
                payload[name] = [
                    [float(c["start"]), float(c["end"]), float(c[score_key])]
                    for c in rows[index][key][:100]
                ]
            handle.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")
    (OUT / "predictions.jsonl").write_bytes((OUT / "test_predictions.jsonl").read_bytes())
    (OUT / "test_predictions.jsonl").unlink()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_qd(model: Any, dataset: Any, opt: Any) -> float:
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=recovered_r14.start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=0,
        shuffle=False,
    )
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            inputs, _targets = recovered_r14.prepare_batch_inputs(batch[1], opt.device)
            model.qd_model(**inputs)
    sync()
    return time.perf_counter() - started


def timed_egcg(model: Any, ranker: EvidenceRanker, dataset: Any, opt: Any, cache: FeatureCache) -> float:
    started = time.perf_counter()
    with torch.no_grad():
        records = recovered_r15.enrich(recovered_r14.collect_records(model, dataset, opt))
        examples = full_examples(examples_from_records(records, cache))
        rank_examples(ranker, examples, opt.device)
    sync()
    return time.perf_counter() - started


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def write_efficiency(
    metrics: Mapping[str, Any],
    latency: Mapping[str, Mapping[str, Any]],
    complexity: Mapping[str, Any],
) -> None:
    rows = []
    for system in ["A_QD_DETR", "D_full_EGCG"]:
        row = latency[system]
        rows.append({
            "system": system,
            "queries": metrics["test_size"],
            "batch_size": metrics["batch_size"],
            "runs": row["runs"],
            "mean_latency_sec": row["mean_latency_sec"],
            "std_latency_sec": row["std_latency_sec"],
            "mean_latency_ms_per_query": row["mean_latency_ms_per_query"],
            "peak_gpu_memory_bytes": row["peak_gpu_memory_bytes"],
            "gpu": metrics["gpu"],
        })
    with (OUT / "runtime_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    complexity_rows = [
        {"component": "QD-DETR", "parameters": complexity["qd_detr_parameters"], "additional_parameters": 0, "additional_percentage": 0.0},
        {"component": "Evidence Head", "parameters": complexity["evidence_head_parameters"], "additional_parameters": complexity["evidence_head_parameters"], "additional_percentage": 100.0 * complexity["evidence_head_parameters"] / complexity["qd_detr_parameters"]},
        {"component": "Ranker", "parameters": complexity["ranker_parameters"], "additional_parameters": complexity["ranker_parameters"], "additional_percentage": 100.0 * complexity["ranker_parameters"] / complexity["qd_detr_parameters"]},
        {"component": "Total EGCG additional", "parameters": complexity["additional_parameters"], "additional_parameters": complexity["additional_parameters"], "additional_percentage": complexity["additional_percentage"]},
    ]
    with (OUT / "complexity_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(complexity_rows[0]))
        writer.writeheader()
        writer.writerows(complexity_rows)

    r27_runtime = json.loads(R27_METRICS.read_text())["runtime"]
    (OUT / "efficiency_report.md").write_text(
        "# R31 efficiency benchmark\n\n"
        f"Hardware: {metrics['gpu']}; PyTorch {metrics['torch']}; CUDA {metrics['cuda']}.\n\n"
        f"Both systems use {metrics['test_size']} test queries and batch size {metrics['batch_size']}. Each latency row uses {EFFICIENCY_RUNS} timed runs after one warm-up run.\n\n"
        "## Inference latency\n\n"
        "| System | Mean sec | Std sec | Mean ms/query | Peak GPU memory |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {system} | {latency[system]['mean_latency_sec']:.4f} | {latency[system]['std_latency_sec']:.4f} | {latency[system]['mean_latency_ms_per_query']:.4f} | {latency[system]['peak_gpu_memory_bytes']} |"
            for system in ["A_QD_DETR", "D_full_EGCG"]
        )
        + "\n\n## Parameters\n\n"
        f"- QD-DETR: {complexity['qd_detr_parameters']}\n"
        f"- Evidence Head: {complexity['evidence_head_parameters']}\n"
        f"- Ranker: {complexity['ranker_parameters']}\n"
        f"- Total EGCG additional: {complexity['additional_parameters']} ({complexity['additional_percentage']:.6f}% over QD-DETR)\n\n"
        "## Existing training cost\n\n"
        f"- Evidence Head training: {r27_runtime['evidence_training_seconds']:.3f} sec\n"
        f"- Frozen full Ranker training: {r27_runtime['ranker_training_seconds']:.3f} sec\n"
        "\nTraining times are reused from the existing R27 checkpoint/log record; R31 performs no training.\n"
    )


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {OUT}")
    for path in [EVIDENCE_CHECKPOINT, FULL_RANKER_CHECKPOINT, NO_EVIDENCE_CHECKPOINT, BASE_CHECKPOINT]:
        if not path.exists():
            raise SystemExit(f"Required checkpoint missing: {path}")
    OUT.mkdir(parents=True)

    opt = recovered_r14.load_options()
    test_dataset = recovered_r14.make_dataset(opt, "test")
    model, no_evidence_ranker, full_ranker = load_models(opt)
    cache = FeatureCache()

    before = {
        "evidence_head": state_digest(model.evidence_head),
        "no_evidence_ranker": state_digest(no_evidence_ranker),
        "full_ranker": state_digest(full_ranker),
        "qd_model": state_digest(model.qd_model),
    }
    with torch.no_grad():
        base_records = recovered_r15.enrich(recovered_r14.collect_records(model, test_dataset, opt))
    a_records = base_records
    b_records = ranked_records(base_records, no_evidence_ranker, cache, opt, "qd_only", False)
    c_records = ranked_records(base_records, no_evidence_ranker, cache, opt, "full", False)
    d_records = ranked_records(base_records, full_ranker, cache, opt, "full", True)
    system_records = {
        "A_QD_DETR": a_records,
        "B_no_evidence_ranker": b_records,
        "C_evidence_candidate_generation": c_records,
        "D_full_EGCG": d_records,
    }
    systems = {
        "A_QD_DETR": evaluate(a_records, "e0_qd", "qd_score"),
        "B_no_evidence_ranker": evaluate(b_records, "e2_ranked", "ranker_score"),
        "C_evidence_candidate_generation": evaluate(c_records, "e2_ranked", "ranker_score"),
        "D_full_EGCG": evaluate(d_records, "e2_ranked", "ranker_score"),
    }
    for system in systems.values():
        system["candidate_aliases"] = {
            bin_name: candidate_aliases(values)
            for bin_name, values in system["candidate"].items()
            if isinstance(values, Mapping) and "N" in values
        }
    save_predictions(base_records, system_records)

    latency: Dict[str, Dict[str, Any]] = {}
    for name, fn in [
        ("A_QD_DETR", lambda: timed_qd(model, test_dataset, opt)),
        ("D_full_EGCG", lambda: timed_egcg(model, full_ranker, test_dataset, opt, cache)),
    ]:
        fn()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        values = [fn() for _ in range(EFFICIENCY_RUNS)]
        sync()
        mean, std = mean_std(values)
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        latency[name] = {
            "runs": EFFICIENCY_RUNS,
            "values_sec": values,
            "mean_latency_sec": mean,
            "std_latency_sec": std,
            "mean_latency_ms_per_query": 1000.0 * mean / len(test_dataset),
            "peak_gpu_memory_bytes": peak,
        }

    after = {
        "evidence_head": state_digest(model.evidence_head),
        "no_evidence_ranker": state_digest(no_evidence_ranker),
        "full_ranker": state_digest(full_ranker),
        "qd_model": state_digest(model.qd_model),
    }
    state_unchanged = before == after
    r28 = json.loads(Path("/private/research-artifact").read_text())
    complexity = r28["complexity"]
    metrics = {
        "status": "R31_FINAL_EVALUATION_COMPLETE",
        "test_size": len(test_dataset),
        "batch_size": opt.eval_bsz,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "systems": systems,
        "complexity": complexity,
        "latency": latency,
        "parameter_state_unchanged": state_unchanged,
        "data_hygiene": {
            "split": "test",
            "test_labels_used_during_training": False,
            "parameter_updates": False,
            "training_performed": False,
        },
        "checkpoint_hashes": {
            "evidence_head": {"path": str(EVIDENCE_CHECKPOINT), "sha256": sha256(EVIDENCE_CHECKPOINT)},
            "full_ranker": {"path": str(FULL_RANKER_CHECKPOINT), "sha256": sha256(FULL_RANKER_CHECKPOINT)},
            "no_evidence_control": {"path": str(NO_EVIDENCE_CHECKPOINT), "sha256": sha256(NO_EVIDENCE_CHECKPOINT)},
            "qd_detr": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)},
        },
        "state_digests_before": before,
        "state_digests_after": after,
        "provenance": {"egcg_commit": git_commit(), "raw_audio": False},
    }
    write_json(OUT / "test_metrics.json", metrics)
    (OUT / "metrics.json").write_bytes((OUT / "test_metrics.json").read_bytes())
    (OUT / "test_metrics.json").unlink()
    write_efficiency(metrics, latency, complexity)

    report_lines = [
        "# R31 final test ablation",
        "",
        "## Protocol",
        "",
        f"- Official test split only: N={len(test_dataset)}.",
        "- No training, parameter update, hyperparameter search, or validation-based selection was performed.",
        "- A: original QD-DETR proposals and QD ranking.",
        "- B: QD-only proposals ranked by the frozen R30 no-evidence control checkpoint.",
        "- C: QD+Evidence proposals ranked by the same B checkpoint with no evidence features.",
        "- D: QD+Evidence proposals ranked by the frozen EGCG v1.0 full ranker checkpoint.",
        "",
        "## Overall test metrics",
        "",
        "| System | R1@0.5 | R1@0.7 | mAP | CandidateRecall@10@0.5 | CandidateRecall@100@0.5 | Oracle@10@0.7 | Top1–Oracle gap@0.7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, system in systems.items():
        overall = system["official"]["overall"]
        aliases = candidate_aliases(system["candidate"]["all"])
        report_lines.append(
            f"| {name} | {overall.get('MR-full-R1@0.5')} | {overall.get('MR-full-R1@0.7')} | {overall.get('MR-full-mAP')} | {100*aliases['CandidateRecall@10@0.5']['rate']:.2f}% | {100*aliases['CandidateRecall@100@0.5']['rate']:.2f}% | {100*aliases['Oracle@10@0.7']['rate']:.2f}% | {100*aliases['Top1-Oracle@10@0.7']:.2f}% |"
        )
    report_lines += [
        "",
        "## Short-duration test metrics",
        "",
        "| System | 0–2s N | 0–2s R1@0.7 | 0–2s CandidateRecall@10@0.5 | 0–2s CandidateRecall@100@0.5 | 2–5s N | 2–5s R1@0.7 | 2–5s CandidateRecall@10@0.5 | 2–5s CandidateRecall@100@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, system in systems.items():
        b0 = system["candidate"]["0-2s"]
        b1 = system["candidate"]["2-5s"]
        report_lines.append(
            f"| {name} | {b0['N']} | {system['official']['by_gt_duration_bin']['0-2s']['metrics'].get('MR-full-R1@0.7')} | {100*b0['CandidateRecall@10@0.5']['rate']:.2f}% | {100*b0['Oracle@100@0.5']['rate']:.2f}% | {b1['N']} | {system['official']['by_gt_duration_bin']['2-5s']['metrics'].get('MR-full-R1@0.7')} | {100*b1['CandidateRecall@10@0.5']['rate']:.2f}% | {100*b1['Oracle@100@0.5']['rate']:.2f}% |"
        )
    report_lines += [
        "",
        "## Data hygiene",
        "",
        f"- Checkpoint parameter digests unchanged: `{state_unchanged}`.",
        "- Test labels were used only by the evaluation metrics after inference; no optimizer or selection step consumed them.",
        "- Checkpoint hashes and EGCG commit are recorded in `provenance.md` and `final_manifest.json`.",
        "",
    ]
    (OUT / "test_ablation_report.md").write_text("\n".join(report_lines) + "\n")

    (OUT / "provenance.md").write_text(
        "# R31 provenance\n\n"
        f"- EGCG v1.0 commit: `{git_commit()}`\n"
        "- Evaluation split: official test split only.\n"
        "- No training, parameter update, validation selection, or hyperparameter search.\n"
        f"- Evidence Head checkpoint: `{EVIDENCE_CHECKPOINT}`\n"
        f"- Full EGCG Ranker checkpoint: `{FULL_RANKER_CHECKPOINT}`\n"
        f"- No-evidence control checkpoint from R30: `{NO_EVIDENCE_CHECKPOINT}`\n"
        f"- QD-DETR checkpoint: `{BASE_CHECKPOINT}`\n"
        "- B uses QD-only candidates and the no-evidence control. C uses the QD+Evidence candidate pool with the same B control. D uses the QD+Evidence pool and frozen full EGCG ranker.\n"
        "- Training cost values are read from the existing R27 runtime record.\n"
        "- Complexity values are reused from R28.\n"
        "- Raw audio was not used.\n"
    )

    manifest = {
        "status": metrics["status"],
        "egcg_commit": git_commit(),
        "split_protocol": metrics["data_hygiene"],
        "checkpoint_hashes": metrics["checkpoint_hashes"],
        "artifacts": [],
    }
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "final_manifest.json":
            manifest["artifacts"].append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(OUT / "final_manifest.json", manifest)
    model.close()
    print(json.dumps({"status": metrics["status"], "output": str(OUT), "parameter_state_unchanged": state_unchanged}))


if __name__ == "__main__":
    main()
