#!/usr/bin/env python3
"""Run the clean R30 EGCG ablation on train/validation splits only."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
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
from ranker import (  # noqa: E402
    EvidenceRanker,
    FeatureCache,
    examples_from_records,
    rank_examples,
    train_ranker,
)


OUT = Path("/private/research-artifact")
FROZEN_EVIDENCE = ROOT / "checkpoints/egcg_v1.0_seed2023/evidence_head_seed2023.pt"
R28_METRICS = Path("/private/research-artifact")
BASE_CHECKPOINT = Path("/private/research-artifact")
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


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_model(opt: Any) -> Any:
    model, _criterion = recovered_r15.build_model(opt)
    checkpoint = torch.load(FROZEN_EVIDENCE, map_location="cpu", weights_only=False)
    model.evidence_head.load_state_dict(checkpoint["state_dict"], strict=True)
    model.evidence_head.eval()
    model.qd_model.eval()
    return model


def qd_only_records(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Keep only original QD candidates while preserving the record schema."""
    output: list[Dict[str, Any]] = []
    for record in records:
        current = dict(record)
        current["e3_full_egcg"] = [dict(candidate) for candidate in record["e0_qd"]]
        output.append(current)
    return output


def no_evidence_examples(examples: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Keep audio/query/QD and remove direct evidence and geometry inputs."""
    output: list[Dict[str, Any]] = []
    for example in examples:
        current = dict(example)
        current["audio"] = example["audio"].clone()
        current["query"] = example["query"].clone()
        current["scalar"] = example["scalar"].clone()
        current["scalar"][:, 1:] = 0.0
        output.append(current)
    return output


def full_examples(examples: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Keep all recovered full-EGCG ranker inputs."""
    return [
        {
            **example,
            "audio": example["audio"].clone(),
            "query": example["query"].clone(),
            "scalar": example["scalar"].clone(),
        }
        for example in examples
    ]


def train_no_evidence_ranker(
    train_records: Sequence[Mapping[str, Any]],
    cache: FeatureCache,
    opt: Any,
    work: Path,
) -> tuple[EvidenceRanker, float, int]:
    set_seed(SEED)
    examples = no_evidence_examples(examples_from_records(qd_only_records(train_records), cache))
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
        work / "B_no_evidence_ranker.pt",
        work / "B_no_evidence_ranker_optimizer.pt",
        work / "B_no_evidence_ranker.jsonl",
    )
    return model, time.perf_counter() - started, len(examples)


def train_full_ranker(
    train_records: Sequence[Mapping[str, Any]],
    cache: FeatureCache,
    opt: Any,
    work: Path,
) -> tuple[EvidenceRanker, float, int]:
    set_seed(SEED)
    examples = full_examples(examples_from_records(train_records, cache))
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
        work / "D_full_egcg_ranker.pt",
        work / "D_full_egcg_ranker_optimizer.pt",
        work / "D_full_egcg_ranker.jsonl",
    )
    return model, time.perf_counter() - started, len(examples)


def rank_no_evidence(
    model: EvidenceRanker,
    records: Sequence[Mapping[str, Any]],
    cache: FeatureCache,
    opt: Any,
) -> list[Dict[str, Any]]:
    prepared = qd_only_records(records)
    examples = no_evidence_examples(examples_from_records(prepared, cache))
    ranked = rank_examples(model, examples, opt.device)
    return [{**record, "e2_ranked": candidates} for record, candidates in zip(prepared, ranked)]


def rank_no_evidence_on_full_pool(
    model: EvidenceRanker,
    records: Sequence[Mapping[str, Any]],
    cache: FeatureCache,
    opt: Any,
) -> list[Dict[str, Any]]:
    examples = no_evidence_examples(examples_from_records(records, cache))
    ranked = rank_examples(model, examples, opt.device)
    return [{**record, "e2_ranked": candidates} for record, candidates in zip(records, ranked)]


def rank_full(
    model: EvidenceRanker,
    records: Sequence[Mapping[str, Any]],
    cache: FeatureCache,
    opt: Any,
) -> list[Dict[str, Any]]:
    examples = full_examples(examples_from_records(records, cache))
    ranked = rank_examples(model, examples, opt.device)
    return [{**record, "e2_ranked": candidates} for record, candidates in zip(records, ranked)]


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
    official = {name: recovered_r17.official(subset, key, score_key) for name, subset in subsets.items()}
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


def pct(value: Any) -> float | None:
    return None if value is None else 100.0 * float(value)


def metric(system: Mapping[str, Any], bin_name: str, key: str) -> float | None:
    if bin_name == "all":
        value = system["official"]["overall"].get(key)
    else:
        value = system["official"]["by_gt_duration_bin"][bin_name]["metrics"].get(key)
    return None if value is None else float(value)


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.2f}"


def save_predictions(
    validation_records: Sequence[Mapping[str, Any]],
    systems: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(validation_records):
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


def write_reports(metrics: Mapping[str, Any]) -> None:
    names = ["A_QD_DETR", "B_no_evidence_ranker", "C_evidence_candidate_generation", "D_full_EGCG"]
    lines = [
        "# R30 final EGCG ablation",
        "",
        "## Protocol",
        "",
        "- A uses original QD-DETR proposals and original QD ranking.",
        "- B trains one no-evidence ranker on QD-only train candidates and evaluates it on QD-only validation candidates.",
        "- C applies the same trained B ranker to the full QD-plus-Evidence validation pool; it has no evidence-aware ranker inputs.",
        "- D trains the full evidence-aware ranker on the full QD-plus-Evidence train pool and evaluates the full validation pool.",
        "- B-to-C isolates candidate-generation contribution under the same no-evidence ranker. C-to-D isolates the added evidence-aware ranker, subject to the distinct training-pool definition stated above.",
        "- All ranker training uses train split only, seed 2023, five epochs, AdamW, learning rate 1e-3, weight decay 1e-4, batch size 8, margin 0.1, and gradient clipping 0.1. No test split is constructed or read.",
        "",
        "## Overall validation metrics",
        "",
        "| System | R1@0.5 | R1@0.7 | mAP | CandidateRecall@10@0.5 | Oracle@10@0.7 | Top1–Oracle gap@0.7 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        system = metrics["systems"][name]
        c = system["candidate"]["all"]
        lines.append(
            f"| {name} | {fmt(metric(system, 'all', 'MR-full-R1@0.5'))} | {fmt(metric(system, 'all', 'MR-full-R1@0.7'))} | {fmt(metric(system, 'all', 'MR-full-mAP'))} | "
            f"{fmt(pct(c['CandidateRecall@10@0.5']['rate']))}% | {fmt(pct(c['Oracle@10@0.7']['rate']))}% | {fmt(pct(c['Top1_vs_Oracle10_gap@0.7']))}% |"
        )
    lines += [
        "",
        "## Short-duration validation",
        "",
        "| System | 0–2s N | 0–2s R1@0.7 | 0–2s CandidateRecall@10@0.5 | 0–2s Oracle@10@0.7 | 0–2s gap | 2–5s N | 2–5s R1@0.7 | 2–5s CandidateRecall@10@0.5 | 2–5s Oracle@10@0.7 | 2–5s gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        system = metrics["systems"][name]
        b0 = system["candidate"]["0-2s"]
        b1 = system["candidate"]["2-5s"]
        lines.append(
            f"| {name} | {b0['N']} | {fmt(metric(system, '0-2s', 'MR-full-R1@0.7'))} | {fmt(pct(b0['CandidateRecall@10@0.5']['rate']))}% | {fmt(pct(b0['Oracle@10@0.7']['rate']))}% | {fmt(pct(b0['Top1_vs_Oracle10_gap@0.7']))}% | "
            f"{b1['N']} | {fmt(metric(system, '2-5s', 'MR-full-R1@0.7'))} | {fmt(pct(b1['CandidateRecall@10@0.5']['rate']))}% | {fmt(pct(b1['Oracle@10@0.7']['rate']))}% | {fmt(pct(b1['Top1_vs_Oracle10_gap@0.7']))}% |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "C gains candidate availability relative to B without changing the ranker weights or ranker input definition. D then tests whether the full evidence-aware ranker can exploit that expanded pool. These results are validation-only and should not be compared numerically with historical R18 test metrics.",
        "",
    ]
    (OUT / "final_ablation_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing R30 output directory: {OUT}")
    if not FROZEN_EVIDENCE.exists():
        raise SystemExit(f"Missing frozen Evidence Head checkpoint: {FROZEN_EVIDENCE}")
    OUT.mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix="egcg_r30_", dir="/tmp"))

    set_seed(SEED)
    opt = recovered_r14.load_options()
    train_dataset = recovered_r14.make_dataset(opt, "train")
    validation_dataset = recovered_r14.make_dataset(opt, "val")
    model = load_model(opt)
    cache = FeatureCache()
    train_records = recovered_r15.enrich(recovered_r14.collect_records(model, train_dataset, opt))
    validation_records = recovered_r15.enrich(recovered_r14.collect_records(model, validation_dataset, opt))

    b_ranker, b_seconds, b_examples = train_no_evidence_ranker(train_records, cache, opt, work)
    d_ranker, d_seconds, d_examples = train_full_ranker(train_records, cache, opt, work)

    a_records = validation_records
    b_records = rank_no_evidence(b_ranker, validation_records, cache, opt)
    c_records = rank_no_evidence_on_full_pool(b_ranker, validation_records, cache, opt)
    d_records = rank_full(d_ranker, validation_records, cache, opt)
    record_systems = {
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
    save_predictions(validation_records, record_systems)

    r28 = json.loads(R28_METRICS.read_text())
    metrics = {
        "status": "R30_FINAL_ABLATION_COMPLETE",
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
            "B_no_evidence_ranker_train_examples": b_examples,
            "D_full_EGCG_ranker_train_examples": d_examples,
        },
        "systems": systems,
        "candidate_sources": {
            "A_QD_DETR": "record[e0_qd]",
            "B_no_evidence_ranker": "QD-only record[e0_qd]",
            "C_evidence_candidate_generation": "full record[e3_full_egcg] = QD + Evidence",
            "D_full_EGCG": "full record[e3_full_egcg] = QD + Evidence",
        },
        "ranking_definitions": {
            "A_QD_DETR": "original qd_score order",
            "B_no_evidence_ranker": "trainable EvidenceRanker with candidate audio/query + QD score; evidence and geometry zeroed",
            "C_evidence_candidate_generation": "same trained B ranker applied to full QD + Evidence pool with no-evidence inputs",
            "D_full_EGCG": "trainable EvidenceRanker with full recovered audio/query/QD/evidence/geometry inputs",
        },
        "training": {
            "B_seconds": b_seconds,
            "D_seconds": d_seconds,
            "optimizer": "AdamW",
            "learning_rate": RANKER_LR,
            "weight_decay": RANKER_WD,
            "epochs": RANKER_EPOCHS,
            "batch_size": RANKER_BATCH_SIZE,
            "margin": RANKER_MARGIN,
            "grad_clip": RANKER_CLIP,
            "B_reused_for_C": True,
        },
        "complexity_reused_from_R28": r28["complexity"],
        "runtime_reused_from_R28": r28["runtime"],
        "provenance": {
            "egcg_commit": git_commit(),
            "base_checkpoint": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)},
            "frozen_evidence_checkpoint": {"path": str(FROZEN_EVIDENCE), "sha256": sha256(FROZEN_EVIDENCE)},
            "raw_audio": False,
            "temporary_training_artifacts": str(work),
        },
    }
    write_json(OUT / "metrics.json", metrics)
    write_reports(metrics)
    (OUT / "provenance.md").write_text(
        "# R30 provenance\n\n"
        f"- EGCG v1.0 commit: `{git_commit()}`\n"
        "- QD-DETR checkpoint was frozen; Evidence Head checkpoint was frozen from the train-only R27 artifact.\n"
        "- B ranker was trained on QD-only train candidates. C reuses the same B ranker weights on the expanded QD+Evidence validation pool.\n"
        "- D ranker was trained on the expanded QD+Evidence train pool with the full recovered input vector.\n"
        "- Seed: 2023; optimizer: AdamW; learning rate: 1e-3; weight decay: 1e-4; epochs: 5; batch size: 8; margin: 0.1; gradient clip: 0.1.\n"
        "- Only train and validation splits were constructed. The test split was not constructed or read.\n"
        "- Complexity and runtime values are reused from R28 as requested; no new cost measurement was used for the R30 comparison.\n"
    )
    model.close()
    print(json.dumps({"status": metrics["status"], "output": str(OUT), "egcg_commit": git_commit()}))


if __name__ == "__main__":
    main()
