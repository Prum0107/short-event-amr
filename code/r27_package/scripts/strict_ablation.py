#!/usr/bin/env python3
"""Run R27 strict ranking-input ablations only after reproduction passes."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
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
from ranker import FeatureCache, EvidenceRanker, examples_from_records, rank_examples, train_ranker  # noqa: E402


OUT = Path("/private/research-artifact")
CHECKPOINT_DIR = OUT / "checkpoints"
LOG_DIR = OUT / "logs"
METRICS_PATH = OUT / "metrics.json"


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


def clone_for_mode(records: Sequence[Mapping[str, Any]], mode: str) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []
    for record in records:
        current = dict(record)
        candidates = [dict(candidate) for candidate in record["e3_full_egcg"]]
        if mode == "no_evidence":
            for candidate in candidates:
                candidate["evidence_score"] = 0.0
                candidate["combined_score"] = float(candidate["qd_score"])
        elif mode == "evidence_only":
            for candidate in candidates:
                candidate["qd_score"] = 0.0
                candidate["combined_score"] = float(candidate["evidence_score"])
        current["e3_full_egcg"] = candidates
        result.append(current)
    return result


def mask_examples(examples: Sequence[Mapping[str, Any]], mode: str) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []
    for example in examples:
        current = dict(example)
        current["audio"] = example["audio"].clone()
        current["query"] = example["query"].clone()
        current["scalar"] = example["scalar"].clone()
        if mode == "no_evidence":
            current["scalar"][:, 1] = 0.0
        elif mode == "evidence_only":
            current["audio"][:] = 0.0
            current["query"][:] = 0.0
            current["scalar"][:, 0] = 0.0
            current["scalar"][:, 2:] = 0.0
        result.append(current)
    return result


def attach(records: Sequence[Mapping[str, Any]], ranked: Sequence[Sequence[Mapping[str, Any]]], key: str) -> list[Dict[str, Any]]:
    result = []
    for record, candidates in zip(records, ranked):
        current = dict(record)
        current[key] = list(candidates)
        result.append(current)
    return result


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    if metrics.get("status") != "R27_REPRODUCTION_PASS":
        (OUT / "ablation_report.md").write_text("# R27 strict ablation\n\nNot run because the reproduction gate did not pass.\n")
        print(json.dumps({"status": "R27_ABLATION_NOT_RUN", "reason": metrics.get("status")}))
        return

    opt = recovered_r14.load_options()
    evidence_model, _criterion = recovered_r15.build_model(opt)
    evidence_checkpoint = torch.load(CHECKPOINT_DIR / "evidence_head_seed2023.pt", map_location=opt.device, weights_only=False)
    evidence_model.evidence_head.load_state_dict(evidence_checkpoint["state_dict"], strict=True)
    train_dataset = recovered_r14.make_dataset(opt, "train")
    train_records = recovered_r15.enrich(recovered_r14.collect_records(evidence_model, train_dataset, opt))
    cache = FeatureCache()

    control_histories: Dict[str, Any] = {}
    control_checkpoints: Dict[str, str] = {}
    rankers: Dict[str, EvidenceRanker] = {}
    for mode in ["no_evidence", "evidence_only"]:
        set_seed(2023)
        mode_train_records = clone_for_mode(train_records, mode)
        examples = mask_examples(examples_from_records(mode_train_records, cache), mode)
        ranker = EvidenceRanker().to(opt.device)
        history, _optimizer, _elapsed = train_ranker(
            ranker, examples, opt.device, 5, 8, 1e-3, 1e-4, 0.1, 0.1,
            CHECKPOINT_DIR / f"{mode}_ranker_seed2023.pt",
            CHECKPOINT_DIR / f"{mode}_ranker_optimizer_seed2023.pt",
            LOG_DIR / f"{mode}_ranker.jsonl",
        )
        control_histories[mode] = history
        control_checkpoints[mode] = str(CHECKPOINT_DIR / f"{mode}_ranker_seed2023.pt")
        rankers[mode] = ranker

    full_ranker = EvidenceRanker().to(opt.device)
    full_state = torch.load(CHECKPOINT_DIR / "ranker_seed2023.pt", map_location=opt.device, weights_only=False)
    full_ranker.load_state_dict(full_state["state_dict"], strict=True)

    # Test data is accessed only after all control rankers have finished training.
    test_dataset = recovered_r14.make_dataset(opt, "test")
    test_records = recovered_r15.enrich(recovered_r14.collect_records(evidence_model, test_dataset, opt))
    test_examples = examples_from_records(test_records, cache)
    full_ranked = rank_examples(full_ranker, test_examples, opt.device)

    ranked_records: Dict[str, list[Dict[str, Any]]] = {"D_full_EGCG": attach(test_records, full_ranked, "ranked")}
    for mode, ranker in rankers.items():
        mode_records = clone_for_mode(test_records, mode)
        mode_examples = mask_examples(examples_from_records(mode_records, cache), mode)
        ranked = rank_examples(ranker, mode_examples, opt.device)
        ranked_records["B_no_evidence_ranker" if mode == "no_evidence" else "C_evidence_only_ranker"] = attach(mode_records, ranked, "ranked")

    systems = {
        "A_QD_DETR": recovered_r17.system_metrics(test_records, "original QD-DETR", "e0_qd", "qd_score"),
        "B_no_evidence_ranker": recovered_r17.system_metrics(ranked_records["B_no_evidence_ranker"], "same fused pool with evidence input masked", "ranked", "ranker_score"),
        "C_evidence_only_ranker": recovered_r17.system_metrics(ranked_records["C_evidence_only_ranker"], "same fused pool with only evidence input", "ranked", "ranker_score"),
        "D_full_EGCG": recovered_r17.system_metrics(ranked_records["D_full_EGCG"], "full recovered EGCG ranker", "ranked", "ranker_score"),
    }
    metrics["strict_ablation"] = {
        "status": "R27_STRICT_ABLATION_COMPLETE",
        "training_split_only": True,
        "test_access_after_all_training": True,
        "input_definitions": {
            "B_no_evidence_ranker": "full EGCG candidate pool; evidence scalar zeroed; candidate audio/query and QD score retained",
            "C_evidence_only_ranker": "full EGCG candidate pool; candidate audio/query, QD score, position and duration scalars zeroed; evidence scalar retained",
            "D_full_EGCG": "full recovered R17 feature set and candidate pool",
        },
        "systems": systems,
        "training_histories": control_histories,
        "control_checkpoints": control_checkpoints,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# R27 strict ablation", "",
        "All learned controls were trained on the official train split only. Test data was accessed after both control rankers and the recovered full ranker were available.", "",
        "| System | R1@0.5 | R1@0.7 | mAP | 0–2s R1@0.7 | 2–5s R1@0.7 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, system in systems.items():
        overall = system["official"]["overall"]
        bins = system["official"]["by_gt_duration_bin"]
        lines.append(f"| {name} | {overall.get('MR-full-R1@0.5')} | {overall.get('MR-full-R1@0.7')} | {overall.get('MR-full-mAP')} | {bins['0-2s']['metrics'].get('MR-full-R1@0.7')} | {bins['2-5s']['metrics'].get('MR-full-R1@0.7')} |")
    lines += ["", "## Control definitions", "", "- B removes the evidence scalar while retaining the same full candidate pool and candidate content features.", "- C removes candidate audio/query and non-evidence scalar inputs while retaining the evidence scalar and the same candidate pool.", "- D is the recovered R17 ranker.", ""]
    (OUT / "ablation_report.md").write_text("\n".join(lines) + "\n")

    manifest_path = OUT / "checkpoints_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in sorted(CHECKPOINT_DIR.glob("*.pt")):
        item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        if not any(entry.get("path") == str(path) for entry in manifest["artifacts"]):
            manifest["artifacts"].append(item)
    manifest["strict_ablation"] = {"status": "R27_STRICT_ABLATION_COMPLETE", "control_checkpoints": control_checkpoints}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "R27_STRICT_ABLATION_COMPLETE", "output": str(OUT)}))


if __name__ == "__main__":
    main()
