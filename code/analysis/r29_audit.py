#!/usr/bin/env python3
"""Audit R28 ablation semantics without loading data or running inference."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


R28_SCRIPT = Path("/private/research-artifact")
RANKER_SOURCE = Path("/private/research-artifact")
PIPELINE_SOURCE = Path("/private/research-artifact")
R18_SOURCE = Path("/private/research-artifact")
R28_METRICS = Path("/private/research-artifact")
OUT = Path("/private/research-artifact")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {OUT}")
    sources = {
        "r28": R28_SCRIPT.read_text(encoding="utf-8"),
        "ranker": RANKER_SOURCE.read_text(encoding="utf-8"),
        "pipeline": PIPELINE_SOURCE.read_text(encoding="utf-8"),
        "r18": R18_SOURCE.read_text(encoding="utf-8"),
    }
    metrics = json.loads(R28_METRICS.read_text(encoding="utf-8"))
    commit = metrics.get("provenance", {}).get("egcg_commit")

    required_r28_snippets = [
        'record["e3_full_egcg"]',
        'current["scalar"][:, 1:] = 0.0',
        'current["audio"].zero_()',
        'current["query"].zero_()',
        'current["scalar"][:, 1] = example["scalar"][:, 1]',
        'current["e2_ranked"] = candidates',
    ]
    missing = [snippet for snippet in required_r28_snippets if snippet not in sources["r28"]]
    if missing:
        raise SystemExit(f"R28 implementation snippets missing: {missing}")

    test_call = re.search(r"make_dataset\(opt,\s*[\"']test[\"']\)", sources["r28"])
    if test_call:
        raise SystemExit("R28 script contains a test-split construction call")

    rows = [
        {
            "system": "B_no_evidence_ranker",
            "input_name": "candidate temporal audio feature",
            "implementation": "FeatureCache.audio_by_vid -> temporal_feature(audio, start, end)",
            "dimension": "768",
            "classification": "candidate feature",
            "state": "retained",
            "notes": "Mean M2D audio feature over the candidate interval; passed through audio_proj.",
        },
        {
            "system": "B_no_evidence_ranker",
            "input_name": "query feature",
            "implementation": "FeatureCache.text_by_qid -> mean M2D text embedding",
            "dimension": "768",
            "classification": "query feature",
            "state": "retained",
            "notes": "Passed through query_proj and fused with candidate audio representation.",
        },
        {
            "system": "B_no_evidence_ranker",
            "input_name": "QD score",
            "implementation": "scalar[:, 0] = candidate.qd_score",
            "dimension": "1 per candidate",
            "classification": "QD score",
            "state": "retained",
            "notes": "QD score is retained for no-evidence ranking.",
        },
        {
            "system": "B_no_evidence_ranker",
            "input_name": "evidence score",
            "implementation": "scalar[:, 1] = candidate.evidence_score, then scalar[:, 1:] = 0",
            "dimension": "1 per candidate",
            "classification": "evidence score / evidence-derived feature",
            "state": "zeroed",
            "notes": "Direct evidence scalar is removed from the ranker input.",
        },
        {
            "system": "B_no_evidence_ranker",
            "input_name": "normalized start, end, duration",
            "implementation": "scalar[:, 2:5]",
            "dimension": "3 per candidate",
            "classification": "geometric feature",
            "state": "zeroed",
            "notes": "Position and duration scalars are also zeroed by the R28 no-evidence mask.",
        },
        {
            "system": "C_evidence_only_ranker",
            "input_name": "candidate temporal audio feature",
            "implementation": "FeatureCache.audio_by_vid -> temporal_feature(audio, start, end)",
            "dimension": "768",
            "classification": "candidate feature",
            "state": "zeroed",
            "notes": "Audio tensor is zeroed before ranking.",
        },
        {
            "system": "C_evidence_only_ranker",
            "input_name": "query feature",
            "implementation": "FeatureCache.text_by_qid -> mean M2D text embedding",
            "dimension": "768",
            "classification": "query feature",
            "state": "zeroed",
            "notes": "Query tensor is zeroed before ranking.",
        },
        {
            "system": "C_evidence_only_ranker",
            "input_name": "QD score",
            "implementation": "scalar[:, 0] = candidate.qd_score",
            "dimension": "1 per candidate",
            "classification": "QD score",
            "state": "zeroed",
            "notes": "QD score is removed from the evidence-only ranker input.",
        },
        {
            "system": "C_evidence_only_ranker",
            "input_name": "evidence score",
            "implementation": "scalar[:, 1] = candidate.evidence_score",
            "dimension": "1 per candidate",
            "classification": "evidence score / evidence-derived feature",
            "state": "retained",
            "notes": "This is the only nonzero ranker input after masking.",
        },
        {
            "system": "C_evidence_only_ranker",
            "input_name": "normalized start, end, duration",
            "implementation": "scalar[:, 2:5]",
            "dimension": "3 per candidate",
            "classification": "geometric feature",
            "state": "zeroed",
            "notes": "All geometric scalars are zeroed.",
        },
        {
            "system": "D_full_EGCG",
            "input_name": "candidate temporal audio feature",
            "implementation": "FeatureCache.audio_by_vid -> temporal_feature(audio, start, end)",
            "dimension": "768",
            "classification": "candidate feature",
            "state": "retained",
            "notes": "Passed through audio_proj.",
        },
        {
            "system": "D_full_EGCG",
            "input_name": "query feature",
            "implementation": "FeatureCache.text_by_qid -> mean M2D text embedding",
            "dimension": "768",
            "classification": "query feature",
            "state": "retained",
            "notes": "Passed through query_proj.",
        },
        {
            "system": "D_full_EGCG",
            "input_name": "QD score",
            "implementation": "scalar[:, 0] = candidate.qd_score",
            "dimension": "1 per candidate",
            "classification": "QD score",
            "state": "retained",
            "notes": "Retained in the five-dimensional scalar vector.",
        },
        {
            "system": "D_full_EGCG",
            "input_name": "evidence score",
            "implementation": "scalar[:, 1] = candidate.evidence_score",
            "dimension": "1 per candidate",
            "classification": "evidence score / evidence-derived feature",
            "state": "retained",
            "notes": "Direct evidence score is retained.",
        },
        {
            "system": "D_full_EGCG",
            "input_name": "normalized start, end, duration",
            "implementation": "scalar[:, 2:5]",
            "dimension": "3 per candidate",
            "classification": "geometric feature",
            "state": "retained",
            "notes": "Candidate geometry is retained.",
        },
    ]

    OUT.mkdir(parents=True)
    with (OUT / "feature_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    r28_pool_statement = (
        "R28 calls `replace_candidate_features(records, mode)` for all three rankers. "
        "That function iterates over `record[\"e3_full_egcg\"]` and changes only feature values; "
        "it does not remove, add, or reorder candidates. `rank_control` applies the same construction "
        "to validation records and ranks the resulting list. Therefore B, C, and D use the same full "
        "EGCG candidate pool, including QD proposals and evidence-derived proposals."
    )
    (OUT / "candidate_pool_audit.md").write_text(
        "# R29 candidate-pool audit\n\n"
        + r28_pool_statement
        + "\n\n"
        "This means the R28 no-evidence result is not a full pipeline without evidence. It is a ranker-input "
        "ablation on a candidate pool whose proposal membership was already generated by the learned Evidence "
        "Head and R15 evidence proposal path. It isolates the contribution of the direct evidence scalar (and "
        "R28 also removes geometry scalars), but it does not isolate candidate generation.\n\n"
        "The full EGCG ranker uses the same intervals and candidate count as B. For evidence-only C, the pool is "
        "also identical; only the ranker inputs are masked.\n\n"
        "No train, validation, or test data was loaded by this audit. No model forward or training operation was run.\n"
    )

    report = [
        "# R29 ablation semantics audit",
        "",
        "## Scope",
        "",
        "This is a static audit of the committed R28 script and recovered R17 ranker definition. It does not run a model, train a component, or construct any dataset split.",
        "",
        f"Audited EGCG commit: `{commit}`.",
        "",
        "## Exact ranker inputs",
        "",
        "The recovered `EvidenceRanker` consumes candidate audio features (768), one query feature (768), and a five-value scalar vector: `[QD_score, evidence_score, normalized_start, normalized_end, normalized_duration]`. It projects audio/query features and concatenates `[audio_h, query_h, audio_h * query_h, scalar]`.",
        "",
        "- **No-evidence ranker (B):** retains candidate audio, query feature, and QD score. It zeros the evidence score and all three geometric scalars.",
        "- **Evidence-only ranker (C):** retains only the evidence score. It zeros candidate audio, query feature, QD score, and all geometric scalars.",
        "- **Full EGCG ranker (D):** retains candidate audio, query feature, QD score, evidence score, and normalized geometry.",
        "",
        "The direct evidence score is the only evidence-derived ranker scalar. The candidate pool itself also contains evidence-derived intervals; that separate fact is covered below.",
        "",
        "## Candidate-pool result",
        "",
        "**B and D use the same candidate pool.** R28 constructs both from `record[\"e3_full_egcg\"]`; the ablation changes ranker input values but does not change candidate membership, intervals, or ordering before the ranker. Consequently the R28 comparison is a ranking-input ablation over a full EGCG pool, not a pure no-evidence end-to-end pipeline.",
        "",
        "The same statement applies to C. It is an evidence-only scoring ablation over the full EGCG pool.",
        "",
        "## Comparison with R18 historical definitions",
        "",
        "| Historical definition | Candidate pool | Ranker input | R28 correspondence |",
        "|---|---|---|---|",
        "| R18 E3 ranker-only | QD-DETR candidates only (`source == \"qd\"`) | Full recovered ranker input | **Not equivalent** to R28 B/C/D; R28 uses the full union pool |",
        "| R18 E4 full EGCG | QD-DETR plus evidence proposals | Full recovered ranker input | **Closest match is R28 D**, but R28 is validation-only and retrains the ranker under the R28 control protocol |",
        "",
        "R28 B is therefore not historical E3. Its near-parity with D cannot be interpreted as evidence that evidence-generated candidates are unnecessary. Evidence-generated candidates remain available to B through the shared pool; B only removes the direct evidence scalar and geometry scalars from the ranker input.",
        "",
        "## Audit conclusion",
        "",
        "`R28_NO_EVIDENCE_IS_NOT_A_FULL_NO_EVIDENCE_PIPELINE`. The ablation is internally consistent as a ranker-input mask, and B/D do share candidate intervals. It does not support a claim that the full evidence pathway is redundant at candidate generation level. A true no-evidence end-to-end control would need a QD-only candidate pool, but that is outside this audit and was not run.",
        "",
        "The R28 script contains no `make_dataset(opt, \"test\")` call; no test split was used for this audit.",
        "",
    ]
    (OUT / "ablation_semantics_report.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
