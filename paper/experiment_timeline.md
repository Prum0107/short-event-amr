# AMR experiment timeline (R0–R31)

All entries are archival summaries of existing artifacts. `EXACTLY_OBSERVED` means the statement is directly present in a source artifact; `DOCUMENTED` means the source report records the conclusion; `INFERRED` is used only where the relationship between experiments is reconstructed from the artifact sequence; `UNKNOWN` facts are not filled in.

## R0 — DOCUMENTED

- Classification: `DOCUMENTED`
- Purpose: Official QD-DETR repository/environment and baseline entry point.
- Research question: Can the official baseline be installed and run reproducibly?
- Pre-experiment hypothesis: Clean official environment and frozen baseline are prerequisites.
- Input/model/data: Repository, requirements, baseline outputs.
- Method/result: Official baseline workflow established; exact artifact status is recorded in source files.
- Effect on final method: Established the fixed baseline reference.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R1 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Reproduce M2D/QD-DETR baseline.
- Research question: Does the reproduced QD-DETR run align with the reference?
- Pre-experiment hypothesis: The fixed checkpoint should reproduce the published/reference metrics.
- Input/model/data: M2D features, QD-DETR checkpoint, CASTELLA test split.
- Method/result: M2D-QD reproduction pass is documented; metrics are in the manifest.
- Effect on final method: Provided the representation-control baseline.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R2 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Representation-only M2D feature control.
- Research question: Does a stronger M2D representation improve AMR under the same QD-DETR framework?
- Pre-experiment hypothesis: Representation quality may account for part of the failure.
- Input/model/data: M2D audio/text features plus frozen QD-DETR.
- Method/result: Representation improves overall performance but short-moment degradation remains.
- Effect on final method: Motivated a temporal evidence diagnosis rather than an encoder replacement.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R3 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Short-scale and geometry audit.
- Research question: Are short-moment errors caused by proposal temporal scale and boundary mismatch?
- Pre-experiment hypothesis: Short GT intervals may be over-extended by predictions.
- Input/model/data: Existing M2D/QD predictions and GT windows.
- Method/result: Short cases show severe candidate/geometry degradation; grouping definitions were later audited explicitly.
- Effect on final method: Established the short-scale failure profile.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R4 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Geometry decomposition.
- Research question: Is short failure dominated by missing candidates or ranking?
- Pre-experiment hypothesis: If relevant candidates are absent, candidate generation is the main bottleneck.
- Input/model/data: M2D/QD candidate pools and IoU geometry.
- Method/result: Retrieval/candidate failure dominates the 0–5s decomposition.
- Effect on final method: Moved the investigation toward temporal evidence-to-candidate conversion.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R5 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Temporal evidence-to-candidate analysis.
- Research question: Does representation-level temporal evidence exist when AMR candidates are missing?
- Pre-experiment hypothesis: The representation may contain query-relevant temporal evidence that QD-DETR does not exploit.
- Input/model/data: M2D audio/text embeddings, GT, QD predictions.
- Method/result: Many failures have evidence proxy support but lack suitable QD candidates.
- Effect on final method: Supported evidence-guided candidate generation as an intervention target.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R6 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Evidence proposal oracle study.
- Research question: Can evidence-derived proposals recover missing candidate regions?
- Pre-experiment hypothesis: Evidence-only proposals should raise candidate recall if the evidence is useful.
- Input/model/data: Temporal evidence curves and GT.
- Method/result: Evidence proposals and their union with QD candidates improve oracle/candidate recall in the stored analysis.
- Effect on final method: Justified a non-parametric proposal prototype.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R7 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Non-parametric evidence proposal prototype.
- Research question: Which fixed/adaptive evidence windows are useful?
- Pre-experiment hypothesis: Simple peak/threshold proposals can test proposal conversion without training.
- Input/model/data: M2D temporal evidence and GT.
- Method/result: Evidence proposals are useful but do not by themselves establish final ranking behavior.
- Effect on final method: Led to a minimal implementation rather than a complex neural module.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R8 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: EGCG design specification.
- Research question: What minimal architecture directly addresses the diagnosed candidate bottleneck?
- Pre-experiment hypothesis: Add evidence-guided candidates and evidence-aware ranking while keeping baseline encoders/backbone fixed.
- Input/model/data: Existing QD-DETR and M2D evidence analysis.
- Method/result: Evidence-guided Candidate Generation (EGCG) v1 design selected.
- Effect on final method: Defined the implementation scope.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R9 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Minimal non-parametric EGCG prototype.
- Research question: Does evidence proposal augmentation improve candidate recall?
- Pre-experiment hypothesis: Query-conditioned proposals should recover more relevant candidates than QD alone or controls.
- Input/model/data: M2D evidence, QD proposals, random/shuffled controls.
- Method/result: Candidate recall improves; random/shuffled controls are weaker.
- Effect on final method: Validated the proposal-generation component.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R10 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: End-to-end inference integration.
- Research question: Does candidate recall improvement translate under the original ranking pipeline?
- Pre-experiment hypothesis: Naive augmentation should improve final retrieval if candidate availability is sufficient.
- Input/model/data: QD proposals plus evidence proposals and existing ranker.
- Method/result: Candidate recall rises but final R1 falls; ranking mismatch is exposed.
- Effect on final method: Motivated evidence-aware re-ranking.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R11 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Evidence-aware reranking diagnostic.
- Research question: Can score fusion recover evidence candidates into final ranking?
- Pre-experiment hypothesis: The R10 failure may be caused by score mismatch rather than absent evidence.
- Input/model/data: QD scores, evidence scores, fused candidate pools.
- Method/result: Evidence-aware scoring recovers part of the candidate gain, especially for short moments.
- Effect on final method: Added ranking as the second EGCG component.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R12 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Complete inference-only EGCG v1 pipeline.
- Research question: Do proposal generation plus evidence-aware ranking improve final retrieval?
- Pre-experiment hypothesis: Combining both diagnosed interventions should outperform naive augmentation.
- Input/model/data: Frozen QD-DETR, evidence proposals, fusion and re-ranking.
- Method/result: Full inference pipeline succeeds in the stored artifact.
- Effect on final method: Established the inference-only EGCG structure.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R13 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Trainable EGCG design.
- Research question: Can a lightweight learnable evidence pathway be added without changing encoders/backbone?
- Pre-experiment hypothesis: An auxiliary evidence head should learn temporal alignment from GT intervals.
- Input/model/data: Frozen QD-DETR states and GT token labels.
- Method/result: Selected minimal Evidence Head plus auxiliary evidence loss.
- Effect on final method: Defined trainable EGCG prototype.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R14 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Minimal trainable Evidence Head.
- Research question: Can the evidence head learn temporal alignment?
- Pre-experiment hypothesis: Auxiliary BCE supervision should improve evidence recall without changing final QD predictions.
- Input/model/data: Frozen QD-DETR projected audio/text states.
- Method/result: Evidence recall improves strongly; final AMR is unchanged because the head is not yet connected.
- Effect on final method: Established learnable evidence supervision.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R15 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Connect learned evidence to proposal generation.
- Research question: Can learned evidence improve final AMR after proposal integration?
- Pre-experiment hypothesis: Connecting evidence to proposals should raise candidate recall.
- Input/model/data: Frozen QD-DETR plus trainable Evidence Head.
- Method/result: Candidate recall improves; the remaining final ranking/localization gap persists.
- Effect on final method: Motivated a trainable evidence-aware ranker.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R16 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Localization bottleneck diagnosis.
- Research question: After evidence-guided recovery, is short failure boundary, ranking, or refinement limited?
- Pre-experiment hypothesis: High oracle but low Top-1 would indicate ranking/localization selection failure.
- Input/model/data: EGCG candidate pools and intervals.
- Method/result: Oracle@10 is much higher than Top-1 for short moments; ranking remains a major bottleneck.
- Effect on final method: Led to the trainable ranker.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R17 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Trainable evidence-aware ranker.
- Research question: Can a lightweight ranker close the oracle-to-Top-1 gap?
- Pre-experiment hypothesis: Candidate audio/query features plus QD/evidence/geometry scalars should select better candidates.
- Input/model/data: Frozen QD-DETR, evidence proposals, train-split candidate IoUs.
- Method/result: Stored result reports EGCG ranker success and reduced mismatch.
- Effect on final method: Added trainable ranking to EGCG.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R18 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Paper-level QD-DETR validation.
- Research question: Are the EGCG components supported by ablations, controls, and seeds?
- Pre-experiment hypothesis: Full EGCG should improve over QD-DETR and controls.
- Input/model/data: QD-DETR, evidence head, proposal generation, ranker, controls.
- Method/result: Historical multi-seed validation artifacts were collected; later forensic review found implementation provenance gaps.
- Effect on final method: Prompted reconstruction/forensics before final freezing.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R19 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Cross-model EGCG validation on UVCOM.
- Research question: Does EGCG generalize beyond QD-DETR?
- Pre-experiment hypothesis: The same evidence mechanism may improve another AMR architecture.
- Input/model/data: UVCOM checkpoint with fixed evidence/ranker recipe.
- Method/result: Cross-model result is mixed: candidate metrics improve but final R1 does not materially improve.
- Effect on final method: Limits the claim to one strong cross-model validation, not universal generalization.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R20 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Cross-model failure-profile analysis.
- Research question: Why does EGCG help QD-DETR more than UVCOM?
- Pre-experiment hypothesis: Architecture-specific candidate/ranking interactions may explain the difference.
- Input/model/data: Existing QD-DETR and UVCOM predictions.
- Method/result: Baseline candidate weakness is similar; gain difference is consistent with architecture-specific interaction but not causal with two models.
- Effect on final method: Added an explicit limitation to the research claim.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R21 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Qualitative mechanistic analysis.
- Research question: How do evidence proposals and ranking change individual cases?
- Pre-experiment hypothesis: Representative timelines should show candidate recovery and selection changes.
- Input/model/data: Stored R18/R11 predictions, evidence proposals and figures.
- Method/result: Qualitative cases show improvements and remaining failures; evidence curves are proposal-level visualizations.
- Effect on final method: Provided mechanistic case archive with explicit visualization limitation.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R22 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Paper package preparation.
- Research question: Can existing results be organized into paper tables/figures?
- Pre-experiment hypothesis: A structured package should make claims and sources auditable.
- Input/model/data: Existing experiment outputs.
- Method/result: Tables, figures, summary and outline were organized.
- Effect on final method: Prepared paper-level organization without new experiments.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R23 — COMPLETE_WITH_GAPS

- Classification: `DOCUMENTED`
- Purpose: Reviewer-defense preparation.
- Research question: Are complexity, cost, training details and controls documented?
- Pre-experiment hypothesis: Reviewer questions require exact accounting and additional controls.
- Input/model/data: Existing reports and checkpoints.
- Method/result: Reviewer package was prepared, with remaining gaps identified.
- Effect on final method: Motivated final complexity/ablation closure.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R24 — BLOCKED

- Classification: `DOCUMENTED`
- Purpose: Final ablation/complexity attempt.
- Research question: Can missing ranking controls be run under strict train/test separation?
- Pre-experiment hypothesis: A strict reconstruction is required before accepting historical comparisons.
- Input/model/data: Recovered implementation and artifacts.
- Method/result: Blocked by strict training/provenance mismatch; no forced historical matching.
- Effect on final method: Led to reconstruction and freeze policy.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R25 — BLOCKED_METHOD_SPEC_INCOMPLETE

- Classification: `DOCUMENTED`
- Purpose: Reproducibility reconstruction.
- Research question: Can the historical implementation be reconstructed from artifacts?
- Pre-experiment hypothesis: Forensics should recover exact choices instead of relying on design prose.
- Input/model/data: R14–R18 artifacts and metadata.
- Method/result: Reconstruction documented implementation choices and unresolved mismatch.
- Effect on final method: Established forensic source of truth.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R26 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Implementation forensics.
- Research question: What exact source/bytecode choices define historical EGCG?
- Pre-experiment hypothesis: Recovered source and bytecode can resolve ambiguous design-vs-code differences.
- Input/model/data: Forensics source, bytecode, metadata and git records.
- Method/result: Forensic recovery passed and identified R14/R15/R17 implementation authority.
- Effect on final method: Became the implementation authority for clean EGCG.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R27 — REPRODUCTION_MISMATCH

- Classification: `DOCUMENTED`
- Purpose: Clean EGCG reproduction gate.
- Research question: Does the recovered clean implementation exactly reproduce historical R18 numbers?
- Pre-experiment hypothesis: Baseline should match exactly; trainable EGCG may vary within recovered behavior.
- Input/model/data: Frozen QD-DETR, R14/R17 training, seed 2023.
- Method/result: Baseline reproduces exactly; EGCG differs numerically (historical 25.61 vs clean 26.80 R1@0.7), so mismatch was documented rather than forced.
- Effect on final method: Led directly to freezing EGCG v1.0.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R28 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Freeze EGCG v1.0 and strict validation ablation.
- Research question: What are clean validation-only ranking controls under the frozen implementation?
- Pre-experiment hypothesis: A frozen implementation should support train-only controls without test leakage.
- Input/model/data: Frozen EGCG v1.0, train/validation splits.
- Method/result: Validation-only A–D controls and complexity/runtime artifacts were produced; semantics later required R29 correction.
- Effect on final method: Archived as a validation diagnostic, not the final paper ablation.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R29 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Ablation semantics audit.
- Research question: Did R28's no-evidence control actually remove evidence-guided candidate generation?
- Pre-experiment hypothesis: If all controls share the full pool, R28 is a ranking-input ablation rather than a pipeline ablation.
- Input/model/data: R28 script, R17 ranker, R18 pipeline.
- Method/result: Confirmed B/C/D shared the full EGCG pool; R28 B is not historical E3.
- Effect on final method: Required a final clean A–D ablation.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R30 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Final clean validation ablation.
- Research question: Can candidate generation and evidence-aware ranking be separated cleanly?
- Pre-experiment hypothesis: B QD-only, C QD+Evidence with the same no-evidence ranker, D full EGCG should isolate the stages.
- Input/model/data: Frozen evidence, train/validation split, fixed ranker budget.
- Method/result: Clean semantics show candidate recall gains in C and larger final gains in D.
- Effect on final method: Provided the paper-quality validation ablation definition.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`

## R31 — COMPLETE

- Classification: `DOCUMENTED`
- Purpose: Final frozen test ablation and efficiency benchmark.
- Research question: Do the clean ablation gains hold on the official test split and what is the cost?
- Pre-experiment hypothesis: Frozen checkpoints should reproduce the final test comparison without additional fitting.
- Input/model/data: Frozen Evidence Head/full Ranker, R30 no-evidence control, official test split.
- Method/result: Full EGCG reaches R1@0.7 26.80 vs QD-DETR 18.93; efficiency and checkpoint integrity are recorded.
- Effect on final method: Final evaluation artifact for the research archive.
- Sources:
  - `/private/research-artifact`
  - `/private/research-artifact`
