# Evidence-Shape-Aware Learned Decoder v4

## Question

The two-stage boundary-head experiment showed that boundary knowledge cannot be added cleanly by training only separate start/end heads.

This experiment tests a different hypothesis:

> Boundary knowledge should be decoded from the temporal shape of semantic evidence.

Instead of retraining the evidence model, we keep the semantic-only evidence model fixed and improve only the learned decoder.

## Code Change

Script:

`src/train_learned_evidence_decoder.py`

New option:

```bash
--feature_version shape_v2
```

The old feature set remains available:

```bash
--feature_version basic
```

`shape_v2` adds explicit evidence-shape features:

- global evidence mean/std/max and percentiles
- candidate evidence mass ratio
- candidate peak prominence over nearby context
- fraction of candidate frames above global percentiles
- longest high-evidence run inside the candidate
- plateau fraction around the candidate peak
- left/right boundary contrast
- distance between candidate center and global/top evidence peaks
- local peak position within the candidate

This is a decoder-only change. It does not update CLAP features or the evidence model.

## Input Evidence

Evidence source:

`results/evidence_baseline_v31_semantic_only_lsem025`

This source was chosen because semantic-only had the best strict learned-decoder top1 in v3.1.

## Main Results

| Decoder | Candidate TopN Per Source | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 | Semantic Miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| basic learned decoder | 10 | 35.23% | 25.28% | 34.38% | 42.33% | 0.3276 | - | 97 |
| shape_v2 learned decoder | 2 | 34.94% | 25.85% | 38.35% | 43.18% | 0.3316 | 0.4959 | 77 |
| shape_v2 learned decoder | 10 | 32.67% | 22.44% | 34.94% | 43.18% | 0.3130 | 0.4789 | 90 |

Reports:

- `results/learned_evidence_decoder_shape_v2_semantic_only_top2/report.html`
- `results/learned_evidence_decoder_shape_v2_semantic_only_top10/report.html`

## Key Finding

`shape_v2 + top2` is the best current decoder setting for this evidence source:

- strict top1 improves from `25.28%` to `25.85%`
- top3 improves from `34.38%` to `38.35%`
- top5 improves from `42.33%` to `43.18%`
- semantic misses drop from `97` to `77`

This is the first result that improves strict top1 while also strongly reducing semantic misses.

## Candidate Complexity Finding

The top10 setting is worse than top2 even with the richer shape features.

This means:

> More candidates are not automatically better. Candidate generation is part of the decoder.

The top10 candidate set contains more possible correct windows, but it also contains many confusing high-evidence distractors. The learned scorer is not yet strong enough to rank them reliably.

So the decoder problem has two coupled parts:

1. generate a compact candidate set that keeps the useful windows
2. rerank those windows using evidence-shape features

## Migration: Basic Top10 to ShapeV2 Top2

Report:

`results/decoder_migration_semantic_only_basic_to_shape_v2_top2/report.html`

| Metric | Value |
|---|---:|
| Stable samples | 212 |
| Category improved | 69 |
| Category regressed | 42 |
| IoU improved | 13 |
| IoU regressed | 16 |
| Semantic misses recovered | 26 |
| Semantic miss to good | 0 |
| Good regressed | 19 |
| Avg top1 IoU delta | +0.0040 |
| Avg top5 IoU delta | +0.0290 |
| Avg evidence gap delta | 0.0000 |

Category shift:

| Category | Basic Top10 | ShapeV2 Top2 |
|---|---:|---:|
| good | 89 | 91 |
| candidate_exists | 60 | 61 |
| boundary_error | 73 | 97 |
| evidence_good_decode_bad | 33 | 26 |
| semantic_miss | 97 | 77 |

Important transitions:

- `semantic_miss -> boundary_error`: 22
- `semantic_miss -> candidate_exists`: 4
- `candidate_exists -> good`: 19
- `good -> candidate_exists`: 16
- `boundary_error -> semantic_miss`: 5

## Interpretation

The shape-aware decoder does not magically turn semantic misses directly into good predictions. Instead, it often moves them into `boundary_error`.

That is actually a useful research signal:

> The model is finding the semantically relevant region more often, but still needs sharper temporal boundaries.

This fits our overall direction:

`semantic evidence -> candidate explanation -> boundary-aware decoding`

The shape-aware decoder improves the first two parts. The remaining problem is stricter boundary selection inside already relevant regions.

## Updated Research Direction

The next step should focus on candidate complexity control, not model-level hard-negative tuning.

Recommended next experiment:

### Source-aware candidate gate

Train or design a lightweight gate that decides how many candidates to keep from each decoder source.

Hypothesis:

> A compact source-aware candidate set plus shape-aware reranking will outperform simply increasing topN.

Possible directions:

- top2/top3 candidate curriculum
- source-specific candidate quotas
- penalty for over-wide candidates before reranking
- two-stage reranking: coarse source gate, then shape scorer
- case analysis of the 22 `semantic_miss -> boundary_error` transitions

The important shift is this:

> We are no longer only asking whether evidence is high. We are asking whether the temporal shape of evidence explains the correct boundary.

Follow-up diagnosis:

`docs/feature_ceiling_diagnosis_v1.md`

This diagnosis shows that `shape_v2 top2` mainly reduces semantic misses where MS-CLAP already had usable evidence. The remaining semantic misses are split almost evenly between representation ceiling and decoder/candidate ceiling, so the next step should improve candidate gating before replacing the audio representation.

Candidate-gating follow-up:

`docs/source_aware_candidate_gate_v1.md`

The first source-aware gate improves strict top1 but worsens semantic miss, suggesting that candidate policy controls a trade-off between semantic coverage and boundary precision.
