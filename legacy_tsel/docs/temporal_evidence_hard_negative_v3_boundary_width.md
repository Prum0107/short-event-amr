# Temporal Evidence Hard Negative V3: Boundary Width Split

## Goal

V2 showed that boundary-style hard negatives are more useful than semantic false
peaks for strict `R1@0.7`. V3 splits the boundary group:

```text
boundary_distractor only
over_wide + under_wide only
boundary_distractor -> width staged
```

The goal is to identify whether the V2 gain comes from suppressing nearby
start/end distractors or from correcting span width.

## Filtered Hard-Negative Sets

Generated under:

```text
results/temporal_evidence_hard_negatives_v2_filters/
```

| Filter | Samples | Hard Negatives | Types |
|---|---:|---:|---|
| boundary_distractor_only | 590 | 1201 | `boundary_distractor` |
| width_only | 1265 | 3344 | `over_wide`, `under_wide` |
| boundary_group | 1291 | 4545 | all boundary types |

## Evidence/Decoder Results

All evidence runs start from:

```text
results/evidence_baseline_tclap_inspired_v1/best.pt
```

All decoder runs use:

```text
shape_v2, topn_per_source=2, seed=2026
```

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T-CLAP-inspired baseline | 38.07 | 26.70 | 41.48 | 47.73 | 0.3606 | 0.5437 | 64 | 94 |
| V2 boundary-only | 39.77 | 29.83 | 39.77 | 45.74 | 0.3802 | 0.5300 | 61 | 105 |
| V3 boundary_distractor only | 38.07 | 27.27 | 41.19 | 46.59 | 0.3652 | 0.5347 | 61 | 96 |
| V3 width-only | 39.49 | 29.83 | 40.34 | 47.16 | 0.3787 | 0.5402 | 63 | 105 |
| V3 boundary_distractor -> width | 36.36 | 26.99 | 39.49 | 46.02 | 0.3565 | 0.5323 | 59 | 95 |

Result:

```text
The boundary gain is mostly a width-correction gain.
```

`boundary_distractor only` is much weaker than the full boundary group. The
`width-only` run matches the full boundary group's strict `R1@0.7` and keeps
better top-k quality. Staging boundary_distractor before width hurts the final
decoder, so the current curriculum should not start from boundary_distractor.

## Migration Against Adapter Baseline

V3 width-only vs the original T-CLAP-inspired adapter decoder:

```text
strict gains = 20
strict losses = 9
net strict = +11
avg top1 IoU delta = +0.0181
avg top5 IoU delta = -0.0035
```

Most gains are:

```text
candidate_exists -> good: 19
```

So width negatives mainly help rank an already available good-width candidate
above a worse candidate.

## Candidate-Level Fusion Comparison

The best V3 evidence run was connected back to candidate-level fusion by
replacing the adapter branch with:

```text
results/temporal_evidence_hn_v3_width_only/full_train_eval/predictions_evidence_samples.json
results/temporal_evidence_hn_v3_width_only/full_val_eval/predictions_evidence_samples.json
results/learned_evidence_decoder_temporal_hn_v3_width_only_shape_v2_top2/best.pt
```

Output:

```text
results/candidate_level_representation_fusion_temporal_hn_v3_width_only
```

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate-level fusion V1 | 41.48 | 30.97 | 40.91 | 47.44 | 0.3935 | 0.5624 | 76 | 109 |
| Width-only candidate fusion | 41.19 | 31.53 | 41.19 | 48.01 | 0.3847 | 0.5447 | 78 | 111 |

This is the first run above the previous candidate-level fusion strict top1:

```text
30.97 -> 31.53
```

The improvement is not uniformly cleaner:

```text
strict gains = 22
strict losses = 20
net strict = +2
avg top1 IoU delta = -0.0087
avg top5 IoU delta = -0.0177
```

Interpretation:

- Width-only evidence gives a real strict-top1 signal.
- Fusion can use that signal to exceed the previous best `R1@0.7`.
- The current fusion scorer is still too aggressive: it gains strict cases but
  loses some average IoU and semantic coverage.

## Decision

Keep `width-only` hard-negative evidence as the active evidence direction.

Do not continue `boundary_distractor -> width` staging. It underperforms both
width-only and full boundary-only.

The next decision point should be fusion stability:

```text
Run 5-seed candidate-level fusion on width-only evidence.
If mean R1@0.7 is still above 30.97 or semantic-risk calibration improves it,
continue with fusion/risk calibration.
If the gain collapses to a seed artifact, return to evidence training and test
width-only schedules with safer semantic false-peak auxiliary loss.
```

Update: the follow-up semantic/width curriculum and fusion stability check are
recorded in `docs/temporal_evidence_hard_negative_v4_semantic_width_fusion.md`.
The best branch is `semantic false-peak warmup -> width correction`, not pure
width-only.
