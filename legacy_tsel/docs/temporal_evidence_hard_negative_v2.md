# Temporal Evidence Hard Negative V2

## Goal

V1 showed that dangerous merged candidates can be reused as supervision for the
T-CLAP-inspired temporal evidence model. V2 asks which hard-negative family is
actually responsible for the gain:

```text
semantic_false_peak: high-scoring but wrong semantic region
boundary negatives: boundary_distractor + over_wide + under_wide
```

The experiment keeps the evidence backbone frozen except for the lightweight
heads, then trains the same learned shape_v2 top2 decoder on the exported
evidence candidates.

## Hard Negative Inventory

Mined from the frozen V1 merged-candidate scorer:

| Type | Count | Samples | Avg IoU | Interpretation |
|---|---:|---:|---:|---|
| semantic_false_peak | 2438 | 832 | 0.012 | Strong semantic false positives far from the target |
| boundary_distractor | 1201 | 590 | 0.451 | Near target but wrong start/end choice |
| over_wide | 1856 | 1011 | 0.476 | Candidate covers too much context |
| under_wide | 1488 | 831 | 0.514 | Candidate captures only part of the event |

The boundary group is much closer to the answer in IoU, so it is expected to
affect strict localization more directly than semantic false peaks.

## Controlled Ablations

All runs start from:

```text
results/evidence_baseline_tclap_inspired_v1/best.pt
```

Decoder setup:

```text
shape_v2, topn_per_source=2, hidden_dim=128, lambda_pairwise=0.2, seed=2026
```

| Run | Evidence Training | Decoder Result |
|---|---|---|
| Baseline adapter | no new HN training | `results/learned_evidence_decoder_tclap_inspired_shape_v2_top2` |
| HN V1 combined | semantic + boundary together, 2 epochs | `results/learned_evidence_decoder_temporal_hn_v1_shape_v2_top2` |
| V2 semantic-only | semantic false peaks only, 2 epochs | `results/learned_evidence_decoder_temporal_hn_v2_semantic_only_shape_v2_top2` |
| V2 boundary-only | boundary negatives only, 2 epochs | `results/learned_evidence_decoder_temporal_hn_v2_boundary_only_shape_v2_top2` |
| V2 staged | 1 epoch semantic, then 1 epoch boundary | `results/learned_evidence_decoder_temporal_hn_v2_stage_semantic_boundary_shape_v2_top2` |

## Main Results

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T-CLAP-inspired baseline | 38.07 | 26.70 | 41.48 | 47.73 | 0.3606 | 0.5437 | 64 | 94 |
| HN V1 combined | 36.08 | 28.41 | 39.20 | 48.30 | 0.3625 | 0.5413 | 62 | 100 |
| V2 semantic-only | 36.08 | 27.56 | 39.77 | 47.16 | 0.3561 | 0.5302 | 58 | 97 |
| V2 boundary-only | 39.77 | 29.83 | 39.77 | 45.74 | 0.3802 | 0.5300 | 61 | 105 |
| V2 staged semantic->boundary | 38.07 | 28.41 | 40.91 | 47.44 | 0.3675 | 0.5536 | 65 | 100 |

Single-seed conclusion:

```text
boundary-only is the strongest strict top1 run:
R1@0.7 = 29.83
```

Semantic-only gives the lowest semantic miss count, but it loses more already
correct top1 cases. Staging recovers top-k quality and gives the best Top5 IoU,
but does not beat boundary-only at strict top1.

## Case Transition Analysis

Transitions are measured against the T-CLAP-inspired baseline learned decoder.

| System | Strict Gains | Strict Losses | Net Strict | Avg Top1 Delta | Avg Top5 Delta |
|---|---:|---:|---:|---:|---:|
| HN V1 combined | 21 | 15 | +6 | +0.0019 | -0.0024 |
| V2 semantic-only | 22 | 19 | +3 | -0.0046 | -0.0136 |
| V2 boundary-only | 22 | 11 | +11 | +0.0196 | -0.0138 |
| V2 staged semantic->boundary | 18 | 12 | +6 | +0.0069 | +0.0099 |

Most strict gains are converted from `candidate_exists` cases:

| System | Gains from Candidate Exists | Gains from Boundary Error |
|---|---:|---:|
| HN V1 combined | 20 | 1 |
| V2 semantic-only | 22 | 0 |
| V2 boundary-only | 21 | 1 |
| V2 staged semantic->boundary | 18 | 0 |

The useful effect is therefore not mainly creating new candidates. The candidates
already exist in top-k; hard-negative training changes the evidence shape enough
for the learned decoder to rank a stricter window first.

## Five-Seed Boundary-Only Decoder Check

The boundary-only evidence checkpoint was fixed, and only the learned decoder
seed was changed.

| Seed | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 39.77 | 29.83 | 39.77 | 45.74 | 0.3802 | 0.5300 | 61 | 105 |
| 2027 | 39.77 | 28.69 | 40.34 | 46.59 | 0.3751 | 0.5336 | 59 | 101 |
| 2028 | 39.49 | 28.12 | 40.34 | 46.88 | 0.3660 | 0.5336 | 61 | 99 |
| 2029 | 38.64 | 28.12 | 39.77 | 47.44 | 0.3756 | 0.5468 | 62 | 99 |
| 2030 | 40.06 | 28.41 | 40.34 | 46.59 | 0.3785 | 0.5399 | 62 | 100 |

Mean and population standard deviation:

```text
mean R1@0.7 = 28.64
std  R1@0.7 = 0.63
mean Top1 IoU = 0.3751
mean semantic miss = 61.0
```

This is still above the original T-CLAP-inspired baseline (`26.70`), but the
best-seed result should not be treated as the stable headline result.

## Interpretation

The strongest evidence is:

```text
Boundary negatives are responsible for most of the strict R1@0.7 gain.
```

Reasons:

- Semantic false peaks reduce semantic misses (`64 -> 58`), but the strict gain
  is small because they also damage more previously good predictions.
- Boundary-only produces the largest net strict migration (`+11`) and the best
  single-seed strict result (`29.83`).
- The main successful migration is `candidate_exists -> good`, which means the
  candidate set already contains good windows and the training improves ranking.
- Staged semantic then boundary improves top-k evidence quality (`Top5 IoU=0.5536`)
  but does not preserve the semantic coverage gain and does not beat boundary-only
  in top1 strict IoU.

## Current Decision

Use V2 boundary-only as the current positive evidence-learning direction, but
report it with the five-seed decoder mean:

```text
boundary-only V2: mean R1@0.7 = 28.64, best seed R1@0.7 = 29.83
```

Do not continue gate-only tuning until this evidence line is better understood.

## Next Step

The next useful split is inside the boundary group:

```text
boundary_distractor only
over_wide + under_wide only
boundary_distractor -> width correction staged
```

This should tell us whether the strict gain comes from start/end distractor
suppression or from correcting span width.

Update: this split is now recorded in
`docs/temporal_evidence_hard_negative_v3_boundary_width.md`. The useful signal
comes mainly from width correction (`over_wide + under_wide`).
