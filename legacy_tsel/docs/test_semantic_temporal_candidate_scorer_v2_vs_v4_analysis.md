# V2 Scorer vs V4 Fusion Test Analysis

## Setup

This compares:

```text
V4 candidate-level fusion best frozen-test checkpoint
vs.
V2 quality_guard semantic-temporal-risk scorer, seed 2027
```

Seed 2027 is used for case analysis because it is the validation-best V2 seed
inside the frozen five-seed protocol. The five-seed aggregate remains the main
quantitative result.

## Quantitative Summary

Official-style five-seed test:

| System | R1@0.5 | R1@0.7 | mAP(avg) | mAP@0.5 | mAP@0.75 |
|---|---:|---:|---:|---:|---:|
| V4 candidate-level fusion | 34.65+-0.45 | 23.42+-0.60 | 18.66+-0.31 | 28.15+-0.41 | 18.98+-0.35 |
| V2 quality_guard scorer | 35.13+-0.24 | 23.59+-0.28 | 18.53+-0.08 | 28.06+-0.12 | 18.71+-0.19 |

V2 is slightly better and more stable on strict `R1@0.7`, but slightly lower on
average precision. This supports the current framing:

```text
V2 is an interpretable evidence-selection framework, not a full-metric
replacement for V4 fusion.
```

## Test-Side Intervention Migration

Representative case-analysis comparison:

| Metric | V4 fusion | V2 scorer |
|---|---:|---:|
| Changed | 1212 | 1166 |
| Improved | 577 | 547 |
| Regressed | 283 | 259 |
| Semantic recovery | 170 | 171 |
| Temporal positive | 360 | 341 |
| Anchor regression | 57 | 43 |
| Anchor weakening | 38 | 25 |
| Good regressed | 57 | 43 |
| Intervention precision | 0.476 | 0.469 |

Pairwise migration:

| V4 outcome -> V2 outcome | Count |
|---|---:|
| improved -> improved | 454 |
| same -> same | 381 |
| regressed -> regressed | 202 |
| improved -> same | 90 |
| same -> improved | 63 |
| regressed -> same | 50 |
| regressed -> improved | 40 |
| improved -> regressed | 38 |
| same -> regressed | 29 |

Key counts:

```text
V2 avoids V4 regressions: 90
V2 introduces new regressions: 67
V2 improves cases V4 did not improve: 103
V4 improves cases V2 did not improve: 128
V2 protects anchors regressed by V4: 23
V2 introduces anchor risk not present in V4: 9
```

## Evidence-Head Behavior

Mean V2 top-candidate head values:

| Group | Count | Quality | Semantic | Temporal | Anchor reliable | Anchor guard | Anchor risk | Semantic risk | Temporal risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all changed | 1166 | 0.377 | 0.217 | 0.567 | 0.131 | 0.043 | 0.088 | 0.258 | 0.386 |
| improved | 547 | 0.455 | 0.222 | 0.663 | 0.159 | 0.055 | 0.110 | 0.232 | 0.318 |
| regressed | 259 | 0.402 | 0.133 | 0.644 | 0.165 | 0.051 | 0.109 | 0.345 | 0.334 |
| semantic recovery | 171 | 0.288 | 0.542 | 0.352 | 0.013 | 0.008 | 0.007 | 0.103 | 0.444 |
| temporal positive | 341 | 0.490 | 0.107 | 0.790 | 0.150 | 0.066 | 0.106 | 0.304 | 0.234 |
| anchor protected vs V4 | 23 | 0.568 | 0.006 | 0.583 | 0.598 | 0.054 | 0.431 | 0.265 | 0.532 |
| anchor new risk vs V4 | 9 | 0.554 | 0.003 | 0.835 | 0.104 | 0.046 | 0.064 | 0.219 | 0.300 |

Interpretation:

- `semantic_recovery` cases have high semantic gain and very low anchor
  reliability, which matches the intended behavior.
- `temporal_positive` cases have the highest temporal gain and strict-gain
  evidence.
- Anchor-protected cases have much higher anchor reliability and anchor risk
  than new anchor-risk cases, showing that the V2 heads are learning the right
  reliability signal.
- Temporal gain is still high in some regressed cases, so the remaining
  weakness is risk calibration rather than lack of temporal evidence.

## Decision

The current paper story should be:

```text
Hard-negative evidence improves the temporal-semantic representation.
Candidate-level fusion gives the strongest broad retrieval score.
V2 semantic-temporal-risk scorer makes the selection process interpretable and
slightly improves strict top1 stability by reducing anchor/good-case regressions.
```

Detailed machine-readable analysis:

```text
results/test_semantic_temporal_candidate_scorer_v2_quality_guard_seed2027_cases/v2_vs_v4_analysis.json
results/test_semantic_temporal_candidate_scorer_v2_quality_guard_seed2027_cases/v2_vs_v4_analysis.md
```
