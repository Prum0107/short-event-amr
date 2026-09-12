# Default-Anchored Intervention Gate V3

## Purpose

Candidate-Level Fusion V1 showed the strongest strict top1 result so far:

```text
R1@0.7 = 30.97
```

But it over-intervenes. Candidate-Level Risk Calibration V2b improved semantic coverage but did not solve replacement reliability.

V3 therefore freezes the V1 candidate scorer and trains a smaller default-anchored gate:

```text
MS-CLAP default prediction
V1 top merged candidate
-> should the merged candidate replace MS-CLAP?
```

The goal is not to rerank all candidates again. The goal is to learn the intervention decision directly.

## Script

```text
src/train_default_anchored_intervention_gate.py
```

## Setup

Frozen candidate generator:

```text
results/candidate_level_representation_fusion_v1/best.pt
```

Inputs:

```text
MS-CLAP evidence:
results/evidence_baseline_release_v1/full_train_eval/predictions_evidence_samples.json
results/evidence_baseline_release_v1/full_val_eval/predictions_evidence_samples.json

Temporal-adapter evidence:
results/evidence_baseline_tclap_inspired_v1/full_train_eval/predictions_evidence_samples.json
results/evidence_baseline_tclap_inspired_v1/full_val_eval/predictions_evidence_samples.json
```

Candidate setup:

```text
topn_per_source: 2
feature_version: shape_v2
```

Gate labels:

| Split | Intervene | Gain | Risk | Good risk |
|---|---:|---:|---:|---:|
| Train | 1,093 | 1,075 | 1,107 | 49 |
| Val | 132 | 129 | 223 | 25 |

The validation split has more risky top candidates than beneficial top candidates, so direct intervention learning is a hard calibration problem.

## Five-Seed Validation Result

Seeds:

```text
2026, 2027, 2028, 2029, 2030
```

Balanced gate validation `R1@0.7`:

| Seed | R1@0.7 |
|---:|---:|
| 2026 | 29.26 |
| 2027 | 28.98 |
| 2028 | 28.12 |
| 2029 | 28.41 |
| 2030 | 30.11 |

Five-seed average:

```text
mean R1@0.7 = 28.98
std R1@0.7 = 0.70
```

Precision gate:

```text
mean R1@0.7 = 27.33
std R1@0.7 = 0.97
```

## Best Seed Comparison

Best seed:

```text
2030
```

| System | R1@0.5 | R1@0.7 | Top1 IoU | Semantic miss | Good |
|---|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 34.09 | 25.85 | 0.3271 | 85 | 91 |
| Temporal-adapter shape_v2 top2 | 38.07 | 26.70 | 0.3606 | 64 | 94 |
| Risk-aware representation gate V2 | 38.35 | 29.83 | 0.3738 | 77 | 105 |
| Candidate-level fusion V1 | 41.48 | 30.97 | 0.3935 | 76 | 109 |
| Candidate risk reranker V2b | 38.92 | 30.40 | 0.3751 | 62 | 107 |
| Default-anchored gate V3 best seed | 39.20 | 30.11 | 0.3742 | 78 | 106 |
| Merged candidate oracle | 74.15 | 63.07 | 0.6970 | 15 | 222 |

Best-seed V3 intervention quality:

```text
changed: 106
improved: 63
regressed: 33
good_regressed: 9
recovered_semantic_miss: 12
intervention_precision: 59.43%
```

This is cleaner than V1, but the score is lower.

## Interpretation

V3 is a negative result for the current policy direction:

```text
V1 raw candidate fusion remains the best strict top1 system.
V2b is better for semantic coverage and top-k.
V3 improves intervention cleanliness, but loses too much recall.
```

The five-seed average confirms this is not just one unlucky seed:

```text
V3 balanced mean R1@0.7: 28.98
V1 R1@0.7: 30.97
V2b R1@0.7: 30.40
```

The current bottleneck is not simply whether the gate is default-anchored. The model does not yet have enough signal to decide, from the top merged candidate alone, whether replacement is safe.

## Decision Point

This is a point where the next direction should be chosen deliberately.

Reasonable next options:

```text
1. Keep V1 as the current strict-top1 system and move to stronger representation/evidence supervision.
2. Keep V2b as the semantic-coverage system and analyze top-k candidates instead of top1 only.
3. Build hard-negative intervention training from V1 good-regression cases.
4. Stop tuning gates and return to temporal semantic evidence learning itself.
```

My recommendation is to stop gate-only tuning for now. The merged candidate oracle is huge, but three gate variants have shown the same pattern:

```text
cleaner gates reduce harmful interventions,
but they also give back too much of the candidate-level gain.
```

The next major improvement likely needs stronger evidence supervision or hard-negative intervention data, not another threshold gate.

## Reports

```text
results/default_anchored_intervention_gate_v3/report.html
results/default_anchored_intervention_gate_v3/summary.json
```
