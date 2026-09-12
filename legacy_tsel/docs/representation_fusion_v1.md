# Representation Fusion V1

## Purpose

This experiment tests whether MS-CLAP evidence and the T-CLAP-inspired temporal-adapter evidence are complementary.

The goal is not to build a large two-stream model yet. The goal is to answer:

```text
Can we recover temporal-adapter gains without sacrificing the MS-CLAP cases that were already good?
```

## Setup

Inputs:

```text
MS-CLAP evidence:
results/evidence_baseline_release_v1/full_train_eval/predictions_evidence_samples.json
results/evidence_baseline_release_v1/full_val_eval/predictions_evidence_samples.json

Temporal-adapter evidence:
results/evidence_baseline_tclap_inspired_v1/full_train_eval/predictions_evidence_samples.json
results/evidence_baseline_tclap_inspired_v1/full_val_eval/predictions_evidence_samples.json
```

Decoders:

```text
MS-CLAP decoder:
results/learned_evidence_decoder_shape_v2_semantic_only_top2/best.pt

Temporal-adapter decoder:
results/learned_evidence_decoder_tclap_inspired_shape_v2_top2/best.pt
```

Script:

```text
src/train_representation_fusion_selector.py
```

## Method

The selector sees features from both representation lines:

```text
global MS-CLAP evidence-shape statistics
global adapter evidence-shape statistics
MS-CLAP top candidate features
adapter top candidate features
cross-representation evidence curve similarity
top-window agreement, score margin, length/center difference
evidence-gap difference
```

It predicts whether to use:

```text
MS-CLAP prediction
or
temporal-adapter prediction
```

The default is MS-CLAP. A conservative variant only lets the adapter intervene under a limited budget.

## Representation Complementarity

Selector labels:

```text
train adapter better: 1199
train MS-CLAP better or tie: 983

val adapter better: 162
val MS-CLAP better or tie: 190
```

This confirms the two representations are not redundant. Adapter wins many cases, but MS-CLAP still wins or ties more than half of validation cases.

## Validation Results

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 34.09 | 25.85 | 38.07 | 42.61 | 0.3271 | 0.4887 | 85 | 91 |
| Temporal-adapter shape_v2 top2 | 38.07 | 26.70 | 41.48 | 47.73 | 0.3606 | 0.5437 | 64 | 94 |
| Learned selector | 39.20 | 29.55 | 42.61 | 47.73 | 0.3790 | 0.5545 | 75 | 104 |
| Conservative adapter gate | 39.20 | 28.98 | 40.91 | 46.02 | 0.3755 | 0.5342 | 76 | 102 |
| Oracle MS-or-adapter | 48.01 | 36.08 | 46.88 | 52.27 | 0.4466 | 0.5981 | 64 | 127 |
| Oracle 20% budget | 46.88 | 35.51 | 45.17 | 49.15 | 0.4338 | 0.5497 | 70 | 125 |

Important:

```text
learned selector improves R1@0.7 from 26.70% to 29.55%
oracle upper bound is 36.08%
20% oracle budget already reaches 35.51%
```

This means the fusion problem is worth studying. The main remaining issue is not whether the representations are complementary; it is how reliably we can detect which one to trust.

## Intervention Analysis

| System | Changed | Improved | Regressed | Recovered semantic miss | Good regressed | Intervention precision |
|---|---:|---:|---:|---:|---:|---:|
| Learned selector | 197 | 99 | 54 | 14 | 18 | 50.25 |
| Conservative adapter gate | 106 | 64 | 27 | 11 | 7 | 60.38 |
| Oracle MS-or-adapter | 162 | 127 | 0 | 21 | 0 | 78.40 |
| Oracle 20% budget | 71 | 71 | 0 | 15 | 0 | 100.00 |

Interpretation:

```text
The learned selector finds useful adapter interventions,
but it still intervenes too broadly.

The conservative gate has lower score than the learned selector,
but it is cleaner: fewer regressions and fewer good-case failures.
```

## Research Conclusion

This validates the proposed fusion direction.

The current best research claim is:

```text
Temporal contrastive adaptation improves temporal semantic evidence.
MS-CLAP and temporal-adapter evidence are complementary.
A learned representation selector can exploit this complementarity,
but the selector must be conservative to avoid replacing reliable MS-CLAP evidence.
```

## Next Step

Do not jump to a large two-stream model yet.

Next experiment should focus on:

```text
utility-aware conservative fusion
```

Specifically:

```text
1. treat adapter intervention as a risk-sensitive action
2. train on improvement/regression labels, not only adapter-better labels
3. prioritize high-precision adapter interventions
4. then test score-level candidate fusion after the gate is stable
```

## Report

```text
results/representation_fusion_ms_tclap_inspired_v1/report.html
results/representation_fusion_ms_tclap_inspired_v1/summary.json
```
