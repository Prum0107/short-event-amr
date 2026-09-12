# Risk-Aware Representation Gate V2

## Purpose

Representation Fusion V1 showed that MS-CLAP and the temporal-adapter evidence are complementary, but the learned selector intervened too broadly.

V2 changes the selector objective:

```text
V1: predict whether adapter is better
V2: predict adapter gain, adapter risk, good-case risk, and utility
```

The goal is to use the adapter only when the expected gain is high and the expected regression risk is low.

## Script

```text
src/train_risk_aware_representation_gate.py
```

## Setup

Inputs are the same as Representation Fusion V1:

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
results/learned_evidence_decoder_shape_v2_semantic_only_top2/best.pt
results/learned_evidence_decoder_tclap_inspired_shape_v2_top2/best.pt
```

## Training Labels

For each query, the adapter intervention is treated as an action.

The model predicts:

```text
gain: adapter prediction is better than MS-CLAP prediction
risk: adapter prediction is worse than MS-CLAP prediction
good_risk: MS-CLAP was good, but adapter is not good
utility: numeric improvement target combining IoU, R1 thresholds, semantic recovery, and good-case regression penalty
```

Label counts:

```text
train gain: 1199
train risk: 546
train good_risk: 83

val gain: 162
val risk: 138
val good_risk: 33
```

## Results

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 34.09 | 25.85 | 38.07 | 42.61 | 0.3271 | 0.4887 | 85 | 91 |
| Temporal-adapter shape_v2 top2 | 38.07 | 26.70 | 41.48 | 47.73 | 0.3606 | 0.5437 | 64 | 94 |
| Fusion selector V1 | 39.20 | 29.55 | 42.61 | 47.73 | 0.3790 | 0.5545 | 75 | 104 |
| Risk-aware balanced gate V2 | 38.35 | 29.83 | 42.61 | 46.59 | 0.3738 | 0.5290 | 77 | 105 |
| Risk-aware precision gate V2 | 35.51 | 26.42 | 38.35 | 42.90 | 0.3348 | 0.4920 | 85 | 93 |
| Oracle MS-or-adapter | 48.01 | 36.08 | 46.88 | 52.27 | 0.4466 | 0.5981 | 64 | 127 |

## Intervention Quality

| System | Changed | Improved | Regressed | Recovered semantic miss | Good regressed | Intervention precision |
|---|---:|---:|---:|---:|---:|---:|
| Fusion selector V1 | 197 | 99 | 54 | 14 | 18 | 50.25 |
| Conservative gate V1 | 106 | 64 | 27 | 11 | 7 | 60.38 |
| Risk-aware balanced gate V2 | 85 | 59 | 21 | 8 | 7 | 69.41 |
| Risk-aware precision gate V2 | 11 | 8 | 3 | 0 | 0 | 72.73 |
| Oracle MS-or-adapter | 162 | 127 | 0 | 21 | 0 | 78.40 |

V2 is a meaningful improvement over V1:

```text
R1@0.7: 29.55 -> 29.83
adapter interventions: 197 -> 85
regressions: 54 -> 21
good regressions: 18 -> 7
intervention precision: 50.25% -> 69.41%
```

The precision gate is too conservative for final performance, but it demonstrates that the model can identify very low-risk adapter interventions.

## Interpretation

The result supports the current research direction:

```text
The temporal adapter should not replace MS-CLAP globally.
It should act as a risk-aware intervention when MS-CLAP evidence is likely insufficient.
```

This is stronger than the V1 fusion result because V2 improves strict top1 performance while reducing the number of risky replacements.

## Remaining Gap

Oracle remains much higher:

```text
risk-aware balanced gate R1@0.7: 29.83
oracle MS-or-adapter R1@0.7: 36.08
oracle 20% budget R1@0.7: 35.51
```

So the bottleneck is now selection reliability, not representation complementarity.

## Next Step

The next experiment should move from whole-prediction selection to candidate-level fusion:

```text
1. keep MS-CLAP and adapter candidate lists
2. merge candidates from both representations
3. train a candidate reranker with risk-aware features
4. compare whether candidate-level fusion recovers oracle gains with fewer regressions
```

This is a better next step than simply making a bigger selector, because the oracle result shows many useful adapter candidates exist but whole-prediction replacement is still too blunt.

## Report

```text
results/risk_aware_representation_gate_v2/report.html
results/risk_aware_representation_gate_v2/summary.json
```
