# Transformer-Assisted Selector V5

## Question

The candidate-context Transformer did not outperform the MLP decoder as a direct top-1 decoder, but it improved top-k candidate quality. This experiment asks whether the Transformer can still provide useful auxiliary evidence for a query-adaptive selector.

## Setup

Script:

`src/train_transformer_assisted_selector.py`

Output:

`results/transformer_assisted_selector_v5`

Inputs:

- Coverage decoder: `results/learned_evidence_decoder_shape_v2_semantic_only_top2/best.pt`
- Precision decoder: `results/learned_evidence_decoder_shape_v2_source_gate_v2_tight/best.pt`
- Transformer decoder: `results/candidate_transformer_source_gate_v2_v1/best.pt`
- Evidence source: `results/evidence_baseline_v31_semantic_only_lsem025`

The selector sees inference-available features only:

- global evidence shape statistics
- each mode's top candidate score, margin, length, center, source, and candidate count
- pairwise agreement between coverage, precision, and Transformer windows
- pairwise score and boundary differences

The training label chooses the best of the three modes on train by strict top-1 quality, then top-5 quality as tie-breaker. Logit biases are tuned on train only.

## Validation Results

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 | Semantic Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Coverage MLP | 34.94 | 25.85 | 38.35 | 43.18 | 0.3316 | 0.4959 | 77 |
| Precision MLP | 37.22 | 26.42 | 39.49 | 43.75 | 0.3425 | 0.4905 | 102 |
| Transformer | 34.66 | 25.00 | 38.92 | 44.89 | 0.3212 | 0.5080 | 100 |
| Two-mode selector baseline | 37.50 | 26.14 | 38.07 | 43.18 | 0.3437 | 0.4921 | 84 |
| Transformer-assisted V5 | 35.23 | 26.42 | 39.77 | 44.60 | 0.3340 | 0.4997 | 87 |
| Three-mode oracle | 42.61 | 32.95 | 41.76 | 46.02 | 0.4011 | 0.5248 | 72 |

Report:

`results/transformer_assisted_selector_v5/report.html`

Migration from two-mode selector:

`results/decoder_migration_two_mode_to_transformer_assisted_v5/report.html`

Feature ceiling diagnosis:

`results/feature_ceiling_diagnosis_transformer_assisted_v5/report.html`

## Selector Behavior

Train labels:

- coverage: 1362
- precision: 557
- transformer: 263

Validation oracle labels:

- coverage: 221
- precision: 83
- transformer: 48

Validation chosen modes:

- coverage: 216
- precision: 100
- transformer: 36

The Transformer is genuinely useful in some cases: the three-mode oracle improves R1@0.7 from 30.97 in the old two-mode oracle to 32.95. This means the Transformer finds correct top-1 windows for examples where both MLP modes fail.

## Migration Diagnosis

Compared with the two-mode selector:

- stable: 296 / 352
- category improved: 19
- IoU improved: 7
- category regressed: 21
- IoU regressed: 9
- semantic misses recovered: 3
- good cases regressed: 10
- average top1 IoU delta: -0.0097
- average top5 IoU delta: +0.0076

This confirms the earlier pattern:

The Transformer improves the candidate set and top-k quality, but its top-1 decision is not calibrated enough. Adding it as a third selectable mode increases top-k recall and strict R1@0.7 slightly, but it loses some of the two-mode selector's semantic coverage and top1 stability.

## Interpretation

This is not evidence that we should scale the Transformer immediately. The current input is still derived from MS-CLAP evidence and handcrafted candidate features. A larger Transformer would likely overfit this candidate feature space before solving the representation problem.

The useful finding is narrower:

Transformer context contains complementary boundary/candidate information, but it should be used conservatively.

## Next Direction

Do not replace the MLP decoder with a larger Transformer yet.

Recommended next step:

Build a conservative Transformer-assisted selector that keeps the two-mode decision as the default and only allows Transformer intervention when it agrees with one MLP mode or has a strong top-k consistency signal. This tests whether Transformer can repair selected failures without becoming an unstable third judge.

If raw audio / T-CLAP features become available, reopen the larger temporal Transformer direction there, because that setting gives the model richer sequence information instead of only candidate-level statistics.
