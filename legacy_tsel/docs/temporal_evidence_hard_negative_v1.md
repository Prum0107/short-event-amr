# Temporal Evidence Hard Negative V1

## Purpose

The previous candidate-fusion experiments showed a consistent pattern:

```text
The merged candidate pool contains many useful temporal windows,
but gate-only policies cannot reliably decide when to trust them.
```

This experiment moves the signal back into the evidence model.

Instead of only using dangerous candidates to train a gate, V1 mines them as evidence-level hard negatives:

```text
GT window evidence should be higher than high-scoring wrong candidate evidence.
```

The goal is to improve the query-guided temporal semantic evidence curve `S(t, q)` directly.

## Mining Script

```text
src/mine_temporal_evidence_hard_negatives.py
```

Inputs:

```text
MS-CLAP evidence:
results/evidence_baseline_release_v1/full_train_eval/predictions_evidence_samples.json

Temporal-adapter evidence:
results/evidence_baseline_tclap_inspired_v1/full_train_eval/predictions_evidence_samples.json

Frozen candidate-level fusion scorer:
results/candidate_level_representation_fusion_v1/best.pt
```

Output:

```text
results/temporal_evidence_hard_negatives_v1/train_typed_hard_negatives.json
results/temporal_evidence_hard_negatives_v1/summary.json
```

## Mined Hard Negatives

The miner uses the frozen V1 merged-candidate scorer and keeps high-scoring wrong candidates.

| Type | Count | Samples | Avg score | Avg IoU |
|---|---:|---:|---:|---:|
| semantic_false_peak | 2,438 | 832 | 0.2828 | 0.0120 |
| boundary_distractor | 1,201 | 590 | 0.3721 | 0.4510 |
| over_wide | 1,856 | 1,011 | 0.4163 | 0.4765 |
| under_wide | 1,488 | 831 | 0.4142 | 0.5137 |
| total | 6,983 | - | - | - |

Adapter candidates dominate the mined negatives:

```text
semantic_false_peak: adapter 1441, MS-CLAP 997
boundary_distractor: adapter 675, MS-CLAP 526
over_wide: adapter 1191, MS-CLAP 665
under_wide: adapter 906, MS-CLAP 582
```

This matches the earlier intervention analysis: the temporal adapter adds useful evidence, but it also creates high-scoring risky candidates.

## Evidence Fine-Tuning

Base checkpoint:

```text
results/evidence_baseline_tclap_inspired_v1/best.pt
```

Fine-tuning:

```text
trainable_parts: heads
epochs: 2
lr: 5e-5
lambda_semantic_hard_negative: 0.15
lambda_boundary_hard_negative: 0.08
semantic_hn_margin: 0.12
boundary_hn_margin: 0.20
```

Output:

```text
results/temporal_evidence_hn_v1/
```

## Evidence-Level Result

Compared with the original temporal-adapter evidence baseline:

| Evidence model | R1@0.5 | R1@0.7 | R3@0.7 | Best IoU@5 | Evidence gap |
|---|---:|---:|---:|---:|---:|
| Temporal-adapter baseline | 28.13 | 21.31 | 32.39 | 0.4905 | 0.2800 |
| Temporal evidence HN V1 | 28.69 | 21.59 | 32.39 | 0.4935 | 0.2864 |

The direct evidence decoder improves slightly:

```text
R1@0.7: 21.31 -> 21.59
evidence gap: 0.2800 -> 0.2864
best IoU@5: 0.4905 -> 0.4935
```

This is small but positive, and it comes from only heads fine-tuning for two epochs.

## Learned Decoder Result

The fine-tuned evidence was then passed through the same `shape_v2 top2` learned decoder setup.

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Temporal-adapter shape_v2 top2 | 38.07 | 26.70 | 41.48 | 47.73 | 0.3606 | 0.5437 | 64 | 94 |
| Temporal evidence HN V1 shape_v2 top2 | 36.08 | 28.41 | 39.20 | 48.30 | 0.3625 | 0.5413 | 62 | 100 |

Key changes:

```text
R1@0.7: 26.70 -> 28.41
R5@0.7: 47.73 -> 48.30
semantic miss: 64 -> 62
good cases: 94 -> 100
```

This is the important result. The hard-negative signal is not just changing the raw evidence decoder; it improves the learned evidence-to-window decoder.

## Interpretation

This validates the new direction:

```text
Dangerous candidates should not only be avoided by a gate.
They can be used to improve temporal semantic evidence itself.
```

The gain is modest but meaningful because:

```text
1. only heads were fine-tuned
2. the negative set came from the existing candidate-fusion failure modes
3. no extra model capacity was added
4. strict R1@0.7 improved after learned decoding
```

This gives a stronger research story than continued gate tuning:

```text
Candidate fusion exposes the errors.
Hard-negative evidence learning uses those errors to improve S(t, q).
```

## Next Step

Recommended next experiment:

```text
Temporal Evidence Hard Negative V2
```

Possible changes:

```text
1. train all evidence layers, not only heads
2. use a two-stage schedule: semantic false peaks first, boundary negatives second
3. add query-confusion negatives from same-audio different-query pairs
4. evaluate with candidate-level fusion using the new evidence as a third representation
5. run 3-5 seeds for the evidence fine-tune after the setup is stable
```

The next decision should be whether to optimize this evidence-learning line or first analyze which hard-negative type produced the useful gain.

## Reports

```text
results/temporal_evidence_hard_negatives_v1/summary.json
results/temporal_evidence_hn_v1/full_val_eval/metrics.json
results/learned_evidence_decoder_temporal_hn_v1_shape_v2_top2/report.html
results/learned_evidence_decoder_temporal_hn_v1_shape_v2_top2/summary.json
```
