# Hard-Negative Evidence Training v2

## Goal

This experiment tests whether targeted hard negatives can teach the evidence model to suppress high-scoring but temporally wrong semantic peaks.

The setup follows our research direction:

- Learn query-guided temporal semantic evidence `S(t, q)`.
- Mine windows that look semantically plausible under v1 evidence but have low IoU with the ground truth.
- Add a ranking loss so GT evidence windows score higher than these mined false peaks.
- Re-run both rule-based and learned evidence decoders.

## Hard Negative Mining

Input evidence file:

`results/evidence_baseline_release_v1/full_train_eval/predictions_evidence_samples.json`

Output:

`results/hard_negatives_v1/train_hard_negatives.json`

Mining rule:

- Generate candidate windows from existing evidence decoders.
- Keep windows with IoU `< 0.3` against GT.
- Score candidates by top-25% mean evidence inside the window.
- Keep top 3 hard negatives per training query.

Summary:

| Item | Value |
|---|---:|
| Train samples | 2182 |
| Samples with hard negatives | 2182 |
| Total hard negatives | 6546 |
| Avg hard negatives / sample | 3.0 |
| Avg hard-negative evidence score | 0.7154 |
| Avg hard-negative IoU | 0.0621 |
| Max allowed IoU observed | 0.2973 |

This means the mined windows are not random negatives. They are high-evidence false positives that the current model is likely to confuse with the answer.

## Training Change

New files / edits:

- `src/mine_evidence_hard_negatives.py`
- `src/evidence_baseline.py`
- `src/train_evidence_baseline.py`

New training arguments:

- `--hard_negative_path`
- `--lambda_hard_negative`
- `--hn_margin`
- `--hn_topk`

The added loss compares:

`mean evidence over GT window` vs `mean evidence over mined hard-negative window`

and applies:

`relu(margin - gt_score + hard_negative_score)`

For v2 we used:

```bash
--lambda_hard_negative 0.2 --hn_margin 0.1 --hn_topk 3
```

## Evidence Model Result

Checkpoint:

`results/evidence_baseline_v2_hn_l02/best.pt`

| Model | R1@0.5 | R1@0.7 | R3@0.7 | Best IoU@5 | Evidence Gap |
|---|---:|---:|---:|---:|---:|
| v1 evidence baseline | 23.01% | 17.33% | 25.85% | 0.4578 | 0.2178 |
| v2 hard-negative evidence | 23.30% | 17.05% | 26.42% | 0.4504 | 0.2144 |

The direct start/end decoder did not improve. It stayed essentially tied with v1. This is useful: hard-negative training did not magically solve the task through the original boundary heads.

## Rule-Based Evidence Decoder

Report:

`results/evidence_decoder_experiment_v2_hn_l02/report.html`

| Evidence Source | Best Rule Decoder | R1@0.5 | R1@0.7 | R3@0.7 | Top1 IoU | Best IoU@5 |
|---|---|---:|---:|---:|---:|---:|
| v1 | peak_drop | 32.10% | 22.73% | 34.66% | 0.3032 | 0.4901 |
| v2 hard-negative | peak_drop | 32.67% | 23.01% | 32.10% | 0.3130 | 0.4871 |

Rule decoding is roughly stable, with a small gain in top-1 localization quality. It suggests the v2 evidence curve is not worse, but rule decoding is not sensitive enough to fully expose the hard-negative effect.

## Learned Evidence Decoder

Report:

`results/learned_evidence_decoder_v2_hn_l02/report.html`

| Evidence Source | Decoder | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | learned_scorer | 33.24% | 23.58% | 34.66% | 39.20% | 0.3131 | 0.4537 |
| v2 hard-negative | learned_scorer | 35.23% | 24.43% | 35.80% | 42.90% | 0.3299 | 0.4778 |

The learned decoder benefits more clearly from hard-negative evidence:

- `R1@0.7`: +0.85 points
- `R1@0.5`: +1.99 points
- `R5@0.7`: +3.69 points
- `Top1 IoU`: +0.0168
- `Best IoU@5`: +0.0241

Failure-category shift:

| Category | v1 Learned | v2 Learned |
|---|---:|---:|
| good | 83 | 86 |
| boundary_error | 74 | 89 |
| candidate_exists | 55 | 65 |
| semantic_miss | 105 | 84 |
| evidence_good_decode_bad | 35 | 28 |

The important research signal is the drop in `semantic_miss`: v2 evidence gives the decoder fewer completely wrong semantic peaks. Some of those cases become boundary/candidate-ranking problems, which are easier to study than pure semantic misses.

## Interpretation

This experiment supports a more precise version of our hypothesis:

> Hard-negative learning may not directly improve the old start/end baseline, but it can make the evidence landscape more useful for downstream boundary decoding.

So the value is not simply "higher score". The value is that we can now decompose the problem:

1. Semantic evidence learning: make the correct event region stand out against plausible false peaks.
2. Evidence-to-boundary decoding: learn which evidence shapes correspond to accurate temporal boundaries.
3. Error-targeted iteration: mine the model's own wrong high-confidence regions and train against them.

## Next Step

The next version should mine hard negatives iteratively from v2, not only from v1.

Recommended v3 experiment:

1. Use v2 full train evidence to mine a new hard-negative set.
2. Separate negative types:
   - semantic false peak: IoU `< 0.1`
   - near-boundary distractor: `0.1 <= IoU < 0.5`
   - over-wide / under-wide window around GT
3. Train with type-aware ranking losses instead of one generic hard-negative margin.
4. Report not only recall, but also failure migration across categories.

This keeps the project research-focused: we are studying how temporal semantic evidence changes under targeted error pressure.
