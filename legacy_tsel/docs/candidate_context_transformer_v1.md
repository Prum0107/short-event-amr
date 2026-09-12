# Candidate Context Transformer Decoder v1

## Question

The MLP learned decoder scores candidates independently.

This experiment asks:

> Does modeling candidate-to-candidate interaction improve audio moment retrieval?

The motivation is clear:

- top2 candidates work better than top10 candidates
- source-gated candidates improve strict top1 but hurt semantic coverage
- two-mode oracle shows a much higher upper bound

So candidate relations may matter.

## Model

Script:

`src/train_candidate_transformer_decoder.py`

Architecture:

- input: candidate feature set for one query
- feature version: `shape_v2`
- 2-layer Transformer encoder
- hidden size 128
- 4 attention heads
- dropout 0.1
- output: one score per candidate

Training objective:

- pointwise IoU regression
- pairwise ranking loss within the same query

This is a small task-specific Transformer, not a GPT/Qwen-style language model.

## Experiments

Input evidence:

`results/evidence_baseline_v31_semantic_only_lsem025`

Candidate settings:

1. `shape_v2 top2`
2. `source_gate_v2`

Reports:

- `results/candidate_transformer_shape_v2_top2_v1/report.html`
- `results/candidate_transformer_source_gate_v2_v1/report.html`

## Results

| Decoder | Candidate Setting | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 | Semantic Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | shape_v2 top2 | 34.94% | 25.85% | 38.35% | 43.18% | 0.3316 | 0.4959 | 77 |
| Transformer | shape_v2 top2 | 34.38% | 25.57% | 35.80% | 43.75% | 0.3293 | 0.4982 | 98 |
| MLP | source_gate_v2 | 37.22% | 26.42% | 39.49% | 43.75% | 0.3425 | 0.4905 | 102 |
| Transformer | source_gate_v2 | 34.66% | 25.00% | 38.92% | 44.89% | 0.3212 | 0.5080 | 100 |

## Key Finding

The Transformer does not improve strict top1.

But it often improves top-k candidate quality:

- top2 setting: best IoU@5 improves from 0.4959 to 0.4982
- source_gate setting: best IoU@5 improves from 0.4905 to 0.5080
- source_gate setting: R5@0.7 improves from 43.75% to 44.89%

So the Transformer behaves like a stronger candidate-set explorer, not a stronger final top1 selector.

## Training Dynamics

Both Transformer runs overfit quickly.

For `shape_v2 top2`:

- best epoch: 2
- after that, training loss keeps falling while validation R1 drops

For `source_gate_v2`:

- best epoch: 1
- later epochs also degrade validation R1

Interpretation:

The model capacity is already high relative to the number of training queries. Candidate interaction is learnable, but the current supervision is not enough to make it stable for top1 selection.

## Migration: MLP Top2 to Transformer Top2

Report:

`results/decoder_migration_mlp_top2_to_transformer_top2/report.html`

| Metric | Value |
|---|---:|
| Stable samples | 233 |
| Category improved | 33 |
| Category regressed | 54 |
| IoU improved | 18 |
| IoU regressed | 14 |
| Semantic misses recovered | 4 |
| Good regressed | 20 |
| Avg top1 IoU delta | -0.0023 |
| Avg top5 IoU delta | +0.0023 |

Important transition:

- `boundary_error -> semantic_miss`: 24
- `candidate_exists -> good`: 18

This explains the mixed behavior: the Transformer improves some candidate-exists cases, but it also turns many boundary errors into semantic misses.

## Migration: MLP SourceGate to Transformer SourceGate

Report:

`results/decoder_migration_mlp_source_gate_to_transformer_source_gate/report.html`

| Metric | Value |
|---|---:|
| Stable samples | 262 |
| Category improved | 35 |
| Category regressed | 33 |
| IoU improved | 8 |
| IoU regressed | 14 |
| Semantic misses recovered | 12 |
| Good regressed | 16 |
| Avg top1 IoU delta | -0.0212 |
| Avg top5 IoU delta | +0.0175 |

Important transition:

- `semantic_miss -> boundary_error`: 10
- `candidate_exists -> good`: 10
- `boundary_error -> semantic_miss`: 10

Again, the Transformer increases top-k quality while making top1 less stable.

## Feature Ceiling Diagnosis

Report:

`results/feature_ceiling_diagnosis_transformer_source_gate_v1/report.html`

For Transformer source_gate:

| Semantic Miss Type | Count |
|---|---:|
| representation ceiling | 48 |
| decoder/candidate ceiling | 52 |

This is close to MLP source_gate, but worse than coverage top2 and two-mode selector.

## Research Conclusion

This is a useful negative result.

The answer is not:

> Transformer is better than MLP.

The more accurate conclusion is:

> Candidate interaction helps expose better top-k candidates, but the current Transformer is not a stable top1 selector under limited supervision.

This means we should not simply replace the MLP with a Transformer.

## Next Direction

The Transformer may still be useful, but in a different role:

### Option A: Transformer as Candidate Expander

Use it to improve top-k candidate quality, then let a conservative selector choose top1.

### Option B: Distill Transformer Top-K Into MLP/Selector

Use Transformer top-k improvements as additional signals for the two-mode selector.

### Option C: Regularized / Pairwise-Only Transformer

Try a smaller Transformer or stronger regularization:

- hidden size 64
- 1 encoder layer
- higher dropout
- fewer epochs
- stronger early stopping
- pairwise loss only or lower pointwise regression weight

The immediate recommendation:

> Keep MLP as the primary top1 decoder, and use Transformer results as evidence that top-k candidate context is useful but needs better supervision.

This keeps the research story coherent:

`semantic evidence -> candidate policy -> top-k context modeling -> query-adaptive final selection`
