# Validity assessment

## Decision

`CLEAN_COUNTERFACTUAL_FEASIBLE`

## What remains fixed under Level A

- original full feature sequence, original duration, and original TEF coordinates;
- original `PositionEmbeddingSine` output and original full-sequence encoder masks;
- full T2V encoder and second encoder execution;
- original `memory_local` vectors and their original position embeddings;
- model parameters, text features, number of model queries, and initial reference points;
- normalized-to-seconds output conversion.

## What the intervention changes

- the set of decoder cross-attention keys allowed to receive probability mass;
- the cross-attention softmax denominator and therefore attention weights;
- decoder hidden states, later reference-point updates, class logits, and span outputs as downstream responses;
- the number of competing decoder-accessible temporal positions.

These changes are the intended retrieval-competition intervention and must be reported as side effects, not hidden as if only a scalar candidate count changed.

## Validity checks required before any future run

1. Run the full forward up to `memory_local` once and verify bitwise/numerically identical memory and positional tensors across conditions.
2. Verify that only `memory_key_padding_mask` differs at decoder entry; initial references and query features must be identical.
3. Keep every GT-associated valid token available so no decoder condition has an all-masked key set.
4. Confirm that original padding remains masked and excluded distractors are only additional masked positions.
5. Confirm unchanged duration conversion and unchanged query count.
6. Stop if implementation requires changing `src_aud_mask`, recomputing positions, compacting memory, adding synthetic values, or changing encoder attention.

## Interpretation boundary

This is clean for the causal question “does decoder access to more full-coordinate temporal memory positions change retrieval?” It is not a full-system intervention on global search space because the encoder still saw every original valid position.
