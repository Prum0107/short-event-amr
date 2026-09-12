# Positional and mask dependency trace

## Coordinate construction

1. `src/dataset.py:77-89` loads the stored audio feature sequence, truncates it to `max_a_l`, L2-normalizes it through `_get_audio_feat_by_vid`, and appends TEF coordinates when `ctx_mode=audio_tef`. The TEF values are `i/ctx_l` and `(i+1)/ctx_l`; they are constructed from the original feature sequence length.
2. `src/qd_detr.py:106-114` projects audio and text inputs, concatenates their masks, and calls `self.position_embed(src_aud, src_aud_mask)` before the transformer. The model therefore constructs audio positional embeddings before decoder retrieval.
3. `src/position_encoding.py:47-67` implements `PositionEmbeddingSine`. With valid-audio mask value 1, it uses `mask.cumsum(1)` and, because the baseline builds it with `normalize=True` at line 101, divides by the final cumulative valid position. This is the full-sequence normalized temporal coordinate system.

## Encoder and memory

- `src/qd_detr.py:125-127` sets `audio_length=src_aud.shape[1]` and passes the original concatenated mask and position embedding to the transformer.
- `src/qd_detr_transformer.py:98-121` first runs the T2V encoder, then keeps the global token plus the audio segment for the second encoder. The second encoder receives the original padding mask and original positional embedding.
- `src/qd_detr_transformer.py:121-132` creates `memory_local`, `memory_global`, `mask_local`, and local positional embeddings. `memory_local` contains the original full-audio audio memory positions; padded positions are represented by `mask_local`.

## Decoder retrieval

- `src/qd_detr_transformer.py:126-128` calls the decoder with `memory_local`, `pos_embed_local`, and `memory_key_padding_mask=mask_local`; the baseline does not pass a `memory_mask`.
- `src/qd_detr_transformer.py:221-270` forwards `memory_mask` and `memory_key_padding_mask` through every decoder layer.
- `src/qd_detr_transformer.py:602-635` forms decoder query/key/value projections and passes both masks to `self.cross_attn`. A per-example `memory_key_padding_mask` can therefore exclude selected audio memory positions after the original encoder and positional embedding have been computed.

## Reference points and outputs

- `src/qd_detr_transformer.py:231-283` initializes reference points from the unchanged learned `query_embed`, then updates them after each decoder layer. An access mask leaves the initial reference points fixed but changes later decoder states and hence later reference updates.
- `src/qd_detr.py:127-134` converts decoder states and references into normalized center-width predictions. `src/evaluate.py:compute_mr_results` performs the unchanged conversion to seconds; no output-coordinate conversion needs to change.

## Saliency path

`src/qd_detr.py:136-154` computes `saliency_scores` from the encoder audio memory and global memory, not from decoder cross-attention output. A decoder-only access intervention therefore does not change the native full-encoder saliency map; saliency is a fixed diagnostic covariate for selecting hard distractors, not the counterfactual outcome.

## Consequence

The existing decoder API exposes a coordinate-preserving intervention point: retain the full original `memory_local` and `pos_embed_local`, and replace only `memory_key_padding_mask` with an original-padding mask plus selected distractor exclusions. The resulting counterfactual is clean for decoder-level temporal competition, while deliberately not identifying competition that already occurred inside the full encoder.
