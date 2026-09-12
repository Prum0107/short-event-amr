# Intervention options

| Level | Classification | Assessment |
|---|---|---|
| A. Decoder cross-attention memory masking only | `VALID_COORDINATE_PRESERVING_COUNTERFACTUAL` | Run the full original encoder once. Keep `memory_local` and `pos_embed_local` unchanged, and pass a per-example `memory_key_padding_mask` that retains GT tokens plus a predeclared subset of non-GT tokens. This changes decoder access and attention normalization without recomputing positions. |
| B. Post-encoder memory subset / attention bias | `PARTIALLY_VALID` | A compact subset is valid only if it preserves original memory slots, original positions, and the decoder key mask; physically shortening or reindexing memory changes the decoder positional indexing. An additive bias is not an existing official interface and would add a new intervention semantics. |
| C. Saliency/proposal competition only | `PARTIALLY_VALID` | Restricting post-hoc ranking can quantify a selection bottleneck but does not change model retrieval or decoder representations. It is a secondary descriptive analysis, not a clean model counterfactual for the primary question. |
| D. Encoder-level context restriction | `INVALID` for this interface | The encoder receives the same mask used to construct normalized sine positions and the T2V audio/text attention mask. Changing it alters encoder context and coordinates; cropping resets coordinates, and zero filling creates synthetic audio. |

## Cleanest level

Level A is the scientifically cleanest available intervention. It isolates decoder-level access to the already-computed full-audio memory. Its scope must remain explicit: a positive result supports decoder retrieval competition, not the claim that all encoder-level global competition has been removed.
