# Context-length probe validity

## Status: BLOCKED

The official model receives `src_aud_mask` and computes normalized sine positions with `mask.cumsum(1) / mask.cumsum(1)[:, -1:]`. The same mask also determines the transformer padding mask. Consequently, a local or medium context made by changing the mask changes the position coordinate system, including the retained event's normalized positions. Cropping the feature tensor resets positions, and filling removed context with zeros creates synthetic audio. Neither is a same-event context-only counterfactual under the stated validity criterion.

No context-length probe was run and no counterfactual numbers are reported.
