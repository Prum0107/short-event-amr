# Decoder hard-negative necessity intervention validation

Overall validation: **PASS**

The frozen encoder, full-audio positional coordinates, duration, query embeddings, initial references, decoder parameters, span update equations, output heads, and postprocessing were kept unchanged. Only decoder memory-key padding masks were changed.

## Checks

- FULL_ACCESS reproduction against the official model forward in the same process: **PASS**; maximum serialized difference `0.0`.
- Comparison with the saved baseline submission: **MISMATCH**; maximum serialized difference `292.0`. A mismatch is retained as provenance because the saved file may have been produced under a different CUDA/runtime stack; it does not override the same-run official-path check.
- Mask contract: **PASS**.
- Finite tensors: **PASS**.
- Layer wrapper execution: **PASS**.
- Decoder layers: `2`; model queries: `10`; batches: `59`.

REMOVE_HARD removes only the frozen strongest hard-region tokens from all decoder layers. REMOVE_RANDOM removes the identical token count from valid non-GT tokens, sampling the same broad temporal-distance stratum when enough candidates exist and recording any fallback.

Layer-specific conditions override the decoder layer's memory mask only at the named layer; all other layers receive FULL_ACCESS. No layer-specific condition was run if the wrapper could not be installed without changing the remaining decoder computation.
