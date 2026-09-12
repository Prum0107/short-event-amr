# Selection-bias assessment

The HARD_25 set is selected by the QD-DETR full-access saliency score. Therefore saliency magnitude, hard-token rank, and decoder quantities measured after that selection are selection-conditioned evidence. They can describe what the model selected, but cannot independently establish why the selected region is semantically hard.

The native MS-CLAP comparison is the independent evidence channel: it uses the original raw WAV, a matched absolute window duration, the query's native MS-CLAP text embedding, and the official shared embedding similarity. It is still unavailable for rows without original WAVs, which remain BLOCKED.

- Geometry rows: 466; native valid rows: 54; native blocked rows: 412.
- No stored QD-DETR audio/text tensor cosine was computed.
- No human label was auto-filled. Repeated or incompletely annotated occurrences remain unresolved until review labels are supplied.

Interpretation must therefore separate: (a) selection-conditioned QD behavior, (b) independent native semantic evidence, and (c) unobserved human annotation status.
