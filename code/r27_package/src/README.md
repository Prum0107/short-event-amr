# Clean EGCG reproduction

This implementation is reconstructed from the recovered R14–R18 source files
under `/private/research-artifact`. R13 design prose is not used as an
implementation authority.

The implementation preserves the historical choices:

- R14 Evidence Head: `256 -> 128` audio/query projections, fusion
  `[audio, query, audio*query]`, `384 -> 128 -> 1` with ReLU.
- Evidence targets: one-second token has a positive label when it overlaps any
  ground-truth interval; valid-token masked BCE-with-logits; positive weight 5.
- R15 proposals: query-wise 90th-percentile active tokens, contiguous regions,
  one-second token mapping, max pool 100, exact `(start, end)` deduplication.
- R15 fusion: `0.5 * QD_score + 0.5 * evidence_score`.
- R17 Ranker: mean candidate audio feature, mean query feature, QD score,
  evidence score, normalized start/end/duration; pairwise hinge, margin 0.1.

The recovered QD-DETR model and source implementation remain unchanged.
