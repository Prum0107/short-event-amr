# Provenance and artifact policy

The experiments were run on a private research server. This public repository
records identifiers and hashes where they are useful for verification, but it
does not publish private connection details or large runtime artifacts.

## Official baseline

- Baseline source commit: `45ef471ee47ea75a2141d75bd9cfdb8c45dfc101`
- Checkpoint filename: `best_checkpoint.pth`
- Checkpoint SHA-256:
  `9cdc18a14e906689484f1dde055b42cdcc4b77f0d850f6a0174ff9ef42063d35`
- Configuration SHA-256:
  `195a41b47042bb9a6456e1268ccbcc9ef1a25862bb66508f1427085044aeaaab`
- Frozen test metadata SHA-256:
  `044f141630d4daff984f1bfce1622071520edc27e5f7e49930571c761e6fcaa4`
- QD-DETR source SHA-256:
  `2b87c7537db88a731f3a0f485915d6a044d065b2bad6d34df8844ce7c2af0738`
- Transformer source SHA-256:
  `817c6bc0a678e665b38613e181f09ae6f9d23866660a3c3be9d0f64a847f8d79`
- Dataset source SHA-256:
  `bf08379a3473ce0cac215a34665ef548748987b48e742bb2ce2fcae0e6dc457b`

The original checkpoint, feature tensors, raw audio, and full metadata splits
remain off-repository. A future release can add download instructions only
after redistribution rights are checked.

## Archived implementation

`legacy_tsel/` is a cleaned copy of the earlier DCASE26 Task 6 research code
archive. Its original MIT notice is retained. It is kept for continuity and
does not imply that every archived proposal is the current research direction.

## Latest audit provenance

The duration-mechanism audit used the official baseline, inference-only
execution, 1,347 queries, one-second clip length, random seed `20260912`, and
CUDA. Its complete compact outputs are under
[`audits/audio_duration_mechanism_decomposition/`](../audits/audio_duration_mechanism_decomposition/).

The search-space counterfactual audit is a design/validity document only. It
has `executed: false` and must not be described as an intervention result.
