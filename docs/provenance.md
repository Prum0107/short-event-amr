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

The coordinate-preserving search-space counterfactual audit is complete. Its
full-access reproduction passed and its matched HARD_25 condition reduced
CenterHit relative to RANDOM_25 in the two primary duration bins; the result
supports a high-saliency distractor effect, not a generic global-search law.

## Hard-negative mechanism audit

- Harmful cohort rule: HARD_25 CenterHit@10≤2s lower than the mean of ten
  deterministic RANDOM_25 replicates.
- Cohort sizes: 44 harmful / 46 control in 0–2 s; 127 harmful / 249 control in
  2–5 s.
- Native MS-CLAP package: `msclap==1.3.4`, implementation
  `microsoft/CLAP`, HTSAT/GPT-2 configuration with shared projected dimension
  1024.
- Native MS-CLAP checkpoint SHA-256:
  `2cef4016d47d00eb28d153d522f397222057f95000e9bad6b9583c631284a1e6`
- The native comparison uses equal-duration raw-audio windows and the official
  normalized projected-vector dot product multiplied by `logit_scale.exp()`.
  Missing WAVs are retained as BLOCKED; QD-DETR stored audio/text features are
  never compared by cosine.
- Compact outputs are under
  [`audits/hard_negative_mechanism_audit/`](../audits/hard_negative_mechanism_audit/).
