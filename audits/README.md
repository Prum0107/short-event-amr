# Diagnostic audits

Each subdirectory is a self-contained report bundle. The compact CSV/JSON
files are retained so that conclusions can be inspected without access to the
private server.

The most recent completed run is
`audio_duration_mechanism_decomposition`. It is inference-only and uses the
unchanged official baseline. The most recent counterfactual bundle,
`search_space_counterfactual_run`, is a completed decoder-only intervention
with full-access reproduction checks and paired condition tables. Its result
supports a matched high-saliency distractor effect, while the pre-specified
global-search and dose-response tests are not supported.

The preceding `search_space_counterfactual_feasibility` directory records the
validity/design audit that preceded the run.

The `hard_negative_mechanism_audit` directory is the completed follow-up. It
uses frozen harmful/control cohorts, official QD-DETR decoder traces, a bounded
token-level replacement probe, and native MS-CLAP comparisons where original
WAVs are available. It supports a decoder-reference-attraction component but
does not resolve repeated or incompletely annotated occurrences. Missing raw
audio is recorded as BLOCKED.

The `decoder_hard_negative_necessity_audit` directory is the completed
inference-only removal and layer audit. It masks the pre-defined non-GT
high-saliency hard region against matched random valid non-GT removals and
records same-run official-path validation. The mean harmful-cohort effect is
positive, but concentrated in a minority of queries; reference redirection is
not supported by the pre-specified criterion, so the scientific decision is
`INCONCLUSIVE`. Location-rescued queries retain substantial width error.

The `short_span_scale_construction_audit` directory is the follow-up frozen
baseline attribution study for that residual scale failure. It conditions on
final center error at most 1 s, traces width construction through both decoder
layers, reproduces TRAIN Hungarian matching, evaluates offline matching-cost
counterfactuals, and computes final-checkpoint width gradients. The audit
supports several coupled descriptive factors but does not identify a single
causal mechanism; its method-design gate is `NO`.

The full native local MS-CLAP audit is not represented by fabricated output
here. The hard-negative bundle contains only the native comparisons supported
by the raw WAV coverage on the server; it records the exact model/API
provenance and query-level BLOCKED rows for missing audio.
