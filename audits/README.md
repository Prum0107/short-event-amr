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

The full native local MS-CLAP audit is not represented by fabricated output
here. The hard-negative bundle contains only the native comparisons supported
by the raw WAV coverage on the server; it records the exact model/API
provenance and query-level BLOCKED rows for missing audio.
