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

The native local MS-CLAP validation requested for the next stage is not
represented by fabricated output here. Its validation subset, model/API
provenance, and results will be added only after the exact native shared
embedding pathway is verified and the 30-query gate is actually run.
