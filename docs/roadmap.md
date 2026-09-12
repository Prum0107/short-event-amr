# Research roadmap

## Completed or archived

- R0–R31: baseline, evidence, candidate, ranking, cross-model, and short-event
  diagnostic work preserved in [`paper/experiment_timeline.md`](../paper/experiment_timeline.md).
- Audio-duration mechanism decomposition: completed, inference-only, 1,347
  queries.
- Coordinate-preserving search-space counterfactual: completed with the
  registered checkpoint and frozen full-audio coordinates. The result supports
  a high-saliency distractor effect but not a general global-search or
  monotonic dose-response explanation.
- GT-vs-hard-negative mechanism audit: completed as an inference-only follow-up
  over the predeclared harmful/control cohorts. Decoder-reference attraction is
  supported, near-GT geometry is partial, native semantic competition is not
  dominant on the covered WAV subset, and annotation ambiguity remains
  unresolved without human labels.
- Decoder hard-negative necessity and layer audit: completed with matched
  removal controls and same-run official-path validation. The harmful cohort
  shows a positive average HARD-vs-RANDOM removal effect, but query-level
  heterogeneity and weak reference redirection leave the causal decision
  inconclusive; rescued queries retain substantial width error.
- Short-span scale construction attribution: completed with same-run
  official-path validation. Conditional on center error at most 1 s, the audit
  finds weak short-duration scale response, persistent initial query-slot
  scales, a large first-layer width contraction followed by limited final-layer
  correction, matching scale disadvantage, coordinate compression, and a
  residual audio-duration association. The signals remain coupled and
  descriptive; the method-design gate is `NO`.

## Current gate

The short-span scale construction attribution audit is complete. No trained
intervention or AMR method is authorized by its result. The next branch is a
separate experiment-card decision for the coupled scale factors; it must not
be treated as a method design.

The separately registered P4A native local MS-CLAP validation remains a
measurement gate and has not been replaced by this scale audit; its full
1,347-query run still requires explicit approval.

## Next diagnostic decision

The selected next scientific branch is `MULTIPLE_COUPLED_SCALE_FACTORS`.
Before any intervention, require a new experiment card that separates the
remaining explanations with an explicit falsifiable comparison.

No next-stage intervention or AMR method is implemented by the current archive.

The decoder-competition branch is not promoted to method design. The recorded
next branch for the completed necessity audit is `SHORT_SPAN_SCALE_CONSTRUCTION`,
subject to a separate experiment card and review.
