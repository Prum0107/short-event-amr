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

## Current gate

P4A native local MS-CLAP evidence validation is the active measurement gate.
It begins with a deterministic 30-query subset (10 queries in each of the
0–2 s, 2–5 s, and ≥10 s regimes). The full 1,347-query native audit requires a
successful validation report and explicit approval.

## Next diagnostic decision

After the active gate is reviewed, choose exactly one branch:

- If native local evidence is strong, test whether it is lost when converted
  into the long-audio temporal representation or consumed by QD-DETR.
- If native local evidence is weak, investigate audio representation and
  semantic recognition limitations.
- If evidence is strong only at short context, investigate temporal evidence
  dilution and scale sensitivity.

No next-stage intervention or AMR method is implemented by the current archive.
