# Initial-width probe validity

This was an inference-only audit using the verified official baseline checkpoint. No model, decoder, loss, matcher, or postprocessing code was changed.

- Predeclared alpha values: `0.5, 0.75, 1.0, 1.5`.
- Baseline learned initial normalized-width support: `[0.074138753, 0.842943430]` across the 10 query references.
- Reference support is defined as the empirical min/max of the baseline learned query references; it is not treated as a complete training-distribution estimate.
- Probe classification: **DESCRIPTIVE_OOD_PROBE_ONLY**.

| alpha | perturbed width range | outside reference support | clipping frequency | finite outputs |
|---:|---:|---:|---:|:---:|
| 0.5 | [0.037069377, 0.421471715] | 10.0% | 0.0% | True |
| 0.75 | [0.055604063, 0.632207572] | 10.0% | 0.0% | True |
| 1.0 | [0.074138753, 0.842943430] | 0.0% | 0.0% | True |
| 1.5 | [0.111208126, 0.999000013] | 40.0% | 30.0% | True |

The alpha sweep was selected from the original reference range before test inference. Values outside the learned reference support make the affected comparisons descriptive sensitivity evidence rather than an in-distribution causal claim.
