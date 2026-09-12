# Mechanism-to-method map

This document is a derivation artifact, not an implementation plan. The input evidence is the matched 1,347-query audit. All observations below are exploratory and do not establish causal attribution.

| Diagnosed evidence | Intervention family | Expected mechanism change | First observable metric |
|---|---|---|---|
| Learned normalized widths are 0.086–0.843, while 0–2 s GT width has median 0.00469 | Width representation: log absolute seconds | The regression variable no longer places a 1 s target near zero because `u=log(w_sec/1 s)`; full audio duration is used only at the interface conversion | Conditional decoder-layer `median abs(log(w/GT))` and final width/GT |
| Initial physical widths can be tens to hundreds of seconds | Initialization geometry: continuous log-scale coverage | Initial references cover the short-to-long physical scale without a hand-selected short/long branch; center initialization remains unchanged | Initial width/GT and first-layer contraction ratio |
| Correctly centered proposals contract but retain 4.36× GT width for 0–2 s | Refinement dynamics: multiplicative/log-width update | A layer predicts a width ratio, so 40 s → 1 s is one log-ratio update rather than a large additive displacement | Layerwise log-width residual and contraction ratio |
| Normalized L1 is weak for short absolute errors, while GIoU is relative | Optimization geometry: relative/log-ratio duration error | The scale term is sensitive to relative error without merely multiplying one scalar coefficient; classification and center terms remain separate | Matched short-event width cost and `abs(log(w/GT))` |
| The same 1 s event is harder in 299 s than in 125 s audio | Audio-length invariance: effective temporal clock | A query-conditioned physical scale `s(a,q)` or physical-time coordinate prevents full-recording duration from setting the event's numerical difficulty | Slope of final width/GT and R1@0.7 versus audio duration within GT bins |
| A different span parameterization might remove query-width contraction altogether | Non-DETR control: dense boundary distributions or point-plus-boundary distances | Start/end are predicted directly on the temporal grid; no learned object-query width or center-width reference is required | Boundary error and duration-conditioned IoU |

## Interpretation boundary

The mapping says what each family would change if implemented. It does not say that any family is effective, novel, or causally sufficient. The first intervention is selected for mechanism isolation, not expected headline performance.
