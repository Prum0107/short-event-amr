# Long-event safety

| Family | Expected 20 s+ effect | Shared formulation? | Main risk |
|---|---|---|---|
| Log absolute-seconds width | Should preserve long physical widths and reduce only coordinate compression; no hard short/long branch | Yes | Exponential decoding can overreact to noisy residuals; log-scale prior is common |
| Continuous log-scale initialization | Can cover long widths if the scale range is broad and continuous | Yes, but coverage budget is finite | Short-scale coverage may displace long-scale coverage; generic multi-scale prior risk |
| Multiplicative refinement | Natural for both short and long ratios; no explicit threshold | Yes | Large positive residuals can expand long proposals; requires bounded/stable updates |
| Relative/log-ratio optimization | Treats equal ratios equally across durations | Yes | Short events can dominate gradients because relative error is larger; long events may lose absolute-boundary precision |
| Effective temporal clock | Can preserve long events if `s(a,q)` grows with event extent continuously | Yes | A learned scale estimator may collapse to short-event behavior or silently encode duration bins |
| Dense boundary/non-DETR control | Direct boundary modeling can represent 20 s+ intervals without query-width contraction | Yes in principle | Changes architecture and supervision; dense endpoint ambiguity and prior-art overlap |

The safest first intervention is the one that changes a single width coordinate while retaining the existing center, semantic inputs, decoder count, matching, and GIoU. Long-event safety must be measured, not inferred from the formula.
