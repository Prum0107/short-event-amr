# Short-span scale construction attribution audit

Current gate: frozen baseline attribution (inference, matching, offline cost, and final-checkpoint gradient readout).

Primary decision: determine which measured factors explain final width error conditional on final center error ≤1 s.

Primary treatment/controls: no model treatment. The analysis conditions on well-centered frozen baseline proposals and uses descriptive comparisons across GT duration, query slot, audio duration, decoder layer, official matching costs, and final-checkpoint width gradients.

Fixed contract: no training, no parameter update, no decoder/loss/matcher/width change, no threshold tuning after results, and no method design.

Primary conclusion ceiling: descriptive attribution only. No factor is called causal without a future controlled intervention.
