# Experiment card — log-width identifiability audit

## Problem

The proposed `u=log(w_sec/1 s)` change was selected after a scale audit, but it was not known whether it changes only width representation or also initialization, decoder refinement, attention conditioning, loss gradients, and audio-duration dependence.

## Primary decision

Decide whether `WIDTH_REPRESENTATION_ONLY` is cleanly isolatable and select one narrower first causal pilot if it is not.

## Evidence and scope

Evidence is restricted to the exact remote QD-DETR/M1 source paths, config, and matched checkpoint width statistics already audited. No training, inference, GT-oracle routing, broad method search, or QD-DETR modification is performed in this audit.

## Estimands

1. Whether an absolute log-seconds representation can preserve baseline initialization while removing duration dependence.
2. Whether additive log refinement is merely a coordinate rewrite or a distinct update law.
3. Which gradients change under Variants A, B, and C.
4. The narrowest interpretable first pilot.

## Predeclared analysis

Derive equations from source, compute analytic transitions for `D in {60,120,240,300}` and short-event width pairs, classify each intervention as cleanly isolatable or coupled, then choose one pilot. No result-adaptive window, loss, or parameterization is introduced.

## Decision rule

If a representation requires duration-dependent initialization to preserve the baseline and still feeds normalized losses/attention through `exp(u)/D`, classify it as `PARTIALLY_COUPLED`. Prefer a single-path refinement pilot when it preserves initialization and final geometry.

## Output and limitations

The output is a derivation audit, not a performance result. It establishes what a future intervention would test and what it could not identify. It does not establish that any proposed pilot improves DCASE metrics.
