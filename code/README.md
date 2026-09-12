# Analysis code

This directory contains selected scripts from the AMR diagnostic history.
They are organized by purpose rather than by the private server directory
layout. Most scripts require benchmark metadata, extracted features, or
checkpoints that are deliberately not included in this public repository.

`analysis/` contains audit and ablation utilities, `failure_audit/` contains
the earliest failure-audit builders, `r27_package/` and `r28_package/` retain
reproduction packages, and `review_ui/` contains the small local review
interface.

`search_space_counterfactual_run.py` reproduces the coordinate-preserving
decoder-access intervention documented in
[`audits/search_space_counterfactual_run/`](../audits/search_space_counterfactual_run/).
It requires explicit paths to the official baseline worktree, checkpoint,
metadata, and reference submission.

Before running a script, read its adjacent experiment card and confirm that
its default paths are replaced with explicit local paths.
