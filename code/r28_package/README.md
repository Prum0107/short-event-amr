# EGCG v1.0

This directory is the frozen, clean implementation recovered from the R14–R18
artifacts. R13 design prose is not used as implementation authority.

The implementation keeps QD-DETR, its audio/text features, backbone, and
localization head fixed. It adds the recovered Evidence Head, R15 evidence
proposal/fusion path, and R17 evidence-aware ranker.

R28 freezes this tree together with the seed-2023 Evidence Head and Ranker
checkpoints. The strict final ablation trains only ranker controls on the
official train split and evaluates only the official validation split.

See `src/README.md` for recovered choices and `configs/seed2023.yaml` for the
explicit configuration.
