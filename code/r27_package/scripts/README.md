# R27 scripts

- `reproduce_seed2023.py` trains the recovered Evidence Head and Ranker on the
  official train split, saves all artifacts, then performs the test-only
  reproduction gate.
- `strict_ablation.py` is run only after `reproduce_seed2023.py` reports
  `R27_REPRODUCTION_PASS`.
