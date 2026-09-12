# AMR 实验结果总报告（持续追加）

## 使用说明

本文件作为后续每次实验的统一记录入口。历史 R0–R31 的完整实验时间线保存在同目录的 `experiment_timeline.md`；新实验完成后，在本文末尾复制“单次实验记录模板”并填写，不覆盖历史记录。

实验记录必须区分：实际观测结果、解释、限制和下一步决定。没有运行的实验不得填写结果；没有保存的数字不得凭记忆补写。

## 当前研究状态

- 论文初稿：`manuscript_draft.md`（R39）
- 当前修订稿：`manuscript_v2.md`（R44）
- EGCG v1.0 commit：`07d52c8d8fc2508f6899cd43b3f404add24db719`
- QD-DETR baseline commit：`45ef471ee47ea75a2141d75bd9cfdb8c45dfc101`
- 最近完成的实验阶段：R31 final test ablation and efficiency
- 当前没有正在运行的新实验

## 已归档的关键结果快照

| 对比 | R1@0.7 | 备注 |
|---|---:|---|
| QD-DETR overall | 18.93% | CASTELLA official test |
| Full EGCG overall | 26.80% | frozen seed-2023 checkpoints |
| QD-DETR, GT 0–2s | 1.11% | N=90 |
| Full EGCG, GT 0–2s | 21.11% | N=90 |
| QD-DETR, GT 2–5s | 6.91% | N=376 |
| Full EGCG, GT 2–5s | 21.01% | N=376 |
| UVCOM original | 20.12% | R1@0.7 |
| UVCOM full EGCG | 19.97% | mixed cross-model result |

这些数字来自已归档结果，不代表本文件创建时重新运行了实验。

## R31 final test data

### Main comparison

Official CASTELLA test split. `N=1,347`; all values are percentages.

| System | R1@0.5 | R1@0.7 | mAP |
|---|---:|---:|---:|
| QD-DETR | 38.75 | 18.93 | 16.26 |
| Full EGCG | 47.51 | 26.80 | 24.43 |

### Clean A-D ablation

Candidate metrics use IoU 0.5. Oracle and Top1-Oracle gap use IoU 0.7.

| System | R1@0.5 | R1@0.7 | mAP | CandidateRecall@10@0.5 | CandidateRecall@100@0.5 | Oracle@10@0.7 | Top1-Oracle@10@0.7 gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| A QD-DETR | 38.75 | 18.93 | 16.26 | 64.37 | 64.37 | 37.56 | 18.63 |
| B No-evidence ranker | 41.13 | 20.64 | 16.78 | 64.37 | 64.37 | 37.56 | 16.93 |
| C Evidence candidate generation | 40.98 | 21.08 | 20.38 | 76.69 | 83.96 | 54.42 | 33.33 |
| D Full EGCG | 47.51 | 26.80 | 24.43 | 79.29 | 83.96 | 57.31 | 30.51 |

### Short-duration test results

| System | GT bin | N | R1@0.5 | R1@0.7 | mAP | CandidateRecall@10@0.5 | CandidateRecall@100@0.5 | Oracle@10@0.7 | Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A QD-DETR | 0-2s | 90 | 14.44 | 1.11 | 2.34 | 16.67 | 16.67 | 1.11 | 0.00 |
| B No-evidence ranker | 0-2s | 90 | 13.33 | 1.11 | 2.29 | 16.67 | 16.67 | 1.11 | 0.00 |
| C Evidence candidate generation | 0-2s | 90 | 17.78 | 5.56 | 11.62 | 54.44 | 72.22 | 36.67 | 31.11 |
| D Full EGCG | 0-2s | 90 | 35.56 | 21.11 | 26.81 | 67.78 | 72.22 | 46.67 | 25.56 |
| A QD-DETR | 2-5s | 376 | 26.33 | 6.91 | 9.38 | 43.88 | 43.88 | 14.10 | 7.18 |
| B No-evidence ranker | 2-5s | 376 | 26.33 | 7.18 | 9.70 | 43.88 | 43.88 | 14.10 | 6.91 |
| C Evidence candidate generation | 2-5s | 376 | 33.24 | 11.97 | 18.43 | 73.14 | 83.24 | 48.67 | 36.70 |
| D Full EGCG | 2-5s | 376 | 43.88 | 21.01 | 24.11 | 76.33 | 83.24 | 52.13 | 31.12 |

### Cross-model boundary

| Model | System | R1@0.5 | R1@0.7 | mAP |
|---|---|---:|---:|---:|
| QD-DETR | Original | 38.75 | 18.93 | 16.26 |
| QD-DETR | Full EGCG | 47.51 | 26.80 | 24.43 |
| UVCOM | Original | 31.48 | 20.12 | 15.90 |
| UVCOM | Full EGCG | 31.92 | 19.97 | 16.33 |

### Efficiency

| System | Queries | Batch | Runs | Mean latency (s) | Std (s) | Mean ms/query | Peak GPU memory (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| QD-DETR | 1,347 | 64 | 3 | 7.1116 | 0.1139 | 5.2796 | 749,876,224 |
| Full EGCG | 1,347 | 64 | 3 | 11.1116 | 0.2252 | 8.2491 | 750,203,904 |

Archived training time: Evidence Head `50.4959 s`; Ranker `21.9285 s`.

Raw CSV data is preserved in:

- `final_tables/main_results.csv`
- `final_tables/ablation.csv`
- `final_tables/short_duration.csv`
- `final_tables/cross_model.csv`
- `final_tables/efficiency.csv`

## Earlier diagnostic data milestones

| Experiment | Direct data/result | Interpretation boundary |
|---|---|---|
| R1/R2 | MS-CLAP QD-DETR R1@0.7 `10.32`; M2D feature-only QD-DETR R1@0.7 `18.93` | Representation improvement helps but does not remove short-moment degradation. |
| R4 | Short 0-5s Top-1 success `5.79%`; Best-10 success `11.59%`; candidate missing `65.15%`; ranking failure `6.15%` | Candidate availability is a major diagnosed component, not the sole cause. |
| R9 | Overall CandidateRecall@10: `64.37% -> 77.73%`; short 0-5s: `38.63% -> 71.24%`; random control `60.80%`; shuffled evidence `62.88%` | Query-conditioned evidence proposals improve candidate recall in the prototype. |
| R10 | CandidateRecall `64.37% -> 82.78%`, while final R1 falls `38.75% -> 36.75%` under naive augmentation | Candidate availability alone does not guarantee usable ranking. |
| R11 | Evidence-aware reranking recovers part of the final retrieval loss, especially for short moments | Ranking mismatch is supported as a pipeline issue. |
| R15 | CandidateRecall `64.37% -> 76.61%`; 0-2s R1@0.7 remains `1.11%` | Learned evidence recovery alone does not complete final selection. |
| R16 | 0-2s Top1 R1@0.7 `1.11%`; Oracle@10 `42.22%`; Oracle@100 `51.11%` | Good candidates can exist while Top-1 selection remains weak. |
| R18 | Archived validation seeds: `2023`, `2024`, `2025` | Multi-seed validation artifact; not the final seed-2023 test estimate. |
| R27 | Historical R18 Full EGCG R1@0.7 `25.61`; clean R27 seed-2023 `26.80`; baseline reproduced exactly | Difference documented; historical number was not forced. |

完整的 R0–R31 目的、输入、结果和来源见 `experiment_timeline.md`。

## 单次实验记录模板

复制以下区块作为下一条记录，并替换方括号内容。

### [Experiment ID] — [Short title]

**Status:** `[PLANNED / RUNNING / COMPLETE / BLOCKED / STOPPED]`

**Date:** `[YYYY-MM-DD]`

#### 1. Scientific question

[本实验只回答一个明确问题。]

#### 2. Scope and constraints

- Model/code commit: `[commit]`
- Dataset and split: `[dataset / train / validation / test]`
- Features/checkpoints: `[paths and hashes if available]`
- Seed(s): `[seed]`
- Hardware/environment: `[GPU, Python, PyTorch/CUDA if recorded]`
- Not allowed / not performed: `[training, test tuning, raw-audio download, etc.]`

#### 3. Configuration

| Field | Value |
|---|---|
| Audio/text representation | `[value]` |
| Trainable components | `[value]` |
| Frozen components | `[value]` |
| Optimizer and learning rate | `[value]` |
| Epochs / batch size | `[value]` |
| Loss and coefficients | `[value]` |
| Proposal/ranking configuration | `[value]` |
| Evaluation thresholds | `[value]` |

#### 4. Results

| System / condition | Overall | Short-duration | Candidate metric | Oracle/ranking metric |
|---|---:|---:|---:|---:|
| `[baseline]` | `[value]` | `[value]` | `[value]` | `[value]` |
| `[intervention]` | `[value]` | `[value]` | `[value]` | `[value]` |

Always include the metric name and IoU threshold, for example `R1@0.7`, `CandidateRecall@10@0.5`, or `Oracle@10@0.7`. Include the denominator for every duration bin.

#### 5. Direct observations

- `[Only facts directly supported by logs/tables.]`

#### 6. Interpretation

- `[What the results indicate.]`
- `[What they do not establish.]`

#### 7. Failure/quality checks

- Data split leakage check: `[PASS / FAIL / UNKNOWN]`
- Checkpoint/config recorded: `[PASS / FAIL / UNKNOWN]`
- Reproducibility artifacts saved: `[paths]`
- Unexpected behavior: `[value]`

#### 8. Decision

**Decision:** `[GO / NO-GO / INCONCLUSIVE / BLOCKED]`

**Reason:** `[one concise evidence-based sentence]`

**Next action:** `[specific next step or STOP]`

## Reporting rules for future experiments

1. Do not mix validation and test results in one row.
2. Do not report an oracle or candidate metric without its K and IoU threshold.
3. Do not call a proposal-level GT-overlap score semantic evidence.
4. Do not claim causality or generalization from one ablation or one architecture.
5. Save the exact output paths and commit before interpreting results.
6. If a required artifact is missing, write `UNKNOWN` and record the blocker.
7. If the user says stop, preserve the record and do not start the next experiment.

## Next entry

**Experiment ID:** `NEXT`

**Status:** `NOT STARTED`

**Instruction:** Fill this section only after a new experiment is explicitly authorized and its outputs are saved.
