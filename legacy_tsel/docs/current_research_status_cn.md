# 当前研究状态简记

## 终极目标

我们的目标不是单纯冲 leaderboard，而是把 DCASE26 Task 6 做成一个清楚的研究故事：

```text
从“听起来像”推进到“在时间线上有证据地定位”。
```

核心命题：

```text
音频-文本时刻检索不应该只做全局语义匹配，
还要显式学习 query-guided temporal semantic evidence：
什么时候语义证据是真的，什么时候只是 semantic false peak，
以及边界为什么应该收缩或扩张。
```

论文口径的总框架命名为：

```text
Temporal Semantic Evidence Learning (TSEL)
```

工程里的 V2/V4 编号只用于复现实验，不作为论文主命名。当前推荐映射：

| 论文名 | 工程名 | 含义 |
|---|---|---|
| MS-CLAP Evidence Baseline (MS-EB) | MS-CLAP shape_v2 top2 | 基础语义证据分支 |
| Temporal-Semantic Adapter (TSA) | T-CLAP-inspired temporal adapter | 在 MS-CLAP 特征上学习时序语义证据 |
| Semantic False-Peak Mining (SFP) | semantic-only hard negatives | 建模语义假峰 |
| Boundary-Width Evidence Learning (BWEL) | width-only hard negatives | 建模过宽/过窄边界 |
| Semantic-to-Boundary Evidence Curriculum (SBEC) | Temporal Evidence HN V4 semantic->width | 先语义假峰 warmup，再边界宽度修正 |
| Evidence Candidate Fusion (ECF) | V4 candidate-level fusion | 候选级证据融合 |
| Risk-Aware Evidence Scoring (RAES) | Semantic-Temporal Candidate Scorer V2 quality_guard | semantic/temporal/risk heads 的可解释候选选择 |
| TSEL-ECF | V4 fusion branch | 强融合系统 |
| TSEL-RAES | V2 scorer branch | 可解释 evidence-selection 系统 |

最终方法主线暂定为：

```text
MS-EB
-> TSA
-> SFP + BWEL
-> SBEC
-> TSEL-ECF / TSEL-RAES
```

## 现在做到什么程度

项目已经从“探索方向”进入“可以定型成研究结果”的阶段。

已经成立的结论：

- MS-CLAP 有语义能力，但 temporal evidence 比较粗。
- T-CLAP-inspired adapter 明显改善 semantic coverage。
- hard negative 不是越混越好，收益主要来自可解释的错误类型。
- semantic false peaks 适合做 warmup，width negatives 适合做严格边界修正。
- SBEC 是当前最强 evidence curriculum。
- ECF 证明 candidate-level fusion 比 whole-prediction selection 更有效。
- merged candidate oracle 很高，说明主要瓶颈不是候选不存在，而是候选选择不够稳。
- 当前 fusion 的收益可以拆成 semantic recovery 和 temporal correction/refinement；风险可以拆成 anchor regression/weakening 和 semantic/temporal regression。
- RAES 已经把 semantic/temporal/risk 头融进 candidate scorer 本身，形成了第一个“可选择、可解释”的框架盒子。

## 当前关键结果

Validation:

| System | R1@0.7 | Notes |
|---|---:|---|
| SBEC adapter | 30.11 | 当前最强 adapter evidence |
| TSEL-ECF best seed | 33.24 | 当前 validation best |
| TSEL-ECF 5-seed mean | 32.22 | 已经比旧 Fusion V1 稳定更高 |
| TSEL-RAES | 32.27+-0.93 | 5-seed mean，基本追平 TSEL-ECF，同时显式输出 evidence heads |

Frozen test:

| System | R1@0.5 | R1@0.7 | mAP(avg) | mAP@0.5 | mAP@0.75 | Notes |
|---|---:|---:|---:|---:|---:|---|
| MS-CLAP shape_v2 top2 | 23.83 | 16.11 | 12.77 | 20.20 | 12.91 | fixed MS branch |
| TSA | 33.04 | 21.97 | 17.37 | 26.62 | 17.67 | temporal evidence transfers |
| TSEL-ECF best seed | 34.82 | 24.28 | 19.11 | 28.55 | 19.65 | best checkpoint |
| TSEL-ECF 5-seed mean | 34.65+-0.45 | 23.42+-0.60 | 18.66+-0.31 | 28.15+-0.41 | 18.98+-0.35 | official-style mean+-std |
| TSEL-RAES 5-seed mean | 35.13+-0.24 | 23.59+-0.28 | 18.53+-0.08 | 28.06+-0.12 | 18.71+-0.19 | interpretable semantic/temporal/risk scorer |

Against official-style baselines from the released score statistics:

| Baseline | R1@0.5 | R1@0.7 | mAP(avg) | mAP@0.5 | mAP@0.75 |
|---|---:|---:|---:|---:|---:|
| Only CASTELLA | 22.74+-0.77 | 10.17+-0.86 | 10.49+-0.53 | 21.93+-0.57 | 8.85+-0.58 |
| Clotho-Moment pretrain + CASTELLA finetune | 25.86+-0.74 | 13.85+-1.47 | 11.74+-0.39 | 23.14+-0.33 | 10.54+-0.54 |
| Ours: TSEL-ECF | 34.65+-0.45 | 23.42+-0.60 | 18.66+-0.31 | 28.15+-0.41 | 18.98+-0.35 |
| Ours: TSEL-RAES | 35.13+-0.24 | 23.59+-0.28 | 18.53+-0.08 | 28.06+-0.12 | 18.71+-0.19 |

相对 stronger baseline 的 5-seed mean 提升：

```text
R1@0.5  +8.79
R1@0.7  +9.57
mAP(avg) +6.92
mAP@0.5  +5.01
mAP@0.75 +8.44
```

这说明当前结果已经不是 single checkpoint 的偶然收益，可以作为主结果雏形。

## 还需要努力的地方

最优先的两个可比性问题已经完成：

```text
1. evaluator 已补 mAP(avg) / mAP@0.5 / mAP@0.75
2. TSEL-ECF 已完成 5-seed frozen test
3. TSEL-ECF best checkpoint 已完成 test fusion case analysis
4. semantic-temporal role analysis 已落盘
```

接下来更重要的是把 strong experiment 变成 publishable story：

1. 收敛方法描述
   最终主方法要保持干净：TSA + SBEC + ECF/RAES。其他 gate/fusion/oracle 实验作为 ablation 和 analysis。

2. 深化 failure analysis
   test 上 fusion 的 best checkpoint 记录是 `577 improved / 283 regressed`。初步分析显示，主要收益来自 `semantic_miss` recovery、`candidate_exists -> good`、以及 decode-level repair；主要损失来自 `good -> candidate_exists` 和部分 boundary regressions。

   当前 semantic-temporal 拆解：

   ```text
   semantic_recovery = 170
   temporal_positive = 360
   anchor_regression = 57
   anchor_weakening = 38
   semantic_regression = 62
   temporal_regression = 126
   ```

3. 做 distribution-robust candidate selection
   validation gate 在 test 上没有泛化。说明问题不是简单调 gate 阈值，而是需要更稳的 candidate selection 训练目标。一个 val-only source/confidence rule probe 显示，简单屏蔽弱候选源并不能超过 full fusion。下一版 selector 应该显式学习 semantic recovery、temporal correction、anchor risk，而不是只调 source threshold。

   Semantic-Temporal Candidate Selector V1 已经验证：

   ```text
   validation best R1@0.7 = 32.95
   test balanced R1@0.7 = 22.49
   test full fusion R1@0.7 = 24.28
   ```

   它让干预更干净，但过于保守，不能替代 full fusion。结论是：显式 semantic/temporal/risk 头是对的，但不能只作为后置 gate；下一版应该把这些头融进 candidate scorer 本身。

   RAES 已经完成第一版：

   ```text
   hybrid evidence score: validation R1@0.7 = 31.53
   hybrid + anchor guard: validation R1@0.7 = 31.53, regressions 110 -> 101
   RAES quality-guard scoring: validation R1@0.7 = 32.39
   RAES quality-guard scoring 5-seed: validation R1@0.7 = 32.27+-0.93
   ```

   frozen test 也已经完成：

   ```text
   TSEL-RAES 5-seed test R1@0.7 = 23.59+-0.28
   TSEL-ECF 5-seed test R1@0.7 = 23.42+-0.60
   TSEL-RAES test mAP(avg) = 18.53+-0.08
   TSEL-ECF test mAP(avg) = 18.66+-0.31
   ```

   当前结论是：TSEL-RAES 已经把框架搭起来了，并在 strict R1@0.7 上稳定追平/略高于 TSEL-ECF，同时减少 good-case regression；但 mAP(avg) 和 mAP@0.75 略低，所以它现在更适合作为“可解释 evidence-selection framework”，而不是简单宣称全指标替代 TSEL-ECF。

   TSEL-RAES vs TSEL-ECF 的 test-side case analysis 也已经完成。代表性 seed 2027 显示：

   ```text
   TSEL-RAES avoids TSEL-ECF regressions = 90
   TSEL-RAES introduces new regressions = 67
   TSEL-RAES protects anchors regressed by TSEL-ECF = 23
   TSEL-RAES introduces anchor risk not present in TSEL-ECF = 9
   TSEL-ECF good regressed = 57
   TSEL-RAES good regressed = 43
   ```

   这说明 RAES 的贡献不是“发现更多候选”，而是更明确地学习了什么时候不要破坏可靠 anchor。它牺牲了一些 temporal-positive intervention，但风险侧明显更干净。

4. 准备论文级证据
   需要整理 evidence curve、semantic false peak、over-wide/under-wide、fusion 改对/改错、oracle gap 等可视化案例。

论文资产已开始集中在：

```text
docs/paper_assets/
```

## 下一阶段行动顺序

短期不要换方向，按下面顺序推进：

```text
1. 生成 official baseline 对比表和最终结果表
2. 从 TSEL-RAES/TSEL-ECF 对比中挑 paper cases：semantic recovery、temporal correction、anchor protection、remaining temporal-risk failure
3. 整理 ablation 和可视化
4. 准备写作主线
```

当前判断：

```text
研究假设：已经成立
主结果：5-seed test 已经固化，明显高于 released baselines
机制分析：已有骨架，还需要论文级整理
工程系统：能跑通，但脚本和表格还要进一步收敛
下一阶段目标：从 strong experiment 变成 publishable story
```
