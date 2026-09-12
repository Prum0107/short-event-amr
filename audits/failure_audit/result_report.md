# FAILURE AUDIT v0A 结果报告

## 1. 决策

本实验试图判断官方 QD-DETR prediction 的 Top-1 失败主要表现为候选排序差异、时间几何定位误差，还是可观测的多 GT、音频时长、GT 长度和 confidence 分组差异，并生成后续人工听审候选。实验不做语义失败归因。

## 2. 实际执行

- 当前 Gate：S / M diagnostic。
- 输入：固定 commit `45ef471ee47ea75a2141d75bd9cfdb8c45dfc101` 的 `castella_test_release.jsonl` 和官方 `submission.jsonl`。
- 样本量：1347 queries；固定 seed `20260820`；不使用 GPU，不运行模型。
- 执行内容：严格按 qid join，计算 Top-1/Oracle@K IoU、时间几何字段、分组统计和 100 条候选。
- 偏差：首次严格校验发现官方 prediction 中存在零长度候选窗口；audit 将 prediction 区间校验放宽为 `end >= start`，按 IoU=0 和长度 0 处理。GT 仍要求正长度。该 repair 未修改 baseline。

## 3. 实际观察

### 输入与指标

- GT queries：1347。
- Prediction queries：1347。
- Matched：1347；Missing：0；Duplicates：0。
- Query 和 vid 字段全部一致。
- 计算 R1@0.5：23.16；官方：23.16。
- 计算 R1@0.7：10.32；官方：10.32。

### Oracle ranking

| 指标 | Top1 | Top3 | Top5 | Top10 |
|---|---:|---:|---:|---:|
| R@0.5 | 23.16 | 36.01 | 42.17 | 49.44 |
| R@0.7 | 10.32 | 17.82 | 21.38 | 24.87 |

### Top-1 failure geometry

在 Top-1 IoU@0.7 失败的 1208 条中：

| Top-1 IoU 区间 | N | fraction |
|---|---:|---:|
| IoU = 0 | 610 | 50.50% |
| 0 < IoU < 0.3 | 253 | 20.94% |
| 0.3 ≤ IoU < 0.5 | 172 | 14.24% |
| 0.5 ≤ IoU < 0.7 | 173 | 14.32% |

全体 1347 条的几何计数为：`DISJOINT_EARLY=415`、`DISJOINT_LATE=195`、`PARTIAL_OVERLAP=425`、`PASS_05_ONLY=173`、`PASS_07=139`。

### Single-GT vs Multi-GT

| 组 | N | R1@0.5 | R1@0.7 | Oracle10@0.5 | Oracle10@0.7 |
|---|---:|---:|---:|---:|---:|
| Single-GT | 573 | 19.02 | 10.65 | 38.57 | 19.37 |
| Multi-GT | 774 | 26.23 | 10.08 | 57.49 | 28.94 |

### Audio duration

| duration | N | R1@0.5 | R1@0.7 | Oracle10@0.7 |
|---|---:|---:|---:|---:|
| 0–60 sec | 1 | 100.00 | 0.00 | 0.00 |
| 60–120 sec | 286 | 26.57 | 10.49 | 27.62 |
| 120–180 sec | 211 | 23.22 | 9.95 | 27.96 |
| 180–240 sec | 171 | 23.98 | 16.37 | 30.41 |
| 240+ sec | 678 | 21.39 | 8.85 | 21.39 |

### GT length

GT length binning uses the maximum GT window length per query.

| max GT length | N | R1@0.5 | R1@0.7 |
|---|---:|---:|---:|
| 0–2 sec | 90 | 2.22 | 0.00 |
| 2–5 sec | 376 | 11.17 | 3.72 |
| 5–10 sec | 273 | 26.37 | 10.26 |
| 10–20 sec | 251 | 31.47 | 16.33 |
| 20+ sec | 357 | 32.77 | 15.69 |

### Confidence deciles

Deciles are rank-based, ascending confidence, with deterministic qid tie-break.

| decile | N | mean confidence | R1@0.5 | R1@0.7 |
|---|---:|---:|---:|---:|
| 01 | 135 | 0.8484 | 14.81 | 7.41 |
| 02 | 135 | 0.9051 | 11.85 | 5.19 |
| 03 | 135 | 0.9249 | 16.30 | 5.19 |
| 04 | 134 | 0.9371 | 20.15 | 4.48 |
| 05 | 135 | 0.9453 | 15.56 | 8.89 |
| 06 | 135 | 0.9520 | 15.56 | 7.41 |
| 07 | 134 | 0.9591 | 16.42 | 4.48 |
| 08 | 135 | 0.9656 | 33.33 | 14.07 |
| 09 | 135 | 0.9734 | 34.81 | 11.85 |
| 10 | 134 | 0.9862 | 52.99 | 34.33 |

### Manual review sample

100 queries were selected with fixed seed and no duplicate qids:

- A high-confidence catastrophic: 20
- B ranking-gap: 20
- C boundary/localization: 20
- D multiple-GT failure: 20
- E general failure: 20

No query was manually listened to or given a semantic label.

## 4. 广泛性与证据等级

结果由全量 1347 个独立 query 支持，且 Top-1 指标精确复现官方结果。证据等级为 exploratory evidence。当前结果支持定量几何和排名诊断，不支持语义机制结论。

## 5. 推论边界

### 实际支持的推论

- Oracle@K 随 K 增加，说明部分 Top-1 失败中存在更高排名位置的合格候选。
- 超过一半的 Top-1 IoU@0.7 失败属于完全不相交几何关系。
- Multi-GT 组的 Oracle@10 高于 Single-GT 组，但其 Top-1@0.7 几乎相同；这是候选诊断现象。
- 240+ 秒组的 Top-1@0.7 低于 180–240 秒组，但这是描述性分组结果。

### 不支持的推论

- 不能据此断言语义失败、重复事件混淆、音频理解失败或标注错误。
- 不能据此证明某个新方法、排序模块或训练策略有效。
- 不能把时间戳几何类别当作人工听审标签。

## 6. 下一步状态

- 最终状态：GO。
- 判断依据：输入严格对齐，官方 Top-1 指标复现通过，且诊断输出完整。
- 最便宜的下一判别实验：对固定 100 条候选进行人工听审，并记录 query、音频 PCM provenance、审听者和标签定义。
