# 最小实验卡：FAILURE AUDIT v0A

## 基本信息

- 实验名称：DCASE 2026 Task 6 QD-DETR quantitative failure audit
- 日期：2026-08-20
- 当前 Gate：S / M diagnostic
- 主要决策：判断官方 baseline 的失败主要表现为候选排序、时间几何定位，还是可观测的结构性分组差异，并确定后续人工听审的抽样框架。
- 负责人：UNSPECIFIED
- 证据等级：exploratory evidence

## 问题

- 观察到的问题（不写方法名）：测试集 moment retrieval 的 Top-1 指标低于完美检索，需要分解窗口几何误差与候选排序结构。
- 问题出现的条件：官方 CASTELLA test split、固定 checkpoint、官方 prediction file。
- 当前问题证据：官方评测 R1@0.5=23.16，R1@0.7=10.32。
- 受影响的独立样本数/比例：1347 条 query；具体失败比例由本 audit 计算。
- 为什么值得研究：它决定后续人工审听应优先覆盖排序 gap、边界定位、完全不相交和多 GT 等诊断 regime。
- 什么结果会证明问题不值得研究：输入对齐失败、指标无法复现官方结果，或所有诊断分组都没有可区分的样本结构。

## 假设

- 机制假设：失败可能由候选窗口未被正确排序、Top-1 与 GT 仅部分重叠、完全不相交，或多 GT / 音频时长相关的结构差异组成。
- 如果假设成立，预期现象：Oracle@K 随 K 提升；不同几何标签、多 GT 分组和时长分组的 Top-1/Oracle 指标存在可量化差异。
- 如果假设错误，预期现象：Oracle@10 几乎不改善，且各诊断分组指标接近。
- 最小干预：无模型干预；仅对已有 GT 和 prediction 做确定性 join、IoU、排名和分组统计。
- Kill Criterion：GT/prediction 对齐或官方 Top-1 指标不能复现时，停止并判为 INVALID。

## 估计量与对照

- 主要估计量：描述性指标与 Oracle@K 的差异，不做 treatment-control 因果比较。
- 实验组：按几何、GT 数量、音频时长、GT 长度和 confidence decile 划分的 query 子组。
- 对照组：全体 query 及互补子组。
- 每个对照排除的竞争性解释：仅用于描述异质性；不将分组差异解释为语义机制。
- 必须保持不变的变量：官方 checkpoint、prediction 文件、GT、IoU 定义、query 顺序和测试集。

## 数据与资产

- 数据来源：`/private/research-artifact` 与 `results/submission.jsonl`。
- 最低样本量：全量 1347 queries。
- train/evaluation 隔离：仅读取已完成的 test evaluation 输出。
- provenance：baseline commit `45ef471ee47ea75a2141d75bd9cfdb8c45dfc101`。
- 所需人工工作：本阶段无；仅生成约 100 条未来听审候选。
- 所需资产是否已验证：VERIFIED
- canonical ledger / duplicate / conflict / current-file 审计：不涉及人工标签；输入 qid duplicate/missing 由脚本严格检查。

## 指标

- 主要指标：Top-1 R@0.5、R@0.7；Oracle@1/3/5/10。
- 样本级广泛性指标：query-level IoU、几何标签、single-GT vs multi-GT、duration/GT-length/confidence 分组。
- 副作用指标：无模型副作用指标。
- invalid / ABSENT / 极端值处理：非法 JSON、重复/缺失 qid、query/vid 不一致、非法区间或非有限分数均 fail loudly。
- 只用于探索的指标：failure IoU 分布、first-hit rank、confidence deciles 和人工听审候选分组。

## 决策规则

### GO

必须满足：输入严格对齐，且计算得到的 Top-1 R@0.5/R@0.7 在 0.02 个百分点内复现官方 23.16/10.32。

### NO_GO

满足以下任一条件即停止：

- 对齐失败或官方指标无法复现。

### INCONCLUSIVE

- 仍缺失的信息：时间戳几何统计不能证明语义原因；需后续人工听审。
- 最便宜的下一判别实验：按本 audit 产生的固定候选样本进行人工听审，并保存 provenance。

## 停止与资源上限

- 科学停止条件：完成全量确定性统计与约 100 条候选抽样。
- 资源停止条件：不下载原始音频，不运行模型，不使用 GPU。
- 方向停止条件：若输入无效，停止而不修补 baseline。
- 最大 GPU 小时：0
- 最大费用：UNSPECIFIED
- 最大人工时间：0（本阶段）
- 最大调试轮数：3
- 最大超参数尝试：0
- 最晚停止日期：UNSPECIFIED
- 禁止抢救方式：修改 baseline、重训、调整阈值/NMS、引入新模型或用时间戳推断语义。

## 结论上限

即使实验成功，最多支持对官方 prediction 的定量几何与排序结构描述，以及未来人工听审的抽样；不支持语义失败归因或新方法有效性结论。
