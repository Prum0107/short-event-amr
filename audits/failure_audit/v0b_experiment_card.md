# 最小实验卡：FAILURE AUDIT v0B

- 当前 Gate：S — short-moment and confound diagnostic。
- 主要决策：判断短 GT moment 的性能下降是否表现为相对时间误差放大，及 audio-duration 分层后是否仍存在明显下降。
- 证据等级：exploratory evidence。
- 输入资产：仅使用 v0A 的 `/private/research-artifact`，预期 1347 queries。
- 实验操作：按 `max_gt_length` 分箱，计算 Top-1/Oracle@K、匹配 GT 的时间误差及 IoU 分布；再按粗 GT 长度 strata 交叉列出 audio-duration 结果。
- 时间误差定义：Top-1 prediction 匹配到产生最大 Top-1 temporal IoU 的 GT window；绝对误差为对应边界差的绝对值；归一化误差除以该匹配 GT 的 `gt_length`；只统计匹配 GT 长度为正且字段有效的行。
- 主要决策规则：若输入行数、字段和 v0A 输出可读，则完成诊断；本阶段不建立复杂模型、不做因果结论。
- 资源上限：0 GPU 小时；不运行模型；不下载原始音频；不修改 baseline。
- 结论上限：最多支持短 moment 的定量误差与分层相关性描述，不能支持语义归因或方法有效性结论。
