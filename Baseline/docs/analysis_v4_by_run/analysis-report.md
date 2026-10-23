# Analysis Report — 任务30 v4 sweep 按 source run 重新分析

> **2026-10-24 更新**：补训 2 个种子后的 run × 种子分析见 `seed_x_run/analysis-report.md`，其中 Q2（预测下加 scene）和 Q4（预测 vs GT target）的结论已经修改，以那份为准。

数字和检验细节见 `stats-appendix.md`，图见 `figure-catalog.md`。

## 分析问题

任务30/32 的结论是在 val 的 1528 行上 pooled 计算出来的，相当于默认各行相互独立。这次把 **source run（legacy + seed1–5，n = 6）** 作为分析单位，同一 run 上的不同模型配对比较。要回答四个问题：
1. GT 输入下，加 scene 是否稳定有帮助？（Q1）
2. 预测输入下，加 scene 是否有帮助？（Q2）以及两种输入下的 scene 增益是否不同？（Q7 交互）
3. 预测 occupancy 相比 GT 损失了多少？（Q3、Q4）
4. 预测 occupancy 是否稳定好于 majority 基线？（Q5、Q6）

主指标：面板对准确率（TX、RX 两侧面板都选对）。

## 主要发现

1. **GT 下加 scene 的增益在每个 run 都成立**：+18.2 个百分点 [10.4, 25.9]，6/6 个 run 为正（最小 +10.0，seed2），d_z = 2.5，留一 run 后均值仍在 +15–20 点；6 个次指标上也都是 6/6 为正。
2. **预测输入下加 scene 平均为 0，但不是“每个 run 都约等于 0”**：均值 −0.0 [−9.5, +9.5]，各 run 从 −9.4（seed1）到 +16.5（seed5）。任务30 说“差在 1–2 点以内”只在 pooled 层面成立；按 run 看是正负抵消，seed5（L/R 和 NLOS 最多的 run）是唯一明显受益的 run。
3. **交互效应稳定**：GT 下的 scene 增益比预测下高 +18.2 点 [12.2, 24.1]，6/6 run 为正，d_z = 3.2。“场景信息有价值，但在 occupancy 预测这一步丢失了”这一判断，现在有了 run 级别的支持。
4. **任务30 记录中的“反常现象”（pred/target 好于 gt/target）在 run 层面不成立**：pooled 差 +4.7 点、行级 McNemar p = 0.0006，看起来高度显著；但按 run 只有 4/6 为正，seed3（−9.2）和 seed5（−13.2）方向相反，p_perm = 0.47。这是把相关行当成独立样本造成的假阳性。
5. **预测 vs GT（scene_target）**：+13.9 点 [1.7, 26.0]，5/6 run 为正（seed2 −0.9），p_perm = 0.062。方向基本一致，但幅度在不同 run 间差异很大（+0.7 到 +28.5 点）。
6. **对比 majority 基线主要受 run 构成影响**：各 run 的 majority 准确率从 0.03（seed5）到 0.996（seed4）。在 seed4 上，两个预测模型都**低于** majority（−4.7 / −10.9 点）。pooled 的“0.71 vs 0.52”掩盖了这一点。
7. **gt/scene 在每个 run 上都与 majority 完全相同**，在 run 级别再次确认了任务30 的“坍缩”诊断。
8. **少数类和 NLOS 实际上只是几个事件**：整个 val 只有 7 个 L/R 事件和 6 个 NLOS 事件；L/R recall 在 run 内几乎非 0 即 1。legacy 和 seed2 的 L/R 事件所有模型都完全失败；seed2 的 41 行 NLOS（1 个事件）所有模型准确率都 ≤ 0.07，而 seed2 的 train 部分一行 NLOS 都没有。

## 证据最强的比较

- Q1（GT scene 增益）和 Q7（交互）：6/6 run 方向一致，CI 远离 0，效应量很大，留一 run 后结论不变。
- 统计上的限制：n = 6 时精确置换检验的最小 p 是 0.031，7 个对比做 Holm 校正后不可能低于 0.22。只有在配对 t 检验的正态假设下，Q1（Holm p = 0.011）和 Q7（Holm p = 0.0035）才能通过校正。因此措辞应写成“6 个 run 中全部一致”，而不是“显著（p < 0.05）”，除非同时说明是参数检验、并说明假设。

## 主要注意事项

- 每组只有 **1 个训练种子**；run 间差异不包含训练随机性。Q2/Q4 里 ±10 点的单 run 波动，本身就可能是一次训练的偶然结果。
- val 是每个 run 的时间尾段，与 train 紧邻，并且其中的 NLOS/L/R 事件在 train 里几乎没有。这个划分测的更接近“同一 run 内向后外推”，不是 i.i.d. 泛化，也不是跨 run 泛化。
- seed4 的 val 几乎全是 majority 类（0.996），信息量很低；seed5 则相反。
- 本次没有重新训练，也没有修改任何标签；所有数值都来自任务32 缓存的 val logits。

## 对实验理解的更新

- 任务30 结论 1（scene_target 最好）和“场景信息在预测这一步丢失”的判断**被加强**：在 run 层面也一致。
- “pred/target > gt/target 的反常”**应当撤回**：不再作为需要解释的现象，最多算作需要用多种子重新确认的弱信号。
- “预测 scene 没有额外帮助”应改写为“**平均没有帮助，且在不同 run 间不稳定**”；唯一受益的 seed5 恰好是 L/R/NLOS 最多的 run，值得用多种子检查是否可重复。
- 当前 split 下，L/R 和 NLOS 的任何 recall 数字都只是事件级的个例，不能作为模型能力的估计。

## Claim Candidates

- Claim: GT occupancy 下，加入 scene 通道能提高面板对准确率。
  - Source evidence: Q1，+18.2 点 [10.4, 25.9]，6/6 run，d_z 2.5；6 个次指标也都是 6/6；图1、图2。
  - Allowed wording: "With GT occupancy, adding scene channels improved panel-pair accuracy in all six source runs (mean +18 pp, 95% CI 10–26 pp; single training seed)."
  - Forbidden stronger wording: "significantly improves (p<0.05)"（无分布假设的检验做校正后达不到）；“scene 信息对 beam 选择普遍必要”（只有 GT 输入下成立）。
  - Uncertainty: 单一训练种子；val 是时间尾段。
  - Next check: 每组 3 个种子，重新做 run × seed 的分析。
  - Decision: keep

- Claim: 预测 occupancy 下加 scene 没有带来平均收益，且这一点与 GT 下的增益明显不同。
  - Source evidence: Q2 −0.0 [−9.5, 9.5]（2/3 正负）；Q7 交互 +18.2 [12.2, 24.1]，6/6 run。
  - Allowed wording: "The scene-channel gain observed with GT occupancy did not carry over to predicted occupancy (mean change ≈0 pp, run-level range −9 to +17 pp)."
  - Forbidden stronger wording: “预测的 scene occupancy 没有任何信息”；“预测 scene 在所有场景都无用”（seed5 +16.5 点）。
  - Uncertainty: 单一种子；各 run 间方向不一致。
  - Next check: 多种子；直接评估预测 scene occupancy 在 TX 车和周围车辆附近的准确率（任务30 补充里已经列出、尚未执行）。
  - Decision: keep（采用加了限定的措辞）

- Claim: 预测 occupancy 相比 GT 会让面板对准确率下降。
  - Source evidence: Q3 +13.9 点 [1.7, 26.0]，5/6 run，p_perm 0.062，Holm 0.31。
  - Allowed wording: "Replacing GT with predicted occupancy lowered panel-pair accuracy in 5 of 6 runs (mean −14 pp)."
  - Forbidden stronger wording: “显著下降”；给出一个固定的下降幅度（各 run 从 −1 到 −29 点）。
  - Uncertainty: 中等；seed2 方向相反（−0.9）。
  - Next check: 多种子。
  - Decision: weaken

- Claim: pred/target 优于 gt/target（任务30 记录中的反常现象）。
  - Source evidence: Q4 +4.3 点 [−8.7, 17.4]，4/6 run，p_perm 0.47；行级 McNemar p = 0.0006 是聚类导致的假阳性。
  - Allowed wording: 不作为结论使用；最多写“两者在 run 层面无一致差异”。
  - Forbidden stronger wording: “预测 target 比 GT target 更好/更容易优化”。
  - Uncertainty: 高。
  - Next check: 多种子；查清 gt/target 欠拟合的原因（任务30 补充里已列出）。
  - Decision: discard

- Claim: 预测 occupancy 模型优于 majority 基线。
  - Source evidence: Q5 +18.8 点 [1.2, 36.4]，5/6 run，seed4 −4.7；Q6 CI 跨 0。
  - Allowed wording: "Predicted-occupancy models exceeded the majority-panel baseline in 5 of 6 runs; in the run where the majority class covers 99.6% of rows, they fell below it."
  - Forbidden stronger wording: “稳定/显著优于基线”。
  - Uncertainty: 中等，主要由各 run 的类别构成决定。
  - Next check: 改用 balanced accuracy 或按面板加权的指标，减少构成的影响。
  - Decision: weaken

- Claim: 模型在 L/R 少数类和 NLOS 上的能力（任何 recall 数字）。
  - Source evidence: val 只有 7 个 L/R 事件、6 个 NLOS 事件；recall 在 run 内非 0 即 1；seed2 的 NLOS 在 train 中没有出现过。
  - Allowed wording: 只描述事件数量和逐事件结果。
  - Forbidden stronger wording: “L/R recall 从 0.03 提升到 0.x”这类按行计算的比例。
  - Uncertainty: 极高。
  - Next check: 按事件划分 split（整段事件放进 train 或 val），或采集更多超车/NLOS 事件。
  - Decision: revise

## 建议的下一步（按对决策影响排序）

1. 每组补 2 个训练种子，按 run × seed 重新分析（Q2/Q4 是否成立取决于这一步）。
2. 改成按事件/时间块划分，或做 leave-one-run-out 训练，让 val 测的是跨场景泛化，而不是同一 run 内向后外推。
3. 主表补充 balanced accuracy / 面板宏平均 recall，减轻 seed4/seed5 这类构成极端的 run 对结论的影响。
