# Analysis Report — v4 sweep，run × 训练种子（任务34b）+ 实验修改方案

本报告取代上一级目录 `analysis-report.md`（单种子版本）中的结论；单种子版本保留作对照。数字见 `stats-appendix.md`，图见 `figure-catalog.md`。
本次用到的 skill：results-analysis（分析框架）、experimental-design（设计诊断）、statistical-power（功效分析）。后两个来自 K-Dense Scientific Agent Skills（MIT）。

## 分析问题

每个组合补训了 2 个种子（共 3 次训练），要回答两件事：
1. 单种子分析得出的结论，哪些在训练随机性下仍然成立？
2. 场景差异和训练噪声各占多少？由此决定实验该怎么改：补 run 还是补种子、怎么划分数据、需要多少数据。

分析单位仍然是 source run（n = 6），每个 run 先对 3 个种子取均值，再按 run 配对。主指标：面板对准确率。

## 主要发现

1. **方差的大头是场景，不是训练。** 种子之间的差异在汇总值上是 1–4 个点，在单个 run 内一般 3–7 个点；run 间差异是 12–28 个点。汇总层面，Q1–Q4 四个对比在 9 种“A 种子 × B 种子”组合下**方向全部一致**。
2. **GT 下加 scene 的增益成立**：+15.9 点 [6.4, 25.4]，6/6 run 为正，Holm(t) = 0.047。但 seed2 这个 run 上只有 +0.8，而且 gt/scene_target 在 seed2 上 3 次训练的结果分别是 .29、.55、.59，很不稳定。
3. **交互效应成立**：GT 下的 scene 增益比预测下高 +20.8 点 [9.8, 31.9]，6/6，Holm(t) = 0.032。
4. **（与单种子结论不同）预测输入下加 scene 偏向变差**：−5.0 点 [−13.1, +3.2]，5/6 run 为负；汇总层面 9/9 种种子组合都是负的（pred/scene_target .634–.689，pred/target .692–.723）。单种子时的“≈0”只是 s0 碰巧接近。
5. **（与单种子结论不同）pred/target 优于 gt/target 在汇总层面可以复现**：9/9 种种子组合都为正，种子均值 .705 vs .638；但 run 层面只有 4/6 为正（seed3、seed5 是负的）。原因很可能是优化问题：gt/target 3 次训练都欠拟合（最佳 epoch 时 train 面板 .69–.77），pred/target 能拟合到 .98–.99。这个差异不能解读为“预测 occupancy 比 GT 信息量更大”。
6. **NLOS 上所有模型都基本失败**（面板对准确率 .20–.25，与组合和种子无关）；L/R 上 GT 输入约 .5、预测输入约 .2。只有 6 个 NLOS 事件和 7 个 L/R 事件，只能作为描述。
7. **设计层面的问题（分析补救不了）**：(a) 模型选择和结果报告用的是同一个 val；(b) val 是每个 run 的时间尾段，测的是同一 run 内向后外推，NLOS/L/R 事件几乎不在 train 里出现；(c) 只有 6 个独立 run；(d) 预测 occupancy 来自只用 legacy 训练过的 Stage2B，pred 组的输入质量在 legacy 和 seed1–5 之间本来就不同。按现在的规模，要有 80% 的把握检测出 10 点的差异，需要 7–19 个 run；检测 5 点需要 21–69 个 run。现在 6 个 run 时，检测 10 点差异的功效只有 28–72%。

## 证据最强的比较

- Q1、Q7：6/6 run 为正，在 t 检验的正态假设下能通过 Holm 校正，种子之间方向一致。
- Q2、Q4：在汇总层面不随种子改变，但在 run 层面不显著，而且功效不足（现在的设计下这两项最多只能检测到约 10 点以上的差异），所以“不显著”不等于“没有差异”。

## Claim Candidates（更新单种子版本）

- Claim: GT occupancy 下，加入 scene 通道提高面板对准确率。
  - Source evidence: Q1 +15.9 [6.4, 25.4]，6/6 run，d_z 1.75，Holm(t) .047；9/9 种种子组合为正。图4、图5。
  - Allowed wording: "With GT occupancy, adding scene channels improved panel-pair accuracy in all six source runs (mean +16 pp, 95% CI 6–25 pp; 3 training seeds per model)."
  - Forbidden stronger wording: “在所有场景都大幅提升”（seed2 只有 +0.8，且训练很不稳定）；不带检验方法说明地写“p < 0.05”。
  - Uncertainty: 只有 6 个 run；val 同时用于选模型。
  - Next check: 按新划分重新评估（见修改方案）。
  - Decision: keep

- Claim: 预测 occupancy 下加入 scene 通道没有帮助，且偏向变差。
  - Source evidence: Q2 −5.0 [−13.1, 3.2]，5/6 run 为负；9/9 种种子组合汇总差为负。
  - Allowed wording: "With predicted occupancy, adding scene channels did not improve accuracy; pooled accuracy was lower in all 9 seed pairings (−3 to −9 pp), and lower in 5 of 6 runs."
  - Forbidden stronger wording: “预测 scene 显著有害”；“scene 信息本身无用”（GT 下增益很大，见 Q1/Q7）。
  - Uncertainty: run 层面功效不足。
  - Next check: 直接评估预测 scene occupancy 在 TX 车和周围车辆附近的准确率；加入正则化后重试。
  - Decision: revise（原来写的是“约等于 0”）

- Claim: 场景信息的价值在 occupancy 预测这一步丢失了（交互效应）。
  - Source evidence: Q7 +20.8 [9.8, 31.9]，6/6，Holm(t) .032。
  - Allowed wording: "The benefit of scene channels observed with GT occupancy (+16 pp) was absent with predicted occupancy (−5 pp); the difference was positive in all six runs."
  - Forbidden stronger wording: 把原因直接归到预测误差（也可能是过拟合，预测输入的组 train .96–.99）。
  - Uncertainty: 机制还不清楚。
  - Next check: 同上；同时控制过拟合。
  - Decision: keep

- Claim: 预测 target 比 GT target 好。
  - Source evidence: 汇总层面 9/9 种种子组合为正（+5–9 点），run 层面 4/6，p_perm .31；gt/target 每次都欠拟合。
  - Allowed wording: "pred/target exceeded gt/target on pooled val in every seed pairing, but not consistently across runs; gt/target underfit in all three trainings (train panel ≤ 0.77), so the gap likely reflects optimization rather than input information."
  - Forbidden stronger wording: “预测 occupancy 比 GT 信息量更大/更好”。
  - Uncertainty: 优化问题和输入信息量的影响混在一起，无法分开。
  - Next check: 先把 gt/target 调到能正常拟合（学习率、patience、选模型的标准），再比较。
  - Decision: revise（单种子版本是 discard；现在汇总层面可以复现，但原因是优化问题）

- Claim: 预测 occupancy 模型优于 majority 基线。
  - Source evidence: Q5 +20.1 [−0.2, 40.3]，5/6；seed4 上低于 majority。
  - Allowed wording: 只写“5/6 run 高于 majority，在类别几乎全是 majority 的 run 上低于它”。
  - Forbidden stronger wording: “稳定/显著优于基线”。
  - Decision: weaken

- Claim: 模型在 NLOS / L/R 上的能力。
  - Source evidence: NLOS 准确率 .20–.25，与组合无关；只有 6 个 NLOS 事件、7 个 L/R 事件。
  - Allowed wording: 只描述事件数和逐事件结果。
  - Forbidden stronger wording: 任何按行计算的 recall 比例。
  - Decision: revise

## 实验修改方案

按 experimental-design 的框架：被比较的“处理”是 6 个输入组合；独立重复单位是 run（场景）；事件、帧、链路嵌套在 run 内；训练种子是技术重复。现在的设计以 run 为区组（每个 run 都评估所有组合），这一点是对的。问题在于测试集的划分方式、模型选择，以及独立单位太少。

### A. 数据划分（最优先；不需要新数据，可以马上做）

1. **按 run 分组的 K 折交叉验证，取代“每个 run 的时间尾段”**。现在 6 个 run，可以每次留出 1 个 run（6 折）。这样每个 run 的**全部**数据（约 1000–1600 行，而不是 220–314 行）都会在某一折里当测试集，每个 run 的估计会更准，NLOS/L/R 事件也都能进入测试。
2. **测试集不参与任何选择**：在训练用的 run 内部，再按事件或连续时间块（中间留 ≥ 2 s 的间隔）切出一个内部 val，用来早停和选 epoch。测试 run 只在最后评估一次。
3. **预测 occupancy 的模型也要避开测试 run**。已核查（实验记录 §21、第 507 行）：pred 输入来自冻结的 Stage2B checkpoint，它**只用 legacy run 的 train 部分（sample_id 2–626）训练过**。因此：
   - seed1–5 对 Stage2B 来说完全没见过，不存在泄漏；但它们是新路线，预测 occupancy 的质量本来就可能比 legacy 差。所以**现有结果里，pred 组在不同 run 之间的差异和“Stage2B 是否见过这个 run”混在一起**，pred 相关的对比（Q2、Q3、Q4）都受这个影响。
   - 做“留一个 run”交叉验证时，legacy 被留作测试的那一折有泄漏（Stage2B 见过它 80% 的帧），这一折的 pred 结果要单独标出或排除。
   - 彻底的做法是按同样的折重新训练 Stage2B（代价最大，需要你决定）；折中的做法是在新 run 上统一使用这个 Stage2B（它没见过任何新 run），把 legacy 只当作额外的参考。

### B. 重复：补 run，而不是补种子

1. 每组保持 3 个种子就够了。从 3 加到 5，所需 run 数只少 2–4 个。
2. 新增 run 的目标：**总数 ≥ 20 个**（τ ≤ 0.14 时，能以 80% 功效检测 10 点的差异）。要检测 5 点的差异需要 40 个以上的 run，代价可能不划算，建议主结论以 10 点为最小关心差异（这个阈值请你定）。
3. 按目前的流程估算每个新 run 的成本：CARLA 采集约 13 分钟，Sionna 约 1.4 小时（单进程），再加上 occupancy GT 和预测的提取。补 14 个 run 约需 1.5–2 天机器时间。
4. 新数据开始采集之前，要先修任务33 发现的背景车遮挡盒与本车重叠的问题（后 4 个 run 里有 6.3% 的样本受影响），否则新 run 会继承这个问题。

### C. 分层：保证少数类有足够的独立事件

1. 采集时按场景类型分层：每个新 run 用 seed 控制，至少包含一次并排超车（L/R）和一次被遮挡的 NLOS 段，并记录事件 ID。
2. 少数类按**事件**汇报（每个事件里预测正确的帧的比例），并单独汇报事件数。不再报告按行计算的 recall。
3. 主指标同时报告面板对准确率和按面板的宏平均 recall（balanced），减轻 seed4/seed5 这类构成极端的 run 对结论的影响。

### D. 组合和训练

1. 主实验保留 4 个组合（gt/pred × target/scene_target）。scene-only 两组的坍缩机制已经查清，以后只作为消融对照，在一折上跑一次即可。这样训练量从 6 × 3 × K 减到 4 × 3 × K：K = 5 时是 60 次训练，按目前每次约 30–60 分钟、两条线并行，约 15–30 小时。
2. 比较 GT 和预测之前，先解决 gt/target 欠拟合（调学习率或 warmup，加大 patience，按 val loss 而不是 Top1 早停），以及预测输入组的过拟合（weight decay、dropout、数据增强）。否则 Q4 这类对比会一直被优化问题干扰。
3. 长时间训练要控制内存：两条线并行时，加上其他用户的进程，曾两次触发 oomd 杀掉整个会话。建议把 DataLoader 的 num_workers 从 4 降到 2，或者在内存紧张时只跑一条线。

### E. 预先确定分析方案（在看新结果之前写定）

- 分析单位：测试 run；每个 run 先对 3 个种子取均值。
- 主对比：Q1、Q2、Q3（和 Q7），用配对 t 检验，在主对比族内做 Holm 校正；同时报告置换检验和 run 级区间。
- 最小关心差异：10 个点（待你确认）。
- 次要指标只描述，不做推断。

## 建议的执行顺序

1. 不需要新数据的部分：用现有 6 个 run 做 6 折“留一个 run”交叉验证（A1、A2，以及 D2 的训练修复），legacy 那一折的 pred 结果单独标出（A3）。这一步的结果就能回答 Q2/Q4 是不是被划分方式和优化问题扭曲了。
2. 修背景车重叠问题（B4），再按 C1 分层采集新 run，目标是总数 ≥ 20。
3. 新数据到位后按 E 的方案正式分析。

## 引用说明

本次用到了 K-Dense 的 experimental-design 和 statistical-power skill。按这两个 skill 的要求，如果这部分分析写进论文或报告，应引用：Kassis, T. et al. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065。
