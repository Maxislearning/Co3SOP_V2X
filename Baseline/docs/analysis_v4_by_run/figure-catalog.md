# Figure Catalog

所有图由 `plot_figures.py` 从 `data/*.csv` 生成，每张图都有 PDF 和 PNG 两个版本。

## figure-01-per-run-scene-gain

- 文件：`figures/figure-01-per-run-scene-gain.{pdf,png}`
- Purpose：主图。按 run 展示加 scene 通道前后的面板对准确率，对比 GT 输入和预测输入。
- 数据来源：`data/per_run_long.csv`（metric = panel_pair）。
- 绘制变量：x = 输入通道（target → scene+target）；y = 每个 run 的 val 面板对准确率；灰线 = 一个 run；蓝色 ◆ = 6 个 run 的均值 ± 95% t-CI（df = 5）；坐标轴左侧的短横线 = 该 run 的 majority 基线。
- 误差棒含义：run 间（场景构成）的 95% t-CI，**不包含训练种子带来的方差**。
- Caption 必须说明：n = 6 个 source run；单一训练种子；val = 每个 run 的时间尾段；误差棒的含义。
- 关键观察：左图 6 条线全部上升；右图的线有升有降，平均持平，seed5 上升，seed1/seed4 下降。
- 解读检查：(1) 为什么有这张图：检验 scene 增益是否每个 run 都成立。(2) 应该注意到：GT 下的增益在 run 间一致，预测下不一致。(3) 改变了什么：预测 scene 的问题从“没用”变成“不稳定”，下一步需要多种子和对预测 scene 质量的直接评估。
- 已知限制：run 的 majority 基线差异巨大（0.03–1.00），绝对水平的高低主要由构成决定，应主要看连线的斜率，而不是高度。

## figure-02-run-level-contrasts

- 文件：`figures/figure-02-run-level-contrasts.{pdf,png}`
- Purpose：主族 7 个配对对比的森林图，同时对照 run 级别和 pooled 行级两种估计。
- 数据来源：`data/contrasts_primary.csv`。
- 绘制变量：灰点 = 每个 run 的 Δ（纵向稍微错开以免重叠）；蓝色 ◆ + 横线 = run 均值 ± 95% t-CI；橙色 ▼ = pooled 行级 Δ（仅作参考）；右侧文字 = 正的 run 数、精确置换 p、Holm 校正后 p。
- Caption 必须说明：置换检验在 n = 6 时最小 p = 0.031；Holm 族的大小是 7；行级 p 值没有显示，因为它不是有效证据。
- 关键观察：Q1 和交互项的 CI 远离 0，6/6 run 为正；“target: Pred − GT” 的 pooled 值偏正，但各 run 点分布在 0 两侧。
- 解读检查：(1) 为什么有这张图：把“哪些结论在 run 层面站得住”放在一张图里。(2) 应该注意到：pooled 点估计和 run 均值很接近，变化的是不确定性，行级检验严重低估了它。(3) 改变了什么：撤回 pred > gt (target) 这一反常结论；Q3/Q5 的措辞降级。
- 已知限制：t-CI 依赖近似正态；只有 6 个点。

## figure-03-run-by-model-heatmap

- 文件：`figures/figure-03-run-by-model-heatmap.{pdf,png}`
- Purpose：辅助图。展示完整的 run × 模型矩阵和各 run 的构成，用来解释 run 间异质性的来源。
- 数据来源：`data/per_run_long.csv`、`data/per_run_composition.csv`；事件数来自 `stats-appendix.md` §2。
- 绘制变量：格子 = 面板对准确率（单色蓝色顺序色阶，0–1，并直接标注数值）；x 轴标签 = run 名、val 行数、NLOS 行数（事件数）、L/R 事件数。
- Caption 必须说明：每个格子是一个模型在一个 run 上的准确率；单一种子；事件按“相邻 sample_id 间隔 ≤ 0.5 s”定义。
- 关键观察：majority 行和 gt/scene 行完全相同（坍缩）；seed4 列几乎全部接近 1（退化 run）；seed5 列的 majority 只有 0.03，且 pred/target 在这里最差（0.33）。
- 解读检查：(1) 为什么有这张图：说明 run 构成决定了绝对水平。(2) 应该注意到：真正有信息量的是 L/R/NLOS 较多的 run（legacy、seed2、seed5），而它们只包含少数几个事件。(3) 改变了什么：建议改为按事件划分 split，并加入 balanced accuracy。
- 已知限制：颜色只编码一个量；格子里直接写了数值，不依赖颜色读数。
