# 按折重训 Stage2B 后的留一个 run 交叉验证（任务37–38）

每一折都用其余 5 个 run 重训 Stage2B（从 phaseB epoch_12 微调，最多 5 个 epoch，按内部验证 target IoU 选 epoch），
再用它提取占用，训练 pred/target 和 pred/scene_target 两个 beam 头（gated CoordConv，1 个 seed）。测试 run 对
Stage2B 和 beam 头都没见过。流水线：`tools/stage2b_fold_pipeline.sh <fold>`；分析：`analyze_stage2b_loro.py`。

| 折 | 选的 epoch | 内部验证 target IoU |
|---|---|---|
| legacy | 5 | .569 |
| seed1 | 4 | .499 |
| seed2 | 5 | .512 |
| seed3 | 5 | .516 |
| seed4 | 5 | .540 |
| seed5 | 3 | .625 |

## 各折面板对准确率（%）

| 折 | GT target | 旧 pred target | **新 pred target** | GT scene_target | 旧 pred scene_target | **新 pred scene_target** |
|---|---|---|---|---|---|---|
| legacy | 84.1 | 81.8 | 84.1 | 84.1 | 84.1 | 81.5 |
| seed1 | 97.4 | 59.4 | 91.6 | 94.1 | 81.2 | 82.3 |
| seed2 | 88.1 | 67.0 | 71.2 | 87.2 | 63.3 | 69.9 |
| seed3 | 98.6 | 82.3 | 97.3 | 98.5 | 82.5 | 97.7 |
| seed4 | 96.9 | 86.4 | 95.4 | 98.5 | 86.6 | 96.6 |
| seed5 | 94.1 | 77.0 | 84.2 | 93.3 | 74.5 | 83.3 |
| **均值** | 93.2 | 75.7 | **87.3** | 92.6 | 78.7 | **85.2** |

左右面板均值：GT 74.1，旧 pred 29.0 → 新 pred 58.9（target）；scene_target 40.5 → 57.1。legacy 的左右面板所有模型都是 0。

## 对比（百分点，按 run 配对，n=6；最小关心差异 10 点）

| 对比 | 均值 | 95% CI | 正向 run | p（置换） | p（t） | Holm（置换 / t） |
|---|---|---|---|---|---|---|
| target：新 pred − 旧 pred | +11.6 | [0.1, 23.2] | 6/6 | .031 | .049 | .125 / .166 |
| scene_target：新 pred − 旧 pred | +6.6 | [−0.2, 13.3] | 5/6 | .094 | .054 | .125 / .166 |
| target：GT − 新 pred | +5.9 | [−0.9, 12.7] | 5/6 | .062 | .076 | .125 / .166 |
| scene_target：GT − 新 pred | +7.4 | [0.4, 14.4] | 6/6 | .031 | .041 | .125 / .166 |
| 新 pred：scene_target − target（次要） | −2.1 | [−6.0, 1.9] | 2/6 | .219 | .240 | — |

## 结论

- **重训 Stage2B 让 pred 组明显变好**：target 组平均 +11.6 点，6 个 run 全部提升，均值超过 10 点的最小关心差异；
  pred 和 GT 的差距从 17.5 点（任务35）缩小到 5.9 点。提升主要在左右面板（29 → 59）。
- **统计上**：n=6 时精确置换检验最小 p 是 2/64 = .031，4 个对比 Holm 后最小只能到 .125，所以单看 Holm 都不显著；
  这里以效应量和 CI 为主。target 的提升 CI 下限刚好在 0 以上；GT − 新 pred 的 CI 包含 0 也包含 10，还不能说已经追平。
- **seed2 是例外**：只提升 4 点，左右面板 30（GT 74），离 GT 还有 17 点，需要单独诊断。
- **加 scene 仍然没有帮助**（−2.1 [−6.0, 1.9]）。
- 局限：1 个 seed；Stage2B 只跑到 5 个 epoch，多数折最好的 epoch 是最后一个，可能还没收敛；target 定位依赖邻车
  （见 `../analysis_stage2b_diag/report_loro_seed5.md` 的去邻车消融）。
