# Stage2B 按折重训：seed5 这一折（任务37）

配置 `projects/configs/co3sop_base/co3sop_base_carla_v2v_target_occ_loro_seed5.py`：从 phaseB epoch_12 微调，
训练 = legacy + seed1–4（5419 行），内部验证 = 各训练 run 后 15%（962 行），测试 = seed5（1233 行，Stage2B 和 beam 头都没见过）。
启动用 `tools/dist_train.sh`（单卡分布式，验证才能走 CustomDistEvalHook）。

## Stage2B 内部验证 target IoU

| epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| target IoU | .545 | .616 | **.625** | .601 | .623 |

选 epoch 3，提取到 `beam_occ_native_loro_seed5`。

## 预测 target 诊断（seed5，逐行平均，`compare_diag.py`）

| 子集 | 旧 IoU | 新 IoU | 旧方位误差 | 新方位误差 |
|---|---|---|---|---|
| 全部 | .25 | .64 | 6.3° | 0.8° |
| 侧方 30–150° | .00 | .72 | 76° | 2.8° |
| 10 m 以内 | .09 | .77 | 61° | 1.5° |
| 正后 150–180° | .30 | .48 | 2.4° | 1.1° |
| 30 m 以上 | .13 | .32 | 13° | 1.2° |

预测落在本车 3 m 内：23% → 0%（TX_CAR2 45% → 0%）。

## 去掉邻车的消融（只做对比）

`data.test.connect_range=-1`，邻车特征置零。test.py 汇总口径：新 Stage2B target IoU .608 → .026，旧 .229 → .047。
target 定位几乎全靠两辆 TX 车的输入（相机画面 + 位姿，这个实验分不开）；模型没见过零邻车输入，下降里有分布外成分。

## beam 头（seed5，1 个 seed）

| 组合 | 全部 | 左右面板 | 非左右 | NLOS |
|---|---|---|---|---|
| 旧 pred/target | .770 | .458 | .855 | .244 |
| 新 pred/target | .842 | .833 | .869 | .244 |
| 旧 pred/scene_target | .745 | .675 | .780 | .244 |
| 新 pred/scene_target | .833 | .773 | .869 | .268 |
| GT target / scene_target | .941 / .933 | .946 / .951 | | .244 |

pred 与 GT 的差距从 17–19 点缩小到约 10 点，主要来自左右面板。只有 1 个 run，结论待其余 5 折。
