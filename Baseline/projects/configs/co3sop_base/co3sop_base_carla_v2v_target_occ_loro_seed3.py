# 任务38：Stage2B 留一个 run 交叉验证，seed3 这一折（沿用任务37 seed5 折的设置，epoch 定为 5：seed5 实际跑到 5，最好在 3）。
# 训练 = 其余 5 个 run（fold_seed3.json 的 train），早停/选 epoch 用各训练 run 的时间尾段（val），
# seed3 这一折的测试部分不参与训练和选择。新数据里 TX_CAR2（橙车）约 25% 在前、28% 在侧方，迫使 target 分支
# 靠外观（颜色）认车，而不是 legacy 里“角色 0 = 前车 / 角色 1 = 后车”的位置捷径（任务36）。
# 从原 Stage2B（phaseB epoch_12，只见过 legacy sid 2–626）微调，其余设置与 merged_v4 相同。
_base_ = ['./co3sop_base_carla_v2v_target_occ_merged_v4.py']

split_file = '/home/admin0/Co3SOP/Baseline/work_dirs/loro_cv/splits/fold_seed3.json'

data = dict(
    workers_per_gpu=2,  # 共享服务器内存紧张（systemd-oomd）
    train=dict(split_file=split_file),
    val=dict(split_file=split_file),
    test=dict(split_file=split_file, ann_file='test'),
)

total_epochs = 5
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'work_dirs/carla_v2v_target_occ_phaseB/epoch_12.pth'
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
