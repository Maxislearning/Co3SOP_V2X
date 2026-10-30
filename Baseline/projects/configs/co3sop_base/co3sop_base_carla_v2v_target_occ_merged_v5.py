# 任务48：v5 数据集 = 按设计值（28 GHz / 1584 子载波 / 120 kHz）重算的波束标签（v4 实际是 24 GHz，见实验记录任务47）。
# 只换 beam_csv_path，其余与 merged_v4 相同。占用 GT、目标占用与频率无关，不变；
# 这里只用来让数据集按 v5 的标签过滤"无传播路径"的样本，以及给 occ_beam_dataset / loro_cv 构造划分。
_base_ = ['./co3sop_base_carla_v2v_target_occ_merged_v4.py']

beam_csv_path = '/home/admin0/carla_V2V/channel_analysis/dataset_full_merged_20261030_28ghz.csv'
data = dict(
    train=dict(beam_csv_path=beam_csv_path),
    val=dict(beam_csv_path=beam_csv_path),
    test=dict(beam_csv_path=beam_csv_path),
)
