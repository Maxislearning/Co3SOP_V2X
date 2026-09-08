_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin_carla_v2v/'

# Beam selection phase 1: TX in {vehicle_1 (TX_CAR1), vehicle_3 (TX_CAR2)},
# RX fixed = vehicle_2. Roadside TX_ROADSIDE_01..05 (no matching camera view)
# are phase 2, see 记录something/ prompt-doc discussion + plan
# effervescent-shimmying-turing.md.
#
# Reuses the same backbone/fusion architecture and volume/embed dims as
# co3sop_base_carla_v2v_fusion.py -- only pts_bbox_head, the dataset, and
# the pipeline/collect keys differ (no occupancy GT needed here).
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]

cam_num = 6
max_connect_car = 2  # ego(vehicle_2) + TX_CAR1(vehicle_1) + TX_CAR2(vehicle_3)
max_connect_range = 100
use_semantic = True
fusion_layers = 3

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = ['empty', 'road', 'vehicle']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = [192]
_ffn_dim_ = [384]
volume_h_ = [50]
volume_w_ = [50]
volume_z_ = [4]
_num_points_ = [4]
_num_layers_ = [3]

model = dict(
    type='Co3SOPBeam',
    cam_num=cam_num,
    car_num=max_connect_car + 1,
    use_grid_mask=True,
    use_semantic=use_semantic,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[512, 1024, 2048],
        out_channels=192,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
        freeze=True,
    ),
    pts_bbox_head=dict(
        type='BeamSelectionHead',
        freeze=True,
        num_tx_beams=64,
        num_rx_beams=64,
        geometry_dim=5,
        geometry_hidden_dim=64,
        beam_hidden_dim=256,
        beam_dropout=0.1,
        max_connect_car=max_connect_car,
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=3,
        conv_input=[192, 96, 96, 48, 48],
        conv_output=[96, 96, 48, 48, 48],
        out_indices=[0, 2, 4],
        upsample_strides=[1, 2, 1, 2, 1],
        embed_dims=_dim_,
        img_channels=[128, 128, 128],
        use_semantic=use_semantic,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            num_cams=cam_num,
            encoder=dict(
                type='Encoder',
                cam_num=cam_num,
                car_num=max_connect_car + 1,
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            num_cams=cam_num,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=4),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                      'ffn', 'norm', 'conv')))),
        v2v_transformer=dict(
            type="V2VFusionTransformer",
            positional_encoding=dict(
                type='SinePositionalEncoding3D',
                num_feats=_dim_[0] // 3,
                normalize=True),
            embed_dims=_dim_[0],
            num_cars=max_connect_car + 1,
            volume_h=volume_h_[0],
            volume_w=volume_w_[0],
            volume_z=volume_z_[0],
            decoder=dict(
                type='DetrTransformerEncoder',
                num_layers=fusion_layers,
                transformerlayers=dict(
                    type='OccLayer',
                    embed_dims=_dim_[0],
                    feedforward_channels=_ffn_dim_[0],
                    ffn_dropout=0.1,
                    attn_cfgs=dict(
                        type='VoxelCrossAttention',
                        pc_range=point_cloud_range,
                        num_cars=max_connect_car,
                        deformable_attention=dict(
                            type='MultiScaleDeformableAttention3D',
                            batch_first=True,
                            embed_dims=_dim_[0],
                            num_heads=4,
                            num_points=4,
                            num_levels=1),
                        embed_dims=_dim_[0],),
                    conv_num=2,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm', 'conv')),
                init_cfg=None),
        )
    ),
)

dataset_type = 'CarlaV2VBeamCo3SOP'
data_root = '/home/admin0/carla_V2V/occupancy_prediction_beam'
csv_path = '/home/admin0/data/run_20260904_142344_bigcar80/rear_vehicle_log.csv'
beam_csv_path = '/home/admin0/carla_V2V/channel_analysis/channel_summary_from_carla_20260904_145458.csv'
file_client_args = dict(backend='disk')

# Same default meta_keys CustomCollect3D ships with (transform_3d.py), plus
# tx_rx_geometry/beam_eval_meta -- overridden here rather than touching the
# shared class default, since occupancy configs still rely on that default.
beam_meta_keys = (
    'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img',
    'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx', 'pcd_scale_factor',
    'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'scene_token',
    'can_bus', 'pc_range', 'occ_size', 'occ_path', 'lidar_token', 'trans2ego',
    'vehicle_id', 'lidar2cams', 'cam_intrinsic',
    'tx_rx_geometry', 'beam_eval_meta',
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'gt_beam'], meta_keys=beam_meta_keys)
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'gt_beam'], meta_keys=beam_meta_keys)
]

find_unused_parameters = True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        csv_path=csv_path,
        beam_csv_path=beam_csv_path,
        ann_file='train',
        max_connect_car=max_connect_car,
        connect_range=max_connect_range,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        csv_path=csv_path,
        beam_csv_path=beam_csv_path,
        ann_file='validate',
        max_connect_car=max_connect_car,
        connect_range=max_connect_range,
        pipeline=test_pipeline,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        csv_path=csv_path,
        beam_csv_path=beam_csv_path,
        ann_file='validate',
        max_connect_car=max_connect_car,
        connect_range=max_connect_range,
        pipeline=test_pipeline,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=4e-5,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 12
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# Reuse the trained occupancy-fusion checkpoint: backbone + per-car encoder +
# v2v_fuse are all already trained, only the new beam_pool/geometry_mlp/
# beam_trunk/tx_beam_head/rx_beam_head layers start from scratch (Experiment
# 2 from the beam-selection prompt doc -- freeze=True freezes the per-car
# encoder via V2VOccHead._freeze_ego, v2v_fuse + new beam layers stay trainable).
load_from = 'work_dirs/carla_v2v_fusion/epoch_12.pth'

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=3)
