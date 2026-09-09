# Stage 2B Phase A (doc 2B.10): freeze everything except the new
# role_embedding/film_proj/target_occ, short warm-up so the fresh target
# branch doesn't destabilize the pretrained scene branch with large initial
# gradients before Phase B unfreezes it.
_base_ = ['./co3sop_base_carla_v2v_target_occ.py']

model = dict(pts_bbox_head=dict(freeze_scene_branch=True))

total_epochs = 4
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
