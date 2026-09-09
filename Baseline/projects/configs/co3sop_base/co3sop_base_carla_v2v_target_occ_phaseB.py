# Stage 2B Phase B (doc 2B.10): unfreeze BEV/fusion/scene decoder/target
# head, joint fine-tune. load_from set to Phase A's resulting checkpoint
# once that run finishes (edit path below after Phase A completes).
_base_ = ['./co3sop_base_carla_v2v_target_occ.py']

model = dict(pts_bbox_head=dict(freeze_scene_branch=False))

load_from = 'work_dirs/carla_v2v_target_occ_phaseA/epoch_4.pth'
