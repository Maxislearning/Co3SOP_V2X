# Stage 2C-v2: same as C1-v1 (co3sop_base_carla_v2v_temporal_c1.py) in every
# respect -- dataset, GT, optimizer, schedule, Stage 2B checkpoint init --
# except pts_bbox_head.type, which swaps v1's naive-concat temporal fusion
# for GatedTemporalTargetAwareOccHead's current-centric gated residual
# fusion (temporal_target_occ_head_v2.py). No lambda_aux here at all (v1's
# aux branch, off by default, isn't even built in v2's head class).
_base_ = ['./co3sop_base_carla_v2v_temporal_c1.py']

model = dict(pts_bbox_head=dict(type='GatedTemporalTargetAwareOccHead'))

work_dir = 'work_dirs/carla_v2v_temporal_c1_v2'
