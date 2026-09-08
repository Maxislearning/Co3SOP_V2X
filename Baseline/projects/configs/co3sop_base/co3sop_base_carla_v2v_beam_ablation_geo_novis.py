# 2x2 localization ablation, cell (geometry=supplement, visual=dropped).
# See /home/admin0/.claude/plans/effervescent-shimmying-turing.md.
_base_ = ['./co3sop_base_carla_v2v_beam.py']

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]

model = dict(
    pts_bbox_head=dict(
        use_visual=False,
        use_geometry=True,
        use_localization=True,
        pc_range=point_cloud_range,
        loc_hidden_dim=32,
        loc_prior_sigma_xy=15.0,
        loc_ego_exclude_radius=3.0,
        loc_loss_weight=1.0,
    )
)
