# 2x2 localization ablation, cell (geometry=replaced by localization, visual=dropped).
# The strictest cell: beam head only ever sees the occupancy-derived TX
# localization embedding (see beam_head.py's loss() docstring for the caveat
# that the localization module itself still uses GPS direction as a coarse
# disambiguation prior internally -- this is not a true no-GPS-at-all test).
# See /home/admin0/.claude/plans/effervescent-shimmying-turing.md.
_base_ = ['./co3sop_base_carla_v2v_beam.py']

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]

model = dict(
    pts_bbox_head=dict(
        use_visual=False,
        use_geometry=False,
        use_localization=True,
        pc_range=point_cloud_range,
        loc_hidden_dim=32,
        loc_prior_sigma_xy=15.0,
        loc_ego_exclude_radius=3.0,
        loc_loss_weight=1.0,
    )
)
