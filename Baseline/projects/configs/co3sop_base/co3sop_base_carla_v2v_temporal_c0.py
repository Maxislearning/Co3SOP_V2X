# Stage 2C, C0: single-frame Camera_t -> Scene/Target Occupancy at t+1.
# Literally Stage 2B's Co3SOPTargetOcc/TargetAwareOccHead UNCHANGED -- the
# only difference from co3sop_base_carla_v2v_target_occ.py is which GT files
# get loaded (t+1 forecast state instead of t's own current-frame state) and
# the sample restriction to the t-index set that also has [t-2,t-1] history
# available (so C0 and C1 -- co3sop_base_carla_v2v_temporal_c1.py -- train on
# exactly the same samples; see plan effervescent-shimmying-turing.md §3/§4).
_base_ = ['./co3sop_base_carla_v2v_target_occ.py']

data_root = '/home/admin0/carla_V2V/occupancy_prediction_beam'
future_occ_root = '/home/admin0/carla_V2V/occupancy_prediction_beam/occupancy_gt_future_t1'
future_target_root = '/home/admin0/carla_V2V/occupancy_prediction_beam/target_occupancy_gt_future_t1'

dataset_type = 'CarlaV2VForecastCo3SOP'

data = dict(
    train=dict(type=dataset_type, future_occ_root=future_occ_root, target_occ_root=future_target_root),
    val=dict(type=dataset_type, future_occ_root=future_occ_root, target_occ_root=future_target_root),
    test=dict(type=dataset_type, future_occ_root=future_occ_root, target_occ_root=future_target_root),
)

load_from = 'work_dirs/carla_v2v_target_occ_phaseB/epoch_12.pth'
work_dir = 'work_dirs/carla_v2v_temporal_c0'
