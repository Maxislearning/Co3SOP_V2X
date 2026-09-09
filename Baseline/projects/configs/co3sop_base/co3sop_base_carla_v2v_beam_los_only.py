# Train/serve beam prediction on LOS samples only. No LOS/NLOS classifier
# for now -- per user decision, assume link condition is known externally at
# serving time; this just drops NLOS from train/val so the model isn't
# fitting a beam<->geometry relationship that doesn't hold once the direct
# path is blocked (see the LOS/NLOS breakdown: exact GPS geometry clearly
# helps in LOS but not in NLOS -- plan discussion in the chat, no separate
# plan file for this one, it's a small dataset-filter change).
#
# Same architecture as the plain baseline (co3sop_base_carla_v2v_beam.py:
# geo+vis, no localization module) -- that cell and geo+vis+localization
# were statistically tied in the mixed-LOS/NLOS ablation, so start from the
# simpler one.
_base_ = ['./co3sop_base_carla_v2v_beam.py']

data = dict(
    train=dict(los_only=True),
    val=dict(los_only=True),
    test=dict(los_only=True),
)
