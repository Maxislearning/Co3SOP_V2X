#!/bin/bash
source /home/admin0/anaconda3/etc/profile.d/conda.sh; conda activate flashocc
cd /home/admin0/Co3SOP/Baseline
for c in gt_scene gt_target gt_scene_target pred_scene pred_target; do
  S=${c%%_*}; C=${c#*_}; W=work_dirs/occ_beam_${c}_a2_merged_v4
  python tools/full_eval_occ_beam.py --source $S --channels $C --architecture hierarchical --work-dir $W > $W/eval.log 2>&1 && echo "[reeval] done $c" || echo "[reeval] FAILED $c"
done
echo "[reeval] ALL DONE"
