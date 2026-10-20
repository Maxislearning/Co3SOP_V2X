#!/bin/bash
# 用法: tools/sweep_v4_run_lane.sh <source>   依次训练+评估该source的三种channels（v4标签，A2配置与v3 sweep一致）
source /home/admin0/anaconda3/etc/profile.d/conda.sh; conda activate flashocc
cd /home/admin0/Co3SOP/Baseline
S=$1
for C in scene target scene_target; do
  W=work_dirs/occ_beam_${S}_${C}_a2_merged_v4
  if [ -f $W/eval.log ] && grep -q "RX ==========" $W/eval.log; then echo "[sweep] skip $S/$C (done)"; continue; fi
  echo "[sweep] $(date +%F_%T) train $S/$C"
  python tools/train_occ_beam.py --source $S --channels $C --architecture hierarchical --loss-weighting sqrt_inv_freq \
      --epochs 30 --patience 10 --work-dir $W > $W.train.log 2>&1 || { echo "[sweep] TRAIN FAILED $S/$C"; continue; }
  python tools/full_eval_occ_beam.py --source $S --channels $C --architecture hierarchical --work-dir $W > $W/eval.log 2>&1 \
      && echo "[sweep] $(date +%F_%T) done $S/$C" || echo "[sweep] EVAL FAILED $S/$C"
done
echo "[sweep] lane $S ALL DONE"
