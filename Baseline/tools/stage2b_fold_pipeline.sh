#!/bin/bash
# 任务38 用法: tools/stage2b_fold_pipeline.sh <fold>    （fold = legacy / seed1 ... seed5）
# 留一个 run 交叉验证的一折，按任务37 seed5 的流程顺序跑：
#   1) 按折重训 Stage2B（dist_train.sh，单卡分布式，否则验证会走 mmdet 默认 EvalHook 报错）
#   2) 按内部验证 target IoU 选 epoch（训练中没跑完验证的 epoch 用 test.py 补算）
#   3) 用选出的 checkpoint 提取占用到 beam_occ_native_loro_<fold>
#   4) 用新占用训练 pred/target、pred/scene_target 两个 beam 头（loro_lane.sh，tag s2b_<fold>）
# 每一步都可以断点续跑：已经完成的步骤跳过，训练从 latest.pth 接着跑。任何一步失败就退出，不重试。
set -e
F=$1
[ -n "$F" ] || { echo "usage: $0 <fold>"; exit 1; }
source /home/admin0/anaconda3/etc/profile.d/conda.sh; conda activate flashocc
cd /home/admin0/Co3SOP/Baseline
export PYTHONPATH=$PWD:$PYTHONPATH
C=projects/configs/co3sop_base/co3sop_base_carla_v2v_target_occ_loro_$F.py
W=work_dirs/stage2b_loro_$F
O=/home/admin0/carla_V2V/occupancy_prediction_beam/beam_occ_native_loro_$F
E=$(python -c "from mmcv import Config; print(Config.fromfile('$C').total_epochs)")
mkdir -p $W
log() { echo "[fold $F] $(date +%F_%T) $*"; }

# 1) Stage2B
if [ ! -f $W/epoch_$E.pth ]; then
  RESUME=""
  [ -e $W/latest.pth ] && RESUME="--resume-from $W/latest.pth"
  log "train Stage2B $RESUME"
  PORT=28107 bash tools/dist_train.sh $C 1 $W $RESUME > $W/train_stdout_$(date +%m%d_%H%M).log 2>&1
fi
[ -f $W/epoch_$E.pth ] || { log "Stage2B 没有训练到 epoch $E"; exit 1; }

# 2) 选 epoch：训练日志里的验证结果 + 缺失的用 test.py 补
val_iou() {  # 打印 "epoch iou"，来自所有 *.log.json 的 val 记录
  python - $W <<'EOF'
import glob, json, sys
best = {}
for p in glob.glob(f'{sys.argv[1]}/*.log.json'):
    for line in open(p):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get('mode') == 'val' and 'target_iou' in r:
            best[r['epoch']] = r['target_iou']
for p in glob.glob(f'{sys.argv[1]}/val_epoch*.log'):
    e = int(p.split('val_epoch')[1].split('.')[0])
    txt = open(p, errors='ignore').read()
    if "'target_iou': " in txt:
        best[e] = float(txt.split("'target_iou': ")[-1].split(',')[0].split('}')[0])
for e in sorted(best):
    print(e, best[e])
EOF
}
for e in $(seq 1 $E); do
  if ! val_iou | grep -q "^$e "; then
    log "补算 epoch $e 验证"
    torchrun --nproc_per_node=1 --master_port=29282 tools/test.py $C $W/epoch_$e.pth --launcher pytorch \
        --deterministic --eval bbox --cfg-options data.test.ann_file=val data.workers_per_gpu=1 > $W/val_epoch$e.log 2>&1
  fi
done
val_iou > $W/val_target_iou.txt
BEST=$(sort -k2 -gr $W/val_target_iou.txt | head -1 | cut -d' ' -f1)
log "内部验证 target IoU: $(tr '\n' ' ' < $W/val_target_iou.txt) -> 选 epoch $BEST"
echo $BEST > $W/best_epoch.txt

# 3) 提取（脚本自带跳过已完成的条目）
if ! grep -q "epoch_$BEST.pth" $O/dataset_version.json 2>/dev/null; then
  log "extract with epoch_$BEST"
  mkdir -p $O
  python tools/extract_stage2b_occ_probs.py --config projects/configs/co3sop_base/co3sop_base_carla_v2v_target_occ_merged_v4.py \
      --beam-csv /home/admin0/carla_V2V/channel_analysis/dataset_full_merged_20260929_regen.csv \
      --out-dir $O --ckpt $W/epoch_$BEST.pth > $W/extract_$(date +%m%d_%H%M).log 2>&1
fi
grep -q "epoch_$BEST.pth" $O/dataset_version.json || { log "提取没完成"; exit 1; }

# 4) beam 头
log "beam heads"
FOLDS=$F bash tools/loro_lane.sh pred 0 "--coord-conv gated --occ-dir $O" s2b_$F
for c in target scene_target; do
  [ -f work_dirs/loro_cv/s2b_$F/logits/pred_${c}_${F}_s0.npz ] || { log "beam $c 没完成"; exit 1; }
done
log "done"
