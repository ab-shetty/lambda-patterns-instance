#!/usr/bin/env bash
# Does a synthetic A/B screened at 1024 predict the same A/B trained at 2048?
#
# Every synth-only screen (run_synth_only_screen.sh) trains at 1024 for speed,
# but resolution interacts with data effects (Gemini +0.035 @1280 vs +0.010
# @2048) and fine-texture changes (--fill-scale: ~14 px fill period at 3168,
# ~2 px after 1024 + --domain-random's 0.4x shrink) may be invisible at 1024.
# This has never been calibrated. Same two pools, same recipe and seeds, only
# the training --image-max-size differs:
#
#   v6d_1600  ids 0.. of v6d_train2000 (seed 6), first 1,600
#   r8_1600   same ids/seed with the r8 flag set (R8 in generate_synthetic_fc.py)
#
# All arms scored at epoch 8 (screen protocol, no selection), paired over the
# 77 validation + 52 HF14 selections. 1024-trained at 2048 inference (as the
# screen does); 2048-trained at 2048 and 4096. Two queues share the GPU
# (2048 bs8 peaks ~67 GiB worst case, 1024 ~17 GiB).
#   ./run_res_calibration.sh [seeds]   (default: one seed, 7)
set -euo pipefail
SEEDS=${1:-7}
declare -A POOL=([v6d]=data/synthetic/v6d_1600 [r8]=data/synthetic/r8_1600)
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
E=data/evaluations/rescal; mkdir -p logs $E
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train() {   # arm res seed
  local A=$1 R=$2 S=$3 CK=data/runs/ck_rescal_$1_r$2_s$3
  local COMMON="--batch-size 8 --num-workers 12 --prefetch-factor 4 --image-max-size $R \
    --ref-size 224 --width 128 --model swin_t --lr 2e-4 --backbone-lr-mult 0.1 \
    --train-split 0.99 --domain-random --mask-thresh 0.35 --seed $S --pad-grid 512"
  [ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data ${POOL[$A]} \
    --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/rescal_${A}_r${R}_s${S}_p1.log 2>&1
  [ -f $CK/epoch_8.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data ${POOL[$A]} \
    --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON --reset-optimizer \
    --init-from $CK/epoch_0.pth > logs/rescal_${A}_r${R}_s${S}_p2.log 2>&1
  local SIZES=2048; [ $R = 2048 ] && SIZES="2048 4096"
  for I in $SIZES; do for split in val hf14; do IDX=$VAL; [ $split = hf14 ] && IDX=$HF14
    local OUT=$E/${A}_r${R}_s${S}_i${I}_$split.json
    [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK/epoch_8.pth \
      --indices $IDX --image-max-size $I --ref-size 224 --mask-thresh 0.35 \
      --metrics-out $OUT > /dev/null 2>&1
  done; done
  echo "== done $A r$R s$S"
}

q() { local R=$1; for S in $SEEDS; do for A in v6d r8; do train $A $R $S; done; done; }
q 2048 & q 1024 & wait
PYTHONPATH=. python3 scripts/res_calibration_report.py --dir $E --seeds $SEEDS
