#!/usr/bin/env bash
# Does the EXISTING architecture reach 0.95 hard train IoU at 10,000 plans?
#
# The 2026-09-20 probe hit 0.9866 train IoU -- on 200 images, frozen backbone,
# 1024 px, 6,000 steps. That is not a statement about 10,000 images, and the
# same day's unfrozen run on 1,995 plans at 2048 with the documented 9-epoch
# budget measured only 0.6811. So the claim "fit is solved, it just needs
# steps" is untested at scale. This tests it.
#
# Three arms at 1024 px, existing RefUNet/ResNet50, backbone TRAINING:
#            7,500 steps        37,500 steps
#   2,000 |  A (30 epochs)     D (150 epochs)
#  10,000 |  B ( 6 epochs)     C ( 30 epochs)
#
# A->B isolates POOL SIZE at a fixed step budget; B->C isolates STEP BUDGET at
# a fixed pool. D is the direct scale-up of the claim this tests: the probe's
# 0.9866 came from 200 images at 6,000 steps, so "small pool, large step
# budget" is the cell where 0.95 should appear if it appears anywhere. Without
# D, a low C cannot distinguish "10,000 plans is too many to fit" from "37,500
# steps is too few".
#
# Every arm completes its OWN cosine anneal (--schedule-epochs == --epochs), so
# B is not merely C stopped early. Fit is read with the HARD threshold via fresh_synth_iou.py
# --seen, never the soft-dice number in the training log.
set -euo pipefail
SEED=${1:-7}
SIZE=${2:-1024}
mkdir -p logs data/evaluations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {  # name pool epochs
  local NAME=$1 POOL=$2 EP=$3
  local CK=data/runs/ck_fit_${NAME}_res${SIZE}_seed${SEED}
  [ -f "$CK/epoch_$((EP-1)).pth" ] && { echo "== $NAME already trained"; return; }
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$POOL" \
    --checkpoint-dir "$CK" --epochs "$EP" --schedule-epochs "$EP" \
    --batch-size 8 --num-workers 32 --prefetch-factor 6 \
    --image-max-size $SIZE --ref-size 224 --width 128 --model unet \
    --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.995 \
    --domain-random --mask-thresh 0.35 --seed "$SEED" --pad-grid 512 \
    > "logs/fit_${NAME}.log" 2>&1
  echo "== $NAME trained"
}

run C10k_30ep data/synthetic/v6d_10000    30
run A2k_30ep  data/synthetic/v6d_train2000 30
run B10k_6ep  data/synthetic/v6d_10000     6
run D2k_150ep data/synthetic/v6d_train2000 150
