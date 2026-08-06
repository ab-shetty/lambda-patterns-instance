#!/usr/bin/env bash
# Dense reference correlation ("learned template matching").
# The reference was collapsed to one global-average vector per scale, discarding
# stroke orientation/spacing/arrangement -- exactly what separates one hatch from
# another. Region identification (not boundary placement) is what limits the
# model: boundary refinement is worth ~0 on validation, while per-image IoU spans
# 0.42-0.98. corr_grid keeps the reference as a GxG token grid and cosine-matches
# every image location against all tokens.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
D=data/mixed/toparea1600_rf1548_gen504
for g in 4 8; do
  ck="data/runs/ck_corr${g}_seed31"
  BASE="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
    --image-max-size 1280 --ref-size 224 --width 128 --corr-grid $g \
    --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
    --domain-random --mask-thresh 0.35 --seed 31"
  python3 scripts/train_refunet.py --local-data $D --checkpoint-dir "$ck" \
    --epochs 1 --schedule-epochs 10 $BASE > "logs_corr${g}_A.log" 2>&1
  python3 scripts/train_refunet.py --local-data $D --checkpoint-dir "$ck" \
    --epochs 8 --schedule-epochs 9 $BASE --reset-optimizer \
    --init-from "$ck/epoch_0.pth" > "logs_corr${g}_B.log" 2>&1
done
echo "CORR DONE"
