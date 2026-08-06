#!/usr/bin/env bash
# Scale-matched reference, with and without dense correlation.
# The reference was resized to 224 regardless of its native extent while the plan
# was resized by 1280/max(h,w), so the hatch appeared a MEDIAN 5.6x larger in the
# reference than in the image (only 3% of selections within 1.5x). Every matching
# mechanism -- NCC, Gabor, and the new learned correlation -- was being asked to
# match textures across that gap, which is the likeliest reason correlation hurt.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
D=data/mixed/toparea1600_rf1548_gen504
run () {  # tag, extra
  ck="data/runs/ck_$1_seed31"
  BASE="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
    --image-max-size 1280 --ref-size 224 --width 128 \
    --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
    --domain-random --mask-thresh 0.35 --seed 31 --scale-matched-ref $2"
  python3 scripts/train_refunet.py --local-data $D --checkpoint-dir "$ck" \
    --epochs 1 --schedule-epochs 10 $BASE > "logs_$1_A.log" 2>&1
  python3 scripts/train_refunet.py --local-data $D --checkpoint-dir "$ck" \
    --epochs 8 --schedule-epochs 9 $BASE --reset-optimizer \
    --init-from "$ck/epoch_0.pth" > "logs_$1_B.log" 2>&1
}
run scaleref ""
run scaleref_corr4 "--corr-grid 4"
echo "SCALEREF DONE"
