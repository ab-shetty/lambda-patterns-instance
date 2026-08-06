#!/usr/bin/env bash
# Remaining data/appearance levers after resolution and ink-density both failed.
#   realism : --realism-aug degrades synth toward real PDF-export appearance.
#             The faint-sheet failure is an appearance gap and thresholding
#             cannot fix it (raising the threshold monotonically hurts), so the
#             representation has to change.
#   vol3200 : double the synthetic half, unchanged selection.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
BASE="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed 31"

run () {  # tag, data, extra
  ck="data/runs/ck_$1_seed31"
  python3 scripts/train_refunet.py --local-data "$2" --checkpoint-dir "$ck" \
    --epochs 1 --schedule-epochs 10 $BASE $3 > "logs_$1_A.log" 2>&1
  python3 scripts/train_refunet.py --local-data "$2" --checkpoint-dir "$ck" \
    --epochs 8 --schedule-epochs 9 $BASE $3 --reset-optimizer \
    --init-from "$ck/epoch_0.pth" > "logs_$1_B.log" 2>&1
}
run realism data/mixed/toparea1600_rf1548_gen504 "--realism-aug"
run vol3200 data/mixed/toparea3200_rf1548_gen504 ""
echo "BATCH3 DONE"
