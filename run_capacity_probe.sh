#!/usr/bin/env bash
# Is the model capacity-bound? (2026-09-18)
#
# On 100,000 single-pass plans RefUNet scored 0.837 train / 0.798 fresh-synthetic
# / 0.713 HF14. A train/fresh gap of only 0.039 means it is NOT memorising -- it
# cannot fit its own training distribution. That is the signature of too little
# capacity (or too little optimisation), not too little data, and it contradicts
# `startup.md`'s "fitting is not the constraint", which was measured on 3-8k
# records where val turned over.
#
# At --width 128 the ResNet50 backbone is ~25.6M of the 28.0M parameters, so the
# reference-conditioned decoder is only ~2.4M. This sweeps that:
#   width 128 -> 28.0M total (~2.4M decoder)
#   width 256 -> 39.6M total (~14M decoder)
#   width 384 -> 58.3M total (~33M decoder)
#
# Run at 1024, one epoch, on the 100k pool: the question is whether more
# capacity FITS BETTER, so resolution is deliberately held out of it and kept
# cheap. Reports train IoU (from the dice term) and fresh-synthetic IoU.
set -euo pipefail
POOL=data/synthetic/v6d_100000
SEED=${1:-7}
mkdir -p logs data/evaluations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for W in 128 256 384; do
  CK=data/runs/ck_cap_w${W}_res1024_seed${SEED}
  TAG=$(basename "$CK")
  if [ ! -f "$CK/epoch_0.pth" ]; then
    PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$POOL" \
      --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 1 \
      --batch-size 8 --num-workers 32 --prefetch-factor 6 \
      --image-max-size 1024 --ref-size 224 --width "$W" \
      --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.995 \
      --domain-random --mask-thresh 0.35 --seed "$SEED" --compile --pad-grid 512 \
      > "logs/$TAG.log" 2>&1
  fi
  echo "== width $W"
  tr '\r' '\n' < "logs/$TAG.log" | grep -oE "bce=[0-9.]+, dice=[0-9.]+" | tail -1
  PYTHONPATH=. python3 scripts/fresh_synth_iou.py --checkpoint "$CK/epoch_0.pth" \
    --image-max-size 1024 --n 300 2>&1 | tail -1
done
