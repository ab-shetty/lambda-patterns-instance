#!/usr/bin/env bash
# Can the swin_t RefSwinUNet fit 64 Revit plans to the ~0.95 resolution
# ceiling at all? If it can't even memorise 64 plans (HF14 protocol, every
# family asked), the ~0.78 own-plan wall is structural, not data or steps.
# One 60-epoch cosine (1,920 steps of 2 plans x 6 questions), no per-epoch
# HF14 eval. Arms (env ARMS): base (documented recipe), bb1
# (--backbone-lr-mult 1.0), nodr (no --domain-random).
#   ARMS="base bb1" ./run_overfit64.sh
set -euo pipefail
POOL=${POOL:-data/synthetic/revit_sub64}
EP=${EP:-60}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs data/evaluations/overfit64
COMMON="--batch-size 2 --refs-per-image 6 --num-workers 12 --prefetch-factor 4 \
  --image-max-size 2048 --ref-size 224 --width 128 --model swin_t --lr 2e-4 \
  --train-split 0.99 --mask-thresh 0.35 --seed 7 --pad-grid 512 --real-indices ''"
declare -A EXTRA=([base]="--backbone-lr-mult 0.1 --domain-random"
                  [bb1]="--backbone-lr-mult 1.0 --domain-random"
                  [nodr]="--backbone-lr-mult 0.1"
                  [bb1nodr]="--backbone-lr-mult 1.0"
                  [roi]="--backbone-lr-mult 0.1 --domain-random --roi-ref"
                  [roinodr]="--backbone-lr-mult 0.1 --roi-ref"
                  [roiadd]="--backbone-lr-mult 0.1 --domain-random --roi-ref --roi-ref-mode add"
                  [roiaddnodr]="--backbone-lr-mult 0.1 --roi-ref --roi-ref-mode add"
                  [selfattn]="--backbone-lr-mult 0.1 --domain-random --swin-decoder selfattn"
                  [roiaddnodr_w256]="--backbone-lr-mult 0.1 --roi-ref --roi-ref-mode add --width 256"
                  [roiaddnodr_swins]="--backbone-lr-mult 0.1 --roi-ref --roi-ref-mode add --model swin_s"
                  [roiaddnodr_swinb]="--backbone-lr-mult 0.1 --roi-ref --roi-ref-mode add --model swin_b")
# 2026-10-03 capacity arms (later --width / --model override COMMON's): can a bigger decoder or
# backbone memorise 64 MINED hard sheets past the ~0.90 the 09-28 arms reached on random ones?
#   POOL=data/synthetic/revitfail_mined64 SUF=_mined64 \
#     ARMS="roiaddnodr roiaddnodr_w256 roiaddnodr_swins roiaddnodr_swinb" ./run_overfit64.sh
for A in ${ARMS:-base bb1}; do (
  CK=data/runs/ck_overfit64_${A}${SUF:-}
  [ -f $CK/epoch_$((EP-1)).pth ] || eval PYTHONPATH=. python3 scripts/train_refunet.py --local-data $POOL \
    --checkpoint-dir $CK --epochs $EP $COMMON ${EXTRA[$A]} > logs/overfit64_${A}${SUF:-}.log 2>&1
  PYTHONPATH=. python3 scripts/fit_diagnose.py --checkpoint $CK/epoch_$((EP-1)).pth --pool $POOL \
    --train-only --n-images 64 --out data/evaluations/overfit64/${A}${SUF:-}.json | sed "s/^/$A${SUF:-}: /"
) & done; wait
