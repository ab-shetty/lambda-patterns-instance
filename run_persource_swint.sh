#!/usr/bin/env bash
# What is one plan from each source worth -- to SWIN_T?
#
# The existing per-source protocol (scripts/run_persource_value.sh) was RefUNet
# only. Every conclusion about which pool to invest labelling hours in rests on
# a CNN, and tonight established the two backbones respond to data completely
# differently: real data is worth +0.1235 to RefUNet and +0.0337 to swin_t, and
# HF14 RISES with synthetic training for RefUNet (0.5993->0.6513) while it FALLS
# for swin_t (0.7484->0.6913). A per-source ranking measured on RefUNet may not
# be swin_t's ranking at all.
#
# MATCHED DESIGN. 86 real vs 28 genreal vs 80 gemini confounds source count with
# source content, so every arm is exactly 28 unique sources expanded to 504
# strong18 records (17 augs + original). Three independent 28-draws from the real
# pool bound draw noise; one draw each elsewhere (genreal has only 28 total).
#
# RESOLUTION. Native long sides differ hugely: real 640, genreal 1536, gemini
# 2752, v6d 3779. Training is at 1280, so the three high-res pools are capped
# there during augmentation -- lossless for this run, and it keeps v6d's 7.7MP
# sheets from dominating disk and build time. Real is 640 native and untouched.
# NOTE this means the arms are NOT resolution-matched; they are matched on what
# the 1280 dataloader can actually see, which is the product-relevant question.
#
# strong18 does less work here than in a mix -- its replication count sets pool
# weighting, and there is no mixture to weight. What remains is the augmentation
# content (rotation/flip/polygon/blur/noise, disjoint from --domain-random's
# downscale+brightness) and an 18x longer epoch. Identical across arms, so it
# cannot bias the ranking.
#
# Recipe is run_persource_value.sh's verbatim, with --model swin_t added.
# RefUNet reference on this protocol: gen504 native = 0.2650.
set -uo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.

COMMON="--epochs 10 --schedule-epochs 10 \
  --batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 --model swin_t \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --seed 31 --mask-thresh 0.35"

for t in real28_s1 real28_s2 real28_s3 genreal28 gemini28 v6d28; do
  CK=data/runs/ck_ps_swint_${t}
  [ -f "$CK/metrics_epoch_9.json" ] && { echo "== $t done"; continue; }
  echo "== $t  $(date '+%H:%M:%S')"
  python3 scripts/train_refunet.py --local-data "data/ps/${t}_strong18" \
    --checkpoint-dir "$CK" $COMMON > "logs/ps_swint_${t}.log" 2>&1
done

echo "== val-selected (real validation complement, never HF14)"
python3 scripts/select_epoch_on_val.py --runs data/runs/ck_ps_swint_* \
  --image-max-size 1280 --mask-thresh 0.35 \
  --out data/evaluations/val_ps_swint.json > logs/val_ps_swint.log 2>&1
cat data/evaluations/val_ps_swint.json
echo "PERSOURCE SWINT DONE $(date '+%H:%M:%S')"
