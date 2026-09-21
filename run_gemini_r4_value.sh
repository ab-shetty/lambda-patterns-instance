#!/usr/bin/env bash
# Is the 4th Gemini round worth anything on its own terms?
#
# r4 (+88 plans) was tested once, inside mix6676_with_r4, and came back null at
# 2 seeds: 0.7366 vs mix5092's 0.7594 (startup.md:518). That test cannot say
# whether r4 is BAD data or merely REDUNDANT with the 194 sources already in the
# mix. Gemini-only isolates it: if r234 beats r23 here, r4 carries signal and the
# mix was saturated; if it does not, r4 is not worth labelling more of.
#
# Also the first per-source number ever measured on swin_t. RefUNet references on
# this exact pool at 1280: 0.5933 +- 0.0170 (9 ep, 3 seeds), 0.6688 +- 0.0350
# (24 ep, 2 seeds). 20 epochs sits between them.
#
# STEP-BUDGET CONFOUND, stated not hidden: r23 is 1,440 records and r234 is
# 3,024, so at equal epochs the r234 arm gets 2.1x the gradient steps. A win is
# therefore "r4 plus the compute it brings", not r4 alone. Run as specified.
set -uo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.

# wait for the r234 strong18 build to land (started separately)
for i in $(seq 1 120); do
  n=$(ls data/roboflow/floz-gen-gemini-r234-strong18/images 2>/dev/null | wc -l)
  [ "$n" -ge 3024 ] && break
  sleep 15
done
echo "r23=$(ls data/roboflow/floz-gen-gemini-r23-strong18/images | wc -l) records, r234=$(ls data/roboflow/floz-gen-gemini-r234-strong18/images | wc -l) records"

COMMON="--epochs 20 --schedule-epochs 20 \
  --batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 --model swin_t \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --seed 31 --mask-thresh 0.35"

for t in r23 r234; do
  CK=data/runs/ck_gem_${t}_swint_e20
  [ -f "$CK/metrics_epoch_19.json" ] && { echo "== $t done"; continue; }
  echo "== $t  $(date '+%H:%M:%S')"
  python3 scripts/train_refunet.py \
    --local-data "data/roboflow/floz-gen-gemini-${t}-strong18" \
    --checkpoint-dir "$CK" $COMMON > "logs/gem_${t}_swint_e20.log" 2>&1
done

echo "== val-selected (real validation complement, never HF14)"
python3 scripts/select_epoch_on_val.py --runs data/runs/ck_gem_r23_swint_e20 data/runs/ck_gem_r234_swint_e20 \
  --image-max-size 1280 --mask-thresh 0.35 \
  --out data/evaluations/val_gem_r4_value.json > logs/val_gem_r4_value.log 2>&1
cat data/evaluations/val_gem_r4_value.json
echo "GEMINI R4 VALUE DONE $(date '+%H:%M:%S')"
