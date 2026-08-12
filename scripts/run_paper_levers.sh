#!/usr/bin/env bash
# Four architecture levers taken from the reference-matching literature, each
# screened against the documented mix3652 recipe.
#
#   self    self-support prototype refinement (Fan et al., via Catalano &
#           Matteucci's FSS review sec. 3.2): pool the first pass's confident
#           QUERY pixels into the prototype and decode again.
#   dynamic hypernetwork / dynamic-filter conditioning (Hossler, CS231n 2017
#           "Where's Waldo?"; FSS review sec. 3.1 conditional networks): generate
#           the final 1x1 classifier from the reference vector.
#
# Two further levers were screened and REMOVED, so they no longer have arms here:
# a central-surround two-stream reference (Zagoruyko & Komodakis 2015) at 0.7173
# and SimAM shrinkage attention (Remote Sensing 2024, 16, 2831) at 0.7129,
# against a 0.7314 baseline at the same seed. See synth_progress.md (2026-08-12).
#
# Usage: run_paper_levers.sh <arm> <seed> [num_workers]
#
# SCREENED ON THE VALIDATION SPLIT. `--real-indices` is the validation
# complement, never HF14, so the per-epoch diagnostic that picks the epoch
# cannot see the acceptance set. Confirm a survivor on HF14 once, at three
# seeds, with scripts/select_epoch_on_val.py.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

ARM="${1:?arm: base|self|dynamic}"
SEED="${2:?seed}"
WORKERS="${3:-16}"
VALIDATION="4,5,6,8,9,10,13,15,17,19,20,21,22,26"

case "$ARM" in
  base)     EXTRA="" ;;
  self)     EXTRA="--self-support 0.5 --self-support-thresh 0.7 --self-support-aux 0.5" ;;
  dynamic)  EXTRA="--dynamic-filter" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

CK="data/runs/ck_lever_${ARM}_s${SEED}"
COMMON="--batch-size 8 --num-workers ${WORKERS} --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed ${SEED} \
  --real-indices ${VALIDATION} ${EXTRA}"

python3 scripts/train_refunet.py --local-data data/mixed/toparea1600_rf1548_gen504 \
  --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 $COMMON \
  > "logs_lever_${ARM}_s${SEED}_A.log" 2>&1

python3 scripts/train_refunet.py --local-data data/mixed/toparea1600_rf1548_gen504 \
  --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from "$CK/epoch_0.pth" \
  > "logs_lever_${ARM}_s${SEED}_B.log" 2>&1

echo "DONE ${ARM} seed ${SEED}"
