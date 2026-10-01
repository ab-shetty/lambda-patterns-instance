#!/usr/bin/env bash
# Can the model fit its own training sheets past 0.90 when given enough steps? (2026-10-01)
#
# Stage A of run_fit_sweep.sh (1+15 epochs, ~1k steps on 500 sheets) ended with train and
# val loss still falling: the cosine ran out, so seen ~= fresh there says "undertrained",
# not "cannot fit". The 64-plan overfit reached 0.901 (ceiling 0.957) only without
# --domain-random. This arm: the same 500 >= 3-family sheets (fitA_train), no DR, ROI add,
# 1 + P2 epochs, every epoch kept; seen (40 training sheets) and fresh (40) scored at
# SCORE_AT epochs by family count. Waits for any running run_fit_sweep.sh first.
#
#   ./run_fit_converge.sh            env: MODEL=swin_t SEED=7 BS=4 P2=60 SCORE_AT="15 30 45 60"
#                                         DR=1 (keep --domain-random, for the paired arm)
#   Fresh-cosine restart: INIT=<ckpt> (reset optimizer, P2 epochs numbered from the
#   checkpoint's epoch + 1, own checkpoint dir with an _r<first epoch> suffix), e.g.
#   INIT=data/runs/ck_fit_conv_swin_t_nodr_s7/epoch_60.pth P2=30 SCORE_AT="70 80 90" ./run_fit_converge.sh
#   EXTRA="..." appends train flags; TAGX=_name suffixes the run's tag (use both together).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-swin_t}; SEED=${SEED:-7}; BS=${BS:-4}; P2=${P2:-60}
SCORE_AT=${SCORE_AT:-"15 30 45 60"}
DRF=""; TAG=conv_${MODEL}_nodr
[ "${DR:-0}" = 1 ] && { DRF="--domain-random"; TAG=conv_${MODEL}_dr; }
TAG=$TAG${TAGX:-}
if [ -n "${INIT:-}" ]; then
  START=$(python3 -c "import torch; c=torch.load('$INIT', map_location='cpu', weights_only=False); print(int(c.get('actual_epoch', c.get('epoch'))) + 1)")
  TAG=${TAG}_r$START
fi
S=data/synthetic; E=data/evaluations/fit_sweep; CK=data/runs/ck_fit_${TAG}_s$SEED
mkdir -p logs $E
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# -x: a bare -f pattern also matches any shell whose command line merely mentions the script
while pgrep -fx "bash \./run_fit_sweep\.sh( .*)?" >/dev/null; do sleep 60; done

COMMON="--batch-size $BS --num-workers 16 --prefetch-factor 4 --image-max-size 2048 \
  --ref-size 224 --width 128 --model $MODEL --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  $DRF --mask-thresh 0.35 --seed $SEED --pad-grid 512 --roi-ref --roi-ref-mode add ${EXTRA:-}"
if [ -n "${INIT:-}" ]; then
  LAST=$((START + P2 - 1))
  [ -f $CK/epoch_$LAST.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $S/fitA_train \
    --checkpoint-dir $CK --epochs $P2 --schedule-epochs $((P2 + 1)) $COMMON --reset-optimizer \
    --init-from $INIT > logs/fit_${TAG}.log 2>&1
else
[ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $S/fitA_train \
  --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/fit_${TAG}_p1.log 2>&1
[ -f $CK/epoch_$P2.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $S/fitA_train \
  --checkpoint-dir $CK --epochs $P2 --schedule-epochs $((P2 + 1)) $COMMON --reset-optimizer \
  --init-from $CK/epoch_0.pth > logs/fit_${TAG}_p2.log 2>&1
fi

for EP in $SCORE_AT; do
  for SET in seen:fitA_train fresh:fitA_fresh; do
    T=${TAG}_e${EP}_${SET%%:*}; OUT=$E/$T.json
    [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK/epoch_$EP.pth \
      --local-pool $S/${SET#*:} --indices $(python3 -c "print(','.join(map(str, range(40))))") \
      --image-max-size 2048 --mask-thresh 0.35 --metrics-out $OUT > logs/fit_score_$T.log 2>&1
    python3 scripts/family_breakdown.py --pool $S/${SET#*:} --metrics $OUT --label $T
  done
done
