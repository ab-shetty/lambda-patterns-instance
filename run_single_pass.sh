#!/usr/bin/env bash
# The single18k recipe (synth_progress.md 2026-10-14, "18,000 fresh sheets seen once"), scripted:
# N fresh sheets at the current generator defaults (ids START+, seed 6), two ONE-epoch restarts of INIT on
# them -- arm dr with --domain-random --dr-scale-min 0.7, arm nodr without (flips / rot90 always on) --
# then the float 50/50 weight blend of the two, scored @4096 with hand boxes on HF14 / val / Gemini30.
#   TAG   name of the run (pool data/synthetic/${TAG}_s$START, checkpoints data/runs/ck_${TAG}_{dr,nodr,blend50})
#   GEN   extra generator flags (default: only the Swiss layouts)
#   INIT  checkpoint to restart from (default longswiss-swa37-41, fetched with scripts/hf_ckpt_to_pth.py)
#   TAG=single18kf ./run_single_pass.sh        # ~8 min pool, ~20 min per arm, ~25 min scoring (GH200)
# The arms run one after the other: two at 2048 need ~2 x 47 GB and OOM on the 94.5 GB GH200.
set -euo pipefail
set -a; . ~/.env; set +a      # HF_TOKEN: train_refunet's per-epoch real-sheet check
. scripts/pool_guard.sh
TAG=${TAG:?set TAG}; N=${N:-18000}; START=${START:-600000}; SEED=${SEED:-7}
INIT=${INIT:-data/runs/ck_best/longswiss-swa37-41.pth}
GEN=${GEN:---plan-source data/reference/swiss_dwellings/layouts.pkl.gz}
POOL=data/synthetic/${TAG}_s$START
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
G=data/eval_pools/gemini30_s20261006
O=data/evaluations/$TAG; mkdir -p logs $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

need_pool $POOL && python3 scripts/generate_synthetic_fc.py --out $POOL --n $N --start $START --seed 6 \
  --revit --revit-plans --shaped --workers 64 --mode-weights 66,16,18 $GEN > logs/gen_${TAG}.log 2>&1
echo "== pool $(ls $POOL/annotations | wc -l) sheets"

for ARM in dr nodr; do
  CK=data/runs/ck_${TAG}_$ARM
  [ -f $CK/.done ] && continue
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data $POOL --checkpoint-dir $CK \
      --epochs 1 --batch-size 8 --num-workers 16 --prefetch-factor 4 \
      --image-max-size 2048 --ref-size 224 --width 128 --model swin_t --lr 2e-4 --backbone-lr-mult 0.1 \
      --train-split 0.99 $([ $ARM = dr ] && echo --domain-random --dr-scale-min 0.7) --mask-thresh 0.35 \
      --seed $SEED --pad-grid 512 --reset-optimizer --init-from $INIT --roi-ref --roi-ref-mode add \
      > logs/${TAG}_$ARM.log 2>&1 || { echo "arm $ARM failed: logs/${TAG}_$ARM.log"; exit 1; }
  touch $CK/.done
done
echo "== trained"

B=data/runs/ck_${TAG}_blend50.pth
[ -f $B ] || python3 - data/runs/ck_${TAG}_dr/epoch_0.pth data/runs/ck_${TAG}_nodr/epoch_0.pth $B <<'PY'
import sys, torch
a, b = (torch.load(p, map_location="cpu") for p in sys.argv[1:3])
avg = {k: ((a["model"][k].float() + b["model"][k].float()) / 2).to(a["model"][k].dtype)
       if a["model"][k].is_floating_point() else a["model"][k] for k in a["model"]}
torch.save({**a, "model": avg, "epoch": None, "actual_epoch": None}, sys.argv[3])
PY

# Four scoring jobs side by side on the one GPU (each is a few GB at 4096).
for job in "hf14:--indices $HF14" "hf14fix2:--indices $HF14 --label-fixes eval_labels/hf14_fixes_v2.json" \
           "val:--indices $VAL" "gemini30:--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json"; do
  OUT=$O/blend50_${job%%:*}_4096.json
  [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $B ${job#*:} \
    --image-max-size 4096 --ref-size 224 --mask-thresh 0.35 --metrics-out $OUT > $O/${job%%:*}.log 2>&1 &
done
wait
echo "== scored: $O"
