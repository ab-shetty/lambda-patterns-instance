#!/usr/bin/env bash
# Underfitting on the 2k screen (2026-10-02): the hard-mined model fits its own mined training
# sheets no better than unseen ones (0.752 vs 0.755), so it is underfit, not memorising; the
# 2026-10-01 convergence run says steps fix that (seen 0.764 -> 0.880 from e15 to e90, fresh
# with it, over-selection 20% -> 4%). This gives a 2k-screen model one long fresh-cosine restart
# and tracks fit and transfer along it.
#   POOL  training pool (2k)                     INIT  checkpoint to restart from
#   CK    output dir                             EP    restart epochs (default 30)
#   GEN   generator flags of the pool (fresh fit sheets are drawn with them, ids 300000+)
#   SCHED planned cosine length (default EP+1, i.e. anneal to ~0); SCHED >> EP keeps the LR near
#         its peak, so a fit plateau there is capacity, not the schedule ending (plateau test)
#   EXTRA train flags (ROI etc.); DR=0.7 sets --dr-scale-min (default 0.7, the 10-01 best-val arm)
# Scored every SCORE_EVERY epochs: seen (pool sheets 0-149 = base half, 1000-1149 = mined half of a
# build_hard_pool.py pool; on a plain pool both halves are just seen sheets), fresh (150 new sheets), val + HF14 @4096 (read-only, no selection).
#   EXTRA="--roi-ref --roi-ref-mode add" POOL=data/synthetic/revitcur_hard2000 \
#     INIT=data/runs/ck_2k_revithardroi_r2048_s7/epoch_8.pth CK=data/runs/ck_long_hardcur ./run_hard_long.sh
set -euo pipefail
. scripts/pool_guard.sh     # need_pool: never silently reuse a pool from older generator defaults
EP=${EP:-30}; SCHED=${SCHED:-$((EP + 1))}; DR=${DR:-0.7}; SCORE_EVERY=${SCORE_EVERY:-10}; SEED=${SEED:-7}
TAG=$(basename $CK)
FRESH=data/synthetic/fit_fresh150_$(echo "${GEN:-none}" | md5sum | cut -c1-8)
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
O=data/evaluations/long/$TAG; mkdir -p logs $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

need_pool $FRESH && python3 scripts/generate_synthetic_fc.py --out $FRESH \
  --n 150 --start 300000 --seed 6 --revit --revit-plans --shaped --workers 64 \
  --mode-weights 66,16,18 ${GEN:-} > logs/gen_$(basename $FRESH).log 2>&1

# train_refunet numbers restart epochs from the init's epoch + 1 (e8 -> epoch_9 .. epoch_{8+EP})
[ -f $CK/.done ] || {
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data $POOL --checkpoint-dir $CK \
    --epochs $EP --schedule-epochs $SCHED --batch-size 8 --num-workers 16 --prefetch-factor 4 \
    --image-max-size 2048 --ref-size 224 --width 128 --model swin_t --lr 2e-4 --backbone-lr-mult 0.1 \
    --train-split 0.99 --domain-random --dr-scale-min $DR --mask-thresh 0.35 --seed $SEED \
    --pad-grid 512 --reset-optimizer --init-from $INIT ${EXTRA:-} > logs/long_${TAG}.log 2>&1
  touch $CK/.done; }

SEEN=$(python3 -c "print(','.join(map(str, list(range(150)) + list(range(1000, 1150)))))")
EPOCHS=$(ls $CK/epoch_*.pth | sed 's/.*epoch_//; s/.pth//' | sort -n)
FIRST=$(echo "$EPOCHS" | head -1)
for E in $EPOCHS; do
  [ $(( (E - FIRST + 1) % SCORE_EVERY )) -eq 0 ] || continue
  P=$CK/epoch_$E.pth
  for job in "seen:--local-pool $POOL --indices $SEEN --image-max-size 2048" \
             "fresh:--local-pool $FRESH --indices all --image-max-size 2048" \
             "val:--indices $VAL --image-max-size 4096" "hf14:--indices $HF14 --image-max-size 4096"; do
    OUT=$O/e${E}_${job%%:*}.json
    [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $P \
      ${job#*:} --ref-size 224 --mask-thresh 0.35 --metrics-out $OUT > /dev/null 2>&1 &
  done
  wait
done

python3 - $O <<'PY'
import json, glob, re, sys, numpy as np
o = sys.argv[1]
rows = {}
for f in glob.glob(f"{o}/e*_*.json"):
    e, s = re.match(r".*/e(\d+)_(\w+)\.json", f).groups()
    sel = json.load(open(f))["selections"]
    if s == "seen":   # base half = pool index < 1000, mined half >= 1000
        rows.setdefault(int(e), {})["seen_base"] = np.mean([r["iou"] for r in sel if r["image_index"] < 1000])
        rows[int(e)]["seen_mined"] = np.mean([r["iou"] for r in sel if r["image_index"] >= 1000])
    else:
        rows.setdefault(int(e), {})[s] = np.mean([r["iou"] for r in sel])
        if s == "val":
            rows[int(e)]["val_no17"] = np.mean([r["iou"] for r in sel if r["image_index"] != 17])
cols = ["seen_base", "seen_mined", "fresh", "val", "val_no17", "hf14"]
print("epoch  " + "  ".join(f"{c:>10s}" for c in cols))
for e in sorted(rows):
    print(f"{e:5d}  " + "  ".join(f"{rows[e].get(c, float('nan')):10.4f}" for c in cols))
PY
