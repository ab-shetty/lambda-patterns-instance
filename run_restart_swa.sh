#!/usr/bin/env bash
# Longer training for the best recorded model, run to completion this time.
#
# abshetty/floz-refunet-swint-mixr4-e8 (HF14 0.7887) was val-selected at its
# FINAL epoch with HF14 still rising. One fresh cosine restart from it (reset
# optimizer, 8 epochs of a planned 9 -- the documented second phase), on its own
# training mix. Then, on VALIDATION ONLY, choose among:
#   * the published e8 and every restart epoch (single checkpoints), and
#   * weight averages of the restart's late epochs (SWA, the 2026-09-07
#     protocol), windows 12-15, 13-15, 14-15,
# each at inference 2048 and 4096 -- and read HF14 once for the choice.
# Averaging is done here (not average_checkpoints.py) so no HF14 number exists
# for any candidate but the chosen one.
#
# Chain further restarts by pointing BASE at a previous choice and giving the
# restart its own CK; START is the first epoch number the restart will save
# (the base checkpoint's epoch + 1), SIZES the inference sizes to score.
#
#   ./run_restart_swa.sh        # ~1.5 h on a GH200
#   BASE=data/runs/ck_swint_mixr4_restart_full/epoch_13.pth START=14 SIZES=4096 \
#     CK=data/runs/ck_swint_mixr4_restart2 ./run_restart_swa.sh
set -euo pipefail
SIZE=2048
DATA=${DATA:-data/mixed/v6dmix_plus_r4}
BASE=${BASE:-data/runs/ck_pub_swint_mixr4/epoch_8.pth}
CK=${CK:-data/runs/ck_swint_mixr4_restart_full}
START=${START:-8}
SIZES=${SIZES:-2048 4096}
LAST=$((START + 7))
TAG=$(basename $CK)
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
mkdir -p logs data/evaluations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f $CK/epoch_$LAST.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $DATA \
  --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 \
  --batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size $SIZE --ref-size 224 --width 128 --model swin_t \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed 7 --pad-grid 512 \
  --reset-optimizer --init-from $BASE > logs/${TAG}.log 2>&1
echo "== trained"

PYTHONPATH=. python3 - "$BASE" "$CK" "$VAL" "$TAG" "$START" "$SIZES" <<'PY' | tee logs/${TAG}_val.txt
import json, os, subprocess, sys
import numpy as np, torch
base, ck, val, tag, start, sizes = sys.argv[1:]
start, sizes = int(start), [int(x) for x in sizes.split()]
last = start + 7
cands = {"base": base}
cands.update({f"restart_e{e}": f"{ck}/epoch_{e}.pth" for e in range(start, last + 1)})
for lo in (last - 3, last - 2, last - 1):
    out = f"{ck}/swa_{lo}-{last}.pth"
    if not os.path.exists(out):
        sds = [torch.load(f"{ck}/epoch_{e}.pth", map_location="cpu") for e in range(lo, last + 1)]
        avg = {k: (sum(s["model"][k].float() for s in sds) / len(sds)).to(sds[-1]["model"][k].dtype)
               if sds[-1]["model"][k].is_floating_point() else sds[-1]["model"][k]
               for k in sds[-1]["model"]}
        torch.save({**sds[-1], "model": avg}, out)
    cands[f"swa_{lo}-{last}"] = out
res = {}
for name, path in cands.items():
    for R in sizes:
        out = f"data/evaluations/{tag}_{name}_val{R}.json"
        if not os.path.exists(out):
            subprocess.run(["python3", "scripts/evaluate_refunet_selection.py", "--checkpoint", path,
                            "--indices", val, "--image-max-size", str(R), "--ref-size", "224",
                            "--mask-thresh", "0.35", "--metrics-out", out],
                           env=dict(os.environ, PYTHONPATH="."), capture_output=True, check=True)
        d = json.load(open(out))
        res[(name, R)] = float(np.mean([r["iou"] for r in d.get("selections", d.get("rows"))]))
        print(f"{name:14s} infer {R}: val {res[(name, R)]:.4f}", flush=True)
best = max(res, key=res.get)
print(f"CHOICE {cands[best[0]]} {best[1]} {best[0]} {res[best]:.4f}")
PY
read -r _ CK_CHOICE R_CHOICE NAME _ < <(grep '^CHOICE' logs/${TAG}_val.txt)
echo "== HF14, read once: $NAME @ $R_CHOICE"
PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK_CHOICE \
  --indices $HF14 --image-max-size $R_CHOICE --ref-size 224 --mask-thresh 0.35 \
  --metrics-out data/evaluations/${TAG}_choice_hf14.json | tail -2
python3 scripts/paired_compare.py --a data/evaluations/pub_swint_mixr4_e8_hf14.json \
  --b data/evaluations/${TAG}_choice_hf14.json --label-a published_e8@2048 --label-b choice \
  | sed -n '/mean  /p;/paired mean/,/sign-test/p'
