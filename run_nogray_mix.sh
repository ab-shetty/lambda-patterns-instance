#!/usr/bin/env bash
# The best recipe (swin_t, 282 sources, 2048, documented two-phase schedule)
# with the colour fix from the 2026-09-24 validation audit:
#
#   * real / generated / Gemini pools re-augmented with --gray-prob 0: the 15%
#     grayscale copies made colour-only family boundaries (val 17's teal stone
#     band vs grey course-pattern roof) pixel-identical under different labels,
#     teaching texture-only matching;
#   * synthetic quarter v6dsf05_1600 (generate_synthetic_v6.py
#     --same-fill-new-colour 0.5): families that share a fill and differ by
#     colour, which v6d never shows.
#
# Trained from ImageNet weights, not fine-tuned: every fine-tune of the
# published (grayscale-trained) model made the band questions worse.
# Then: epoch on VALIDATION, inference size (2048 vs 4096) on VALIDATION,
# HF14 read once.
#
#   ./run_nogray_mix.sh [seed]      # ~1.3 h on a GH200
set -euo pipefail
SEED=${1:-7}
SIZE=2048
DATA=data/mixed/v6dsf05mix_plus_r4_nogray
CK=data/runs/ck_swint_sf05nogray_res${SIZE}_seed${SEED}
TAG=$(basename $CK)
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
mkdir -p logs data/evaluations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d $DATA ] || { echo "missing $DATA"; exit 1; }

COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size $SIZE --ref-size 224 --width 128 --model swin_t \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512"
[ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $DATA \
  --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/${TAG}_p1.log 2>&1
[ -f $CK/epoch_8.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $DATA \
  --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from $CK/epoch_0.pth > logs/${TAG}_p2.log 2>&1
echo "== trained"

PYTHONPATH=. python3 scripts/select_epoch_on_val.py --runs $CK --image-max-size $SIZE \
  --out data/evaluations/${TAG}_valsel.json | tee logs/${TAG}_valsel.log
EP=$(python3 -c "import json;print(json.load(open('data/evaluations/${TAG}_valsel.json'))[0]['epoch'])")
for R in 2048 4096; do
  PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK/epoch_$EP.pth \
    --indices $VAL --image-max-size $R --ref-size 224 --mask-thresh 0.35 \
    --metrics-out data/evaluations/${TAG}_e${EP}_val_inf$R.json > /dev/null 2>&1 &
done
wait
PYTHONPATH=. python3 scripts/val_failures.py data/evaluations/val_pub_mixr4_e8.json \
  data/evaluations/${TAG}_e${EP}_val_inf2048.json data/evaluations/${TAG}_e${EP}_val_inf4096.json 2>/dev/null
BEST=$(python3 - <<PY
import json, numpy as np
m = {R: np.mean([r["iou"] for r in (lambda d: d.get("selections", d.get("rows")))(
         json.load(open(f"data/evaluations/${TAG}_e${EP}_val_inf{R}.json")))]) for R in (2048, 4096)}
print(max(m, key=m.get))
PY
)
echo "== epoch $EP, inference $BEST (chosen on validation); HF14 read once:"
PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK/epoch_$EP.pth \
  --indices $HF14 --image-max-size $BEST --ref-size 224 --mask-thresh 0.35 \
  --metrics-out data/evaluations/${TAG}_e${EP}_hf14_inf$BEST.json | tail -2
python3 scripts/paired_compare.py --a data/evaluations/pub_swint_mixr4_e8_hf14.json \
  --b data/evaluations/${TAG}_e${EP}_hf14_inf$BEST.json --label-a mixr4_published --label-b nogray_sf05 \
  | sed -n '/mean  /p;/paired mean/,/sign-test/p'
