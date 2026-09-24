#!/usr/bin/env bash
# --vocab2 (3D lap shadows, basketweave) synth-only screen: the treatment arm at
# three seeds concurrently, compared with each seed's existing v6d arm from
# run_split_screen_all.sh (same ids / recipe / seed). Records and pushes itself.
set -euo pipefail
E=data/evaluations; VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26; HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--batch-size 8 --num-workers 12 --prefetch-factor 4 --image-max-size 1024 --ref-size 224 \
  --width 128 --model swin_t --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 --domain-random \
  --mask-thresh 0.35 --pad-grid 512"
run() { local SEED=$1 CK=data/runs/ck_so_vocab2_s$1
  [ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data data/synthetic/v6dV2_1600 \
    --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON --seed $SEED > logs/so_vocab2_s${SEED}_p1.log 2>&1
  [ -f $CK/epoch_8.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data data/synthetic/v6dV2_1600 \
    --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON --seed $SEED --reset-optimizer \
    --init-from $CK/epoch_0.pth > logs/so_vocab2_s${SEED}_p2.log 2>&1
  for split in val hf14; do IDX=$VAL; [ $split = hf14 ] && IDX=$HF14
    PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK/epoch_8.pth --indices $IDX \
      --image-max-size 2048 --ref-size 224 --mask-thresh 0.35 --metrics-out $E/so_vocab2_s${SEED}_$split.json > /dev/null 2>&1; done; }
for s in 7 31 99; do run $s & done; wait
python3 - <<'PY' | tee logs/vocab2_screen.txt
import json, numpy as np
E="data/evaluations"; m=lambda f: np.mean([r["iou"] for r in json.load(open(f"{E}/{f}.json"))["selections"]])
dh, dv = [], []
for s in (7, 31, 99):
    dh.append(m(f"so_vocab2_s{s}_hf14") - m(f"so_v6d_s{s}_hf14")); dv.append(m(f"so_vocab2_s{s}_val") - m(f"so_v6d_s{s}_val"))
print("| `--vocab2 0.7` | " + " / ".join(f"{d:+.3f}" for d in dh) + f" | **{np.mean(dh):+.3f}** | " + " / ".join(f"{d:+.3f}" for d in dv) + f" | {np.mean(dv):+.3f} |")
PY
{ echo; echo "**\`--vocab2 0.7\` (3D lap shadows + basketweave; labels identical to v6d_1600), synth-only screen, seeds 7 / 31 / 99, deltas vs same-seed v6d (HF14 | mean | validation | mean):**"; echo; cat logs/vocab2_screen.txt; } >> synth_progress.md
git add synth_progress.md run_vocab2_screen.sh generate_synthetic_v6.py && git commit -q -m "--vocab2 (3D lap shadows, basketweave): three-seed synth-only screen (auto-recorded)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD
