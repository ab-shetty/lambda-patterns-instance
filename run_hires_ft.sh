#!/usr/bin/env bash
# High-resolution fine-tune of gem2s-x (2026-11-11): one arm, sheets drawn at 3072..4096 (the product infers at 3072/4096/5120;
# training so far topped out at 2048-3072), batch 2, lr 5e-5 (backbone x0.1), one epoch over gemho_hr6k (4000 random kq
# synthetic + all 2502 Gemini holdout records of gemho_mix). Outputs ck_hrft_s7 (the fine-tune) and ck_hrft_s7_b50 (0.5 with
# gem2s-x); maps at 2048 / 3072 / 4096 / 5120; product v3 scoring (pp_probe/ttasweep.py, stack.py).
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ~/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
INIT=data/runs/ck_best/gem2s-x.pth; CK=data/runs/ck_hrft_s7
[ -f $CK/.done ] || { python3 scripts/train_refunet.py --local-data data/mixed/gemho_hr6k --checkpoint-dir $CK --epochs 1 \
  --batch-size 2 --num-workers 16 --prefetch-factor 4 --image-max-size 3072 --image-max-size-max 4096 --ref-size 224 \
  --width 128 --model swin_t --lr 5e-5 --backbone-lr-mult 0.1 --train-split 0.99 --mask-thresh 0.35 --seed 7 --pad-grid 512 \
  --reset-optimizer --init-from $INIT --roi-ref --roi-ref-mode add > logs/hrft_s7.log 2>&1 && touch $CK/.done; }
echo "== trained $(date +%H:%M)"
python3 - $CK/epoch_0.pth $INIT data/runs/ck_hrft_s7_b50.pth <<'PY'
import sys, torch
a, b = (torch.load(p, map_location="cpu", weights_only=False) for p in sys.argv[1:3])
avg = {k: ((a["model"][k].float() + b["model"][k].float()) / 2).to(a["model"][k].dtype)
       if a["model"][k].is_floating_point() else a["model"][k] for k in a["model"]}
torch.save({**a, "model": avg, "epoch": None, "actual_epoch": None}, sys.argv[3])
PY
cp $CK/epoch_0.pth data/runs/ck_hrft_s7_ft.pth
for t in ft b50; do
  M=data/floz-diag/hrft_s7_$t
  scripts/gpu_maps.sh data/runs/ck_hrft_s7_$t.pth $M
  scripts/gpu_maps_sizes.sh data/runs/ck_hrft_s7_$t.pth $M "3072" "val hf14 gemini30"
  scripts/gpu_maps_sizes.sh data/runs/ck_hrft_s7_$t.pth $M "5120" "val hf14"
  scripts/gpu_maps_sizes.sh data/runs/ck_hrft_s7_$t.pth $M "5120" "gemini30"
done
echo "== maps $(date +%H:%M)"
for t in ft b50; do M=data/floz-diag/hrft_s7_$t
  for s in valfix4 hf14fix4; do for i in 0 1 2 3 4 5 6; do python3 scripts/pp_probe/ttasweep.py $s $M $i 7 > /dev/null 2>&1 & done; done
  for i in $(seq 0 9); do python3 scripts/pp_probe/ttasweep.py gemini30 $M $i 10 > /dev/null 2>&1 & done
done
wait
for t in ft b50; do for s in valfix4 hf14fix4 gemini30; do python3 scripts/pp_probe/ttasweep.py --summary hrft_s7_$t $s; done; done
echo "HIRES FT DONE $(date +%H:%M)"
