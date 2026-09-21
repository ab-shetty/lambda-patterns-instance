#!/usr/bin/env bash
# The three biggest levers in this project, combined for the first time.
#
# Every swin_t number on record is v6d-only at 1024 -- handicapped on the two
# other largest effects the repo has measured. This run removes both handicaps:
#
#   backbone    swin_t          +0.0707 HF14 unfrozen (2026-09-20), +0.08 fit
#                               at matched budget on 100k (2026-09-21)
#   real data   1,548 real + 504 generated-realistic + 1,440 gemini
#                               mix beats v6d-only by a wide margin on record
#   resolution  2048            +0.05 to +0.07 (1280 -> 2048), largest single
#                               effect in the project
#
# SYNTHETIC QUARTER IS v6d, NOT v5. The documented `mix5092` uses
# `toparea1600_balanced`, selected from the HF 20k export of floz-synth-v5.
# v6d supersedes it: synth-only v6d scores 0.686 against v5's 0.573, and the
# best result ever recorded (0.7834) added v6d on top of the mix. Proportions
# are held identical to mix5092 so the mixture-ratio logic is unchanged --
# offline 18x replication is what sets pool weighting, so changing counts
# would change the sampling ratio, not just the data.
#
# CONSEQUENCE FOR ATTRIBUTION: this differs from the 0.7594 +- 0.0191 baseline
# (RefUNet, mix5092, 2048, 2 seeds) in TWO ways -- backbone and synthetic
# source. A win does not attribute cleanly between them. Both changes are
# independently evidenced as positive; if the split ever matters, run RefUNet
# on this same mix to recover it.
#
# Eager, not --compile: torch.compile is worth ~1.5x but `--domain-random`
# gives every sample a unique shape, and Swin's windowed attention is a fresh
# recompile risk per bucket. The repo's compile parity check was on RefUNet
# only. Not worth debugging on the critical path.
set -euo pipefail
SEED=${SEED:-7}
SIZE=${SIZE:-2048}
DATA=${DATA:-data/mixed/v6d1600_rf1548_gen504_gem1440}
CK=data/runs/ck_swint_v6dmix_res${SIZE}_seed${SEED}
mkdir -p logs data/runs

COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size $SIZE --ref-size 224 --width 128 --model swin_t \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512"

# Two optimizer phases, exactly as documented: epoch 0 on the first epoch of a
# planned ten-epoch cosine, then reload, reset the optimizer, and run epochs
# 1-8 on the first eight of a planned nine-epoch cosine.
if [ ! -f "$CK/epoch_0.pth" ]; then
  echo "== phase 1: epoch 0 (schedule 10)"
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 $COMMON \
    > logs/swint_mix_p1.log 2>&1
fi
echo "== phase 2: epochs 1-8 (schedule 9, optimizer reset)"
PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
  --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from "$CK/epoch_0.pth" \
  > logs/swint_mix_p2.log 2>&1

echo "== per-epoch HF14 (diagnostic; val-select before quoting)"
for f in "$CK"/metrics_epoch_*.json; do
  python3 -c "
import json,sys; d=json.load(open(sys.argv[1]))
print(f\"  epoch {d['actual_epoch']}: hf14={d['real_mean_iou']:.4f} val_loss={d['val_loss']:.4f}\")" "$f"
done
