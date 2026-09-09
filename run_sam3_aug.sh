#!/usr/bin/env bash
# Augmentation ablation for the labelling-assist decoder.
#
# The 2026-09-08 model saw the same 165 sheets every epoch and varied only the
# prompt box. This prices sheet-level augmentation, and splits it far enough to
# attribute a win:
#
#   none         the recorded recipe (box jitter only) -- the control
#   default      full dihedral + small affine + zoom-crop + photometric
#   nodihedral   as default, minus vflip and rot90  -> prices the dihedral group
#   nocrop       as default, minus the zoom-crop     -> prices the resolution lever
#
# Two seeds per arm, because this repo's own standard is that a one-seed gain is
# not a result. 24 epochs, not 16: augmented arms fit more slowly and the point
# is to compare converged runs, not to hold the old budget.
#
# Reported by scripts/sam3_aug_report.py on BOTH protocols -- best-epoch (which
# is what the recorded 0.853 used, and is optimistic because the epoch is chosen
# on the same 29 sheets it reports) and the mean of the last 3 epochs (which
# selects nothing). Prefer the second when the two disagree.
#
#   ./run_sam3_aug.sh            # full matrix
#   ./run_sam3_aug.sh default 7  # one arm, one seed
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ~/.env; set +a

EPOCHS=${EPOCHS:-24}
JOBS=${JOBS:-3}                 # concurrent runs on the one GH200
mkdir -p logs_sam3 data/runs

run_one () {
  local arm=$1 seed=$2
  local out=data/runs/sam3_${arm}_s${seed}
  if [ -f "$out/history.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "skip $arm/$seed (exists)"; return
  fi
  echo "start $arm/$seed -> $out"
  PYTHONPATH=. python3 scripts/train_sam3_boxseg.py \
      --aug "$arm" --seed "$seed" --epochs "$EPOCHS" --out "$out" \
      > "logs_sam3/${arm}_s${seed}.log" 2>&1
  echo "done  $arm/$seed"
}

if [ $# -eq 2 ]; then run_one "$1" "$2"; exit 0; fi

ARMS=${ARMS:-"none default nodihedral nocrop"}
SEEDS=${SEEDS:-"7 31"}
for seed in $SEEDS; do
  for arm in $ARMS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    run_one "$arm" "$seed" &
  done
done
wait
echo
python3 scripts/sam3_aug_report.py
