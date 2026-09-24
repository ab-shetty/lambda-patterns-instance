#!/usr/bin/env bash
# Does generated (Gemini) data transfer to real plans better than procedural
# (v6d) data, at MATCHED size? ~20 min on a GH200.
#
# For each arm: train on 148 sources x18 augmentations, then measure
#   fresh = IoU on unseen plans from the arm's own source
#   real  = HF14
# and report real / fresh. The procedural ratio is ~0.85 for every v6-family
# generator (2026-09-23, synth_progress.md); if Gemini's is near 1 at the same
# size, the gap is the procedural fill vocabulary, not synthetic data as such.
#
# Built for speed, not headline numbers: 1024 px and all arms concurrently on
# one GPU. Absolute scores are low at 1024; only the between-arm comparison
# is meaningful. Arms: gemA / gemB (two held-out folds of 20 Gemini sources)
# and v6d (148 plans; fresh = 400 of v6d_fresh600).
set -euo pipefail
SEED=${1:-7}
SIZE=1024
N_TRAIN=148
D=data/transfer
mkdir -p $D/none/annotations logs   # none: see fresh() below
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- data -------------------------------------------------------------------
GEM=data/roboflow/floz-gen-gemini-r234-clean
[ -d $GEM ] || python3 scripts/merge_local_datasets.py --out $GEM --sources \
  data/roboflow/floz-gen-gemini-r2-clean data/roboflow/floz-gen-gemini-r3-clean \
  data/roboflow/floz-gen-gemini-r4-clean

# split SRC into train/heldout by a fixed shuffle: held-out = positions [a, b)
split() {  # split SRC OUT_TRAIN OUT_HELD a b n_train
  python3 - "$@" <<'PY'
import os, random, sys
src, tr, he, a, b, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
anns = sorted(os.listdir(f"{src}/annotations")); random.Random(1234).shuffle(anns)
held = anns[a:b]; train = [x for x in anns if x not in held][:n]
for out, names in ((tr, train), (he, held)):
    if not names:
        continue
    os.makedirs(f"{out}/images", exist_ok=True); os.makedirs(f"{out}/annotations", exist_ok=True)
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(f"{src}/images")}
    for x in names:
        stem = os.path.splitext(x)[0]
        os.link(f"{src}/annotations/{x}", f"{out}/annotations/{x}")
        os.link(f"{src}/images/{imgs[stem]}", f"{out}/images/{imgs[stem]}")
print(f"{tr}: {len(train)}  {he}: {len(held)}")
PY
}

# augment_local_dataset.py is a serial loop: split into 64 chunks (one per vCPU)
augment() {  # augment SRC OUT
  local src=$1 out=$2 tmp=$2.chunks
  [ -d "$out" ] && return
  rm -rf "$tmp"; mkdir -p "$tmp"
  python3 - "$src" "$tmp" <<'PY'
import os, sys
src, tmp = sys.argv[1:]
anns = sorted(os.listdir(f"{src}/annotations"))
imgs = {os.path.splitext(f)[0]: f for f in os.listdir(f"{src}/images")}
for i, x in enumerate(anns):
    c = f"{tmp}/in{i % 64:02d}"
    os.makedirs(f"{c}/images", exist_ok=True); os.makedirs(f"{c}/annotations", exist_ok=True)
    stem = os.path.splitext(x)[0]
    os.link(f"{src}/annotations/{x}", f"{c}/annotations/{x}")
    os.link(f"{src}/images/{imgs[stem]}", f"{c}/images/{imgs[stem]}")
PY
  for c in "$tmp"/in*; do
    python3 scripts/augment_local_dataset.py --src "$c" --out "$tmp/out${c##*/in}" \
      --aug-per-scene 17 --strong --seed 5858 > /dev/null &
  done
  wait
  python3 scripts/merge_local_datasets.py --out "$out" --sources "$tmp"/out* | tail -1
  rm -rf "$tmp"
}

[ -d $D/gemA_train ] || split $GEM $D/gemA_train $D/gemA_held 0 20 $N_TRAIN
[ -d $D/gemB_train ] || split $GEM $D/gemB_train $D/gemB_held 20 40 $N_TRAIN
[ -d $D/v6d_train ]  || split data/synthetic/v6d_train2000 $D/v6d_train $D/v6d_unused 0 0 $N_TRAIN
# all three arms at once: 3 x 64 processes keeps every vCPU busy until the end
for arm in gemA gemB v6d; do augment $D/${arm}_train $D/${arm}_train18 & done
wait

# ---- train, all arms concurrently ------------------------------------------
COMMON="--batch-size 8 --num-workers 12 --prefetch-factor 4 \
  --image-max-size $SIZE --ref-size 224 --width 128 --model swin_t \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512"
train() {  # documented two-phase schedule
  local arm=$1 CK=data/runs/ck_tr_${1}_res${SIZE}_seed${SEED}
  [ -f $CK/epoch_8.pth ] && return
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data $D/${arm}_train18 \
    --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/$(basename $CK)_p1.log 2>&1
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data $D/${arm}_train18 \
    --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON \
    --reset-optimizer --init-from $CK/epoch_0.pth > logs/$(basename $CK)_p2.log 2>&1
}
for arm in gemA gemB v6d; do train $arm & done
wait

# ---- measure: epoch 8 for every arm (same rule, no selection) ---------------
# held-out pools are disjoint by construction; fresh_synth_iou.py decides
# "seen" by filename and merged pools share item_XXXXXX names, so pass none
fresh() {  # fresh ARM POOL
  PYTHONPATH=. python3 scripts/fresh_synth_iou.py --n 400 --image-max-size $SIZE \
    --checkpoint data/runs/ck_tr_${1}_res${SIZE}_seed${SEED}/epoch_8.pth \
    --all-pool "$2" --trained-pool $D/none 2>&1 | grep -oE "IoU: [0-9.]+" | head -1 | cut -d' ' -f2
}
printf "%-6s %8s %8s %8s\n" arm fresh HF14 ratio
for arm in gemA gemB v6d; do
  pool=$D/${arm}_held; [ $arm = v6d ] && pool=data/synthetic/v6d_fresh600
  f=$(fresh $arm $pool)
  r=$(python3 -c "import json;print(round(json.load(open('data/runs/ck_tr_${arm}_res${SIZE}_seed${SEED}/metrics_epoch_8.json'))['real_mean_iou'],4))" 2>/dev/null || echo NA)
  python3 -c "f,r='$f','$r'; print(f'{\"$arm\":6s} {f:>8s} {r:>8s} {float(r)/float(f):8.3f}' if r!='NA' else f'{\"$arm\":6s} {f:>8s} {r:>8s}')"
done
