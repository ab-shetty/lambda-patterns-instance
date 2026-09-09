#!/usr/bin/env bash
# Rebuild the RefUNet matcher used by scripts/sam3_refunet_chain.py.
#
# This is NOT the product RefUNet. `startup.md` owns that one (mix5092 @2048,
# 0.7594 on HF14). This is a small real-data-only matcher whose only job is to
# find the other regions on a sheet that match the one the labeller pointed at.
#
# It trains on exactly the 165 SAM 3 TRAIN sheets. That matters: the 29 SAM 3
# val sheets come from the same Roboflow pools, so a matcher trained on all 194
# would have seen the sheets the chain is then evaluated on. Training on the
# train split keeps the held-out 29 clean for BOTH models at once.
#
# Result when this was run (seed 7, 1280): real_mIoU 0.631 at epoch 8, peak
# 0.636 at epoch 5. The chain built on it is documented as NOT ADOPTED --
# 98 boxes buy 212 of 468 regions but cost 207 deletions. An oracle matcher
# gives 435 for 129, so the ceiling is a 44% gesture saving and the matcher is
# the binding constraint. See labeling_assist.md.
#
# Prerequisite: the Roboflow pools and data/sam3_ft (see "Reproduce" in
# labeling_assist.md). ~35 min on one GH200.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ~/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SRC=data/refunet_src_165
CK=${CK:-data/runs/ck_refunet_real165_s7}
SEED=${SEED:-7}

# 1. the 165 train sheets, asserted disjoint from the val 29
PYTHONPATH=. python3 - "$SRC" <<'PY'
import json, os, shutil, sys
out = sys.argv[1]
train = [json.loads(l) for l in open("data/sam3_ft/train.jsonl")]
val_names = {os.path.basename(json.loads(l)["image"])
             for l in open("data/sam3_ft/val.jsonl")}
for d in ("images", "annotations"):
    os.makedirs(f"{out}/{d}", exist_ok=True)
for s in train:
    img = s["image"]; base = os.path.basename(img)
    assert base not in val_names, f"LEAK: {base} is in the val split"
    pool = os.path.dirname(os.path.dirname(img))
    stem = os.path.splitext(base)[0]
    shutil.copy(img, f"{out}/images/{base}")
    shutil.copy(os.path.join(pool, "annotations", stem + ".json"),
                f"{out}/annotations/{stem}.json")
print(f"{out}: {len(train)} sheets, 0 overlap with the {len(val_names)} val sheets")
PY

# 2. offline augmentation, same 18x recipe the product mixes use
python3 scripts/augment_local_dataset.py --src "$SRC" --out "${SRC}_strong18" \
  --aug-per-scene 17 --strong --seed 5858            # 165 -> 2,970

# 3. two optimizer phases, as startup.md specifies -- do not collapse into one
COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 --image-max-size 1280 \
  --ref-size 224 --width 128 --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512"
PYTHONPATH=. python3 scripts/train_refunet.py --local-data "${SRC}_strong18" \
  --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 $COMMON
PYTHONPATH=. python3 scripts/train_refunet.py --local-data "${SRC}_strong18" \
  --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from "$CK/epoch_0.pth"

echo
echo "matcher at $CK/epoch_8.pth -- evaluate the chain with:"
echo "  PYTHONPATH=. python3 scripts/sam3_refunet_chain.py --refunet $CK/epoch_8.pth \\"
echo "    --sam3 data/runs/sam3_nodihedral_s7/best.pth"
echo "  (--oracle substitutes a perfect matcher, for the ceiling)"
