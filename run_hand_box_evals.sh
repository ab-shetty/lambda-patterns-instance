#!/usr/bin/env bash
# Re-score models with the hand-placed reference boxes (eval_boxes/hand_v1.json, 2026-10-01)
# next to the automatic boxes, on HF14 and validation at one inference size. Automatic-box
# HF14 runs double as a reproduction check against the published numbers. Jobs run in
# priority order (HF14 hand, HF14 auto, val hand, val auto), P at a time; finished
# metrics are skipped, so re-running resumes. Then a paired hand-vs-auto summary.
#
#   ./run_hand_box_evals.sh name=ckpt [name=ckpt ...]     env: SIZE=4096 P=3
set -euo pipefail
cd "$(dirname "$0")"
SIZE=${SIZE:-4096}; P=${P:-3}
BOXES=eval_boxes/hand_v1.json
E=data/evaluations/hand_boxes
mkdir -p $E logs
declare -A SPLIT=([hf14]=12,16,27,7,11,25,23,1,18,2,0,3,14,24 [val]=4,5,6,8,9,10,13,15,17,19,20,21,22,26)
JOBS=$(for S in "hf14 hand" "hf14 auto" "val hand" "val auto"; do
  for ARM in "$@"; do echo "${ARM%%=*} ${ARM#*=} $S"; done; done)
echo "$JOBS" | while read -r N CK SP BX; do
  OUT=$E/${N}_${SP}_${BX}_$SIZE.json
  [ -f $OUT ] && continue
  echo "PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK --indices ${SPLIT[$SP]} \
    --image-max-size $SIZE --ref-size 224 --mask-thresh 0.35 --metrics-out $OUT \
    $([ $BX = hand ] && echo --boxes $BOXES) > logs/handbox_${N}_${SP}_${BX}_$SIZE.log 2>&1"
done | xargs -P $P -I{} bash -c {}

python3 - $E $SIZE "$@" <<'PY'
import json, sys
from scipy.stats import ttest_rel
e, size, arms = sys.argv[1], sys.argv[2], [a.split("=")[0] for a in sys.argv[3:]]
def rows(n, sp, bx):
    m = json.load(open(f"{e}/{n}_{sp}_{bx}_{size}.json"))
    return {(r["image_index"], r["reference_instance"]): r["iou"] for r in m["selections"]}
print(f"@{size}; hand = eval_boxes/hand_v1.json; paired over the questions both keep")
for sp in ("hf14", "val"):
    print(f"== {sp}")
    for n in arms:
        try:
            a, h = rows(n, sp, "auto"), rows(n, sp, "hand")
        except FileNotFoundError:
            print(f"{n:22s} missing"); continue
        k = sorted(set(a) & set(h)); d = [h[q] - a[q] for q in k]
        print(f"{n:22s} auto {sum(a.values())/len(a):.4f} (n={len(a)})  hand {sum(h.values())/len(h):.4f} (n={len(h)})"
              f"  paired {sum(d)/len(d):+.4f} ({sum(x > .01 for x in d)}/{sum(x < -.01 for x in d)}, "
              f"p={ttest_rel([h[q] for q in k], [a[q] for q in k]).pvalue:.2g})")
PY
