#!/usr/bin/env bash
# Real-sheet numbers for the fit-sweep restart endpoints (2026-10-01), by protocol:
# each arm's checkpoint is scored on VALIDATION at 2048 and 4096, the inference size is
# chosen there, and HF14 is read once at that size. Paired HF14 per-question deltas vs
# the first arm are printed. Compare with revit10k-roi-e5: HF14 0.7868 / 0.8126 @2048 / 4096.
#
#   ./run_fit_realeval.sh <name=ckpt> [<name=ckpt> ...]
set -euo pipefail
cd "$(dirname "$0")"
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
E=data/evaluations/fit_realeval
mkdir -p $E logs
for ARM in "$@"; do
  N=${ARM%%=*}; CK=${ARM#*=}
  for SZ in 2048 4096; do
    OUT=$E/${N}_val_$SZ.json
    [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK \
      --indices $VAL --image-max-size $SZ --ref-size 224 --mask-thresh 0.35 --metrics-out $OUT \
      > logs/fit_realeval_${N}_val_$SZ.log 2>&1
  done
  SZ=$(python3 - $E $N <<'PY'
import json, sys
e, n = sys.argv[1:]
def mean(p):
    m = json.load(open(p))
    rows = next(v for v in m.values() if isinstance(v, list) and v and isinstance(v[0], dict))
    return sum(r["iou"] for r in rows) / len(rows)
v = {s: mean(f"{e}/{n}_val_{s}.json") for s in (2048, 4096)}
print(max(v, key=v.get))
PY
)
  OUT=$E/${N}_hf14_$SZ.json
  [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK \
    --indices $HF14 --image-max-size $SZ --ref-size 224 --mask-thresh 0.35 --metrics-out $OUT \
    > logs/fit_realeval_${N}_hf14_$SZ.log 2>&1
  echo "$N $SZ" >> $E/choices.txt
done
python3 - $E "$@" <<'PY'
import json, sys
from scipy.stats import ttest_rel
e, arms = sys.argv[1], [a.split("=")[0] for a in sys.argv[2:]]
def rows(p):
    m = json.load(open(p))
    return next(v for v in m.values() if isinstance(v, list) and v and isinstance(v[0], dict))
mean = lambda r: sum(x["iou"] for x in r) / len(r)
choice = dict(l.split() for l in open(f"{e}/choices.txt"))
base = None
for n in arms:
    v2, v4 = mean(rows(f"{e}/{n}_val_2048.json")), mean(rows(f"{e}/{n}_val_4096.json"))
    h = rows(f"{e}/{n}_hf14_{choice[n]}.json"); line = f"{n:12s} val 2048 {v2:.4f}  4096 {v4:.4f}  -> HF14 @{choice[n]} {mean(h):.4f}"
    if base is None:
        base = (n, h)
    else:
        a, b = [x["iou"] for x in base[1]], [x["iou"] for x in h]
        d = [y - x for x, y in zip(a, b)]
        line += (f"  vs {base[0]} {sum(d)/len(d):+.4f} ({sum(x > 0.01 for x in d)} better / "
                 f"{sum(x < -0.01 for x in d)} worse, p={ttest_rel(b, a).pvalue:.2g})")
    print(line)
PY
