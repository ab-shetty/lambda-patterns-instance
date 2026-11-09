#!/usr/bin/env bash
# Time-boxed variant of run_edge_test.sh (2026-11-09, ~3 h of VM left): once gemedge4_s7's nodr arm finishes, the GPU goes to
# gemedge10_s7 (run_edge_test.sh W=10) while gemedge4_s7 is blended, scored, mapped and probed on the CPU (4 jobs at a time).
# CPU and GPU scoring of one checkpoint agree to the last decimals (startup.md: minor nondeterminism).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
T=gemedge4_s7; CK=data/runs/ck_${T}_nodr
while pgrep -f "train_refunet.py.*ck_${T}_nodr" > /dev/null; do sleep 20; done
[ -f $CK/epoch_0.pth ] || { echo "no $CK/epoch_0.pth"; exit 1; }
touch $CK/.done
echo "== $T trained $(date +%H:%M)"

# GPU: weight 10 (train + GPU scoring + GPU maps + probes)
ALLOW_STALE_POOLS=1 W=10 nohup ./run_edge_test.sh > logs/edge_test_w10.log 2>&1 &

# CPU: weight 4 blend (same code as run_single_pass.sh), scoring, maps, probes
B=data/runs/ck_${T}_blend50.pth
[ -f $B ] || python3 - data/runs/ck_${T}_dr/epoch_0.pth $CK/epoch_0.pth $B <<'PY'
import sys, torch
a, b = (torch.load(p, map_location="cpu") for p in sys.argv[1:3])
avg = {k: ((a["model"][k].float() + b["model"][k].float()) / 2).to(a["model"][k].dtype)
       if a["model"][k].is_floating_point() else a["model"][k] for k in a["model"]}
torch.save({**a, "model": avg, "epoch": None, "actual_epoch": None}, sys.argv[3])
PY
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26; HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24; G=data/eval_pools/gemini30_s20261006
O=data/evaluations/$T; mkdir -p $O
for job in "hf14:--indices $HF14" "hf14fix2:--indices $HF14 --label-fixes eval_labels/hf14_fixes_v2.json" \
           "val:--indices $VAL" "gemini30:--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json"; do
  OUT=$O/blend50_${job%%:*}_4096.json
  [ -f $OUT ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 python3 scripts/evaluate_refunet_selection.py --checkpoint $B ${job#*:} \
    --image-max-size 4096 --ref-size 224 --mask-thresh 0.35 --workers 0 --metrics-out $OUT > $O/${job%%:*}.log 2>&1 &
done
wait
echo "== $T scored (CPU) $(date +%H:%M)"
for sp in hf14 hf14fix2 val gemini30; do echo "$sp $(python3 -c "import json;print(round(json.load(open('$O/blend50_${sp}_4096.json'))['mean_iou'],4))")"; done
# maps: 4 at a time on CPU (cpu_maps.sh runs 2)
M=data/floz-diag/$T
one() {
  sp=$1; sz=$2; d=$M/${sp}_$sz; [ -f $d/metrics.json ] && return; mkdir -p $d/probs
  case $sp in hf14) a="--indices $HF14";; val) a="--indices $VAL";; gemini30) a="--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json";; esac
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 python3 scripts/evaluate_refunet_selection.py --checkpoint $B $a \
    --image-max-size $sz --ref-size 224 --mask-thresh 0.35 --workers 0 --save-probs $d/probs --metrics-out $d/metrics.json > $d/log.txt 2>&1
}
one gemini30 4096 & one val 4096 & one hf14 4096 & one gemini30 2048 & wait
one val 2048 & one hf14 2048 & wait
echo "== $T maps (CPU) $(date +%H:%M)"
scripts/pp_probe/run_probes.sh $M
echo "EDGE4 CPU SIDE DONE $(date +%H:%M)"
