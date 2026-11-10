#!/usr/bin/env bash
# Full-resolution refinement head A/B on gem2s-x (2026-11-10). Resumable.
#   arm refine32_s7    --refine-head 32 --train-only refine.  (base frozen)
#   arm refine32e4_s7  same + --edge-weight 4 --edge-px 3
# then maps + stack / edgeink / boundary_band on valfix3 / hf14fix2 / gemini30 for gem2s-x and each blend50.
# Gate: stack ms+cc >= +0.005 on BOTH valfix3 and hf14fix2 vs gem2s-x, Gemini30 >= -0.003.
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ~/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. ALLOW_STALE_POOLS=1
SYN=data/synthetic/gemho_s600000; GEM=data/roboflow/floz-gen-gemini-r234-no30; MIX=data/mixed/gemho_mix
INIT=data/runs/ck_best/gem2s-x.pth
ARMS=${ARMS:-"refine32_s7 refine32e4_s7"}
X="--refine-head 32 --train-only refine. --image-max-size-max 3072"
declare -A TX=([refine32_s7]="$X" [refine32e4_s7]="$X --edge-weight 4 --edge-px 3"
               [refine32_unf_s7]="--refine-head 32 --image-max-size-max 3072"
               [refine32e4_unf_s7]="--refine-head 32 --image-max-size-max 3072 --edge-weight 4 --edge-px 3")

probe() {   # probe TAG MAPS_DIR
  local tag=$1 m=$2 o=data/evaluations/refine_ab/$1; mkdir -p $o
  for s in valfix3 hf14fix2 gemini30; do
    [ -f $o/stack_$s.txt ] || python3 scripts/pp_probe/stack.py $s $m > $o/stack_$s.txt 2>&1
    [ -f $o/edgeink_$s.txt ] || python3 scripts/pp_probe/edgeink.py $s $m > $o/edgeink_$s.txt 2>&1
    [ -f $o/band_$s.txt ] || python3 scripts/pp_probe/boundary_band.py $s $m 1,2,4 > $o/band_$s.txt 2>&1
  done
  echo "== probed $tag"
}

# gem2s-x itself (GPU idle while the pool builds)
[ -f data/floz-diag/gem2s_x/.mapsdone ] || { scripts/gpu_maps.sh $INIT data/floz-diag/gem2s_x && touch data/floz-diag/gem2s_x/.mapsdone; }
probe gem2s_x data/floz-diag/gem2s_x &

if [ ! -f $SYN/generation_manifest.json ]; then
  WT=/tmp/gen_2026-10-15b; [ -d $WT ] || git worktree add -f $WT 6b25130^
  (cd $WT && python3 scripts/generate_synthetic_fc.py --out $OLDPWD/$SYN --n 18000 --start 600000 --seed 6 \
    --revit --revit-plans --shaped --workers 64 --mode-weights 66,16,18 \
    --plan-source $OLDPWD/data/reference/swiss_dwellings/layouts.pkl.gz) > logs/gen_gemho.log 2>&1
fi
echo "== synthetic: $(ls $SYN/annotations | wc -l) sheets (kq pool: 17,906)"
[ -d $GEM-strong18 ] || { echo "STOP: $GEM-strong18 missing (run_gemini_holdout.sh steps 1-2)"; exit 1; }
[ -d $MIX ] || python3 scripts/merge_local_datasets.py --out $MIX --sources $SYN $GEM-strong18 > /dev/null
echo "== mix: $(ls $MIX/annotations | wc -l) records"
wait

for T in $ARMS; do
  TAG=$T POOL=$MIX SEED=7 INIT=$INIT BATCH=4 TRAINX="${TX[$T]}" ./run_single_pass.sh
  grep -h "trainable params" logs/${T}_dr.log logs/${T}_nodr.log
  M=data/floz-diag/$T
  [ -f $M/.mapsdone ] || { scripts/gpu_maps.sh data/runs/ck_${T}_blend50.pth $M && touch $M/.mapsdone; }
  probe $T $M
done
echo "REFINE AB DONE $(date +%H:%M)"
