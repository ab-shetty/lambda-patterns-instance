#!/usr/bin/env bash
# Progress for the mixbase-vs-mixv6d experiment.
#   ./watch_v6d_mix.sh          one snapshot
#   ./watch_v6d_mix.sh -w       auto-refresh every 20s
cd "$(dirname "$0")" || exit 1

snap() {
  printf "%s   GPU %s\n" "$(date '+%H:%M:%S')" \
    "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"
  echo
  echo "ARMS (driver):"
  grep -aE "^===== |^ALL ARMS DONE" logs/v6d_mix_driver.log 2>/dev/null | tail -6 | sed 's/^/  /'
  echo
  printf "  %-28s %-6s %-8s %-8s %s\n" RUN EPOCHS LATEST PEAK CURRENT
  printf '  '; printf '%076d' 0 | tr '0' '-'; echo
  for tag in mixbase_2048_seed7 mixv5v6d_2048_seed7 mixbase_2048_seed31 mixv5v6d_2048_seed31; do
    d=data/runs/ck_$tag; log=logs/ck_$tag.log
    started=1; [ -d "$d" ] || started=0
    n=$(ls "$d"/metrics_epoch_*.json 2>/dev/null | wc -l)
    read latest peak <<<"$(python3 - "$d" <<'PY'
import json,glob,re,sys
fs=sorted(glob.glob(sys.argv[1]+"/metrics_epoch_*.json"),key=lambda p:int(re.findall(r'\d+',p)[-1]))
v=[json.load(open(f))["real_mean_iou"] for f in fs]
print(f"{v[-1]:.4f} {max(v):.4f}" if v else "-        -")
PY
)"
    cur=$(tr '\r' '\n' < "$log" 2>/dev/null | grep -aoE 'Epoch [0-9]+ \[[a-z]+\]: +[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+' | tail -1 | tr -s ' ')
    [ "$started" = 0 ] && cur="(pending)"
    [ -z "$cur" ] && [ "$started" = 1 ] && cur="(compiling)"
    printf "  %-28s %-6s %-8s %-8s %s\n" "$tag" "$n" "$latest" "$peak" "$cur"
  done
  echo
  echo "RESULTS so far (val-selected HF14):"
  for f in data/evaluations/val_ck_*.json; do
    [ -e "$f" ] || { echo "  (none yet)"; break; }
    python3 -c "
import json,sys
d=json.load(open('$f'))
print('  $(basename $f .json | sed s/val_ck_//):', json.dumps(d)[:160])
" 2>/dev/null
  done
}

if [ "${1:-}" = "-w" ]; then while true; do clear; snap; sleep 20; done; else snap; fi
