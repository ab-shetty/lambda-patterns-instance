#!/bin/bash
# Compact progress for every training run under data/runs, newest activity first.
#   bash scripts/train_status.sh          one snapshot
#   watch -n 20 bash scripts/train_status.sh   auto-refreshing
cd "$(dirname "$0")/.." || exit 1
printf "%s   GPU %s\n\n" "$(date '+%H:%M:%S')" \
  "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"
printf "%-30s %6s  %-7s %-7s %s\n" RUN EPOCHS LATEST PEAK "CURRENT"
printf '%.0s-' {1..78}; echo
for d in $(ls -dt data/runs/ck_* 2>/dev/null); do
  n=$(ls "$d"/metrics_epoch_*.json 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && [ ! -f "$d/../../../logs/$(basename $d).log" ] && continue
  read latest peak <<<"$(python3 - "$d" <<'PY'
import json,glob,re,sys
fs=sorted(glob.glob(sys.argv[1]+"/metrics_epoch_*.json"),key=lambda p:int(re.findall(r'\d+',p)[-1]))
v=[json.load(open(f))["real_mean_iou"] for f in fs]
print(f"{v[-1]:.4f} {max(v):.4f}" if v else "-      -")
PY
)"
  tag=$(basename "$d" | sed 's/^ck_//')
  log=$(ls -t logs/*.log 2>/dev/null | xargs -r grep -ls "checkpoint-dir $d" 2>/dev/null | head -1)
  cur=""
  [ -n "$log" ] && cur=$(tr '\r' '\n' < "$log" | grep -aoE 'Epoch [0-9]+ \[[a-z]+\]: +[0-9]+%' | tail -1 | tr -s ' ')
  printf "%-30s %6s  %-7s %-7s %s\n" "$tag" "$n" "$latest" "$peak" "$cur"
done
