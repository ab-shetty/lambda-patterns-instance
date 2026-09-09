#!/bin/bash
# Compact progress for the labelling-assist (SAM 3) runs under data/runs/sam3_*.
# Sibling of train_status.sh, which covers the product model's ck_* runs.
#
#   bash scripts/sam3_status.sh              one snapshot
#   bash scripts/sam3_status.sh -w           auto-refresh every 30s
#   watch -n 30 bash scripts/sam3_status.sh  same, via watch(1)
#
# LATEST/PEAK are val polygon IoU (median) -- the metric best.pth is chosen on.
# LAST3 is the mean of the final three epochs so far, which selects nothing and
# is the fair number to compare arms on. MASK is val mask IoU (mean), which is
# far less noisy than the polygon metric: identical weights re-evaluated in a
# fresh process move poly median by ~0.006 but mask mean by ~0.0000.
cd "$(dirname "$0")/.." || exit 1

snapshot () {
  printf "%s   GPU %s\n" "$(date '+%H:%M:%S')" \
    "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
  # Anchored at the start of the command line so this does not count itself,
  # nor any grep/ps whose own arguments happen to contain the script name.
  local n_run
  n_run=$(ps -eo args= | grep -c "^python3 scripts/train_sam3_boxseg\.py" 2>/dev/null || true)
  printf "%s training process(es) alive\n\n" "$n_run"
  printf "%-26s %7s  %-7s %-7s %-7s %-7s %-6s %s\n" \
    RUN EPOCHS LATEST PEAK LAST3 MASK ">=0.8" ETA
  printf '%.0s-' {1..88}; echo
  for d in $(ls -dt data/runs/sam3_* 2>/dev/null); do
    [ -f "$d/history.json" ] || continue
    python3 - "$d" <<'PY'
import json, os, sys, time
d = sys.argv[1]
try:
    h = [e for e in json.load(open(f"{d}/history.json")) if e["epoch"] >= 0]
except Exception:
    sys.exit()
if not h:
    sys.exit()
v = [e["poly_median"] for e in h]
last3 = sum(v[-3:]) / len(v[-3:])
mask = h[-1].get("mask_mean", float("nan"))
ge80 = h[-1].get("ge80", float("nan"))
tag = os.path.basename(d).replace("sam3_", "")
# ETA from the log's per-epoch seconds and the run's --epochs, when both known
eta = ""
log = f"logs_sam3/{tag}.log"
total = None
total = int(os.environ.get("EPOCHS", "24"))     # matches run_sam3_aug.sh's default
if os.path.exists(log):
    txt = open(log, errors="ignore").read()
    secs = [int(t[:-1]) for t in txt.split() if t.endswith("s") and t[:-1].isdigit()]
    left = max(0, total - len(h))
    if not left:
        eta = "done"
    elif secs:
        per = sum(secs[-3:]) / len(secs[-3:])
        eta = f"{left} ep -> {time.strftime('%H:%M', time.localtime(time.time() + left * per))}"
print(f"{tag:<26} {len(h):3d}/{total or '?':<3}  {v[-1]:<7.4f} {max(v):<7.4f} "
      f"{last3:<7.4f} {mask:<7.4f} {ge80:<6.1%} {eta}")
PY
  done
  echo
  echo "compare arms:  python3 scripts/sam3_aug_report.py"
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; snapshot; sleep "${2:-30}"; done
else
  snapshot
fi
