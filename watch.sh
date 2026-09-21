#!/usr/bin/env bash
# Every training on this box, no arguments, no per-run setup.
#   ./watch.sh        one snapshot
#   ./watch.sh -w     refresh every 15s
#
# Finds runs by inspecting live processes rather than by log-name convention:
# each training script redirects stdout to its own log, so /proc/PID/fd/1 is the
# log whatever it was named, and --checkpoint-dir comes off the command line.
# That is why this keeps working when a new run_*.sh invents another log name.
cd "$(dirname "$0")" || exit 1

fmt_run() {   # $1 = checkpoint dir, $2 = log (may be empty)
  local CK="$1" LOG="$2" cur=""
  [ -n "$LOG" ] && [ -r "$LOG" ] && cur=$(tr '\r' '\n' < "$LOG" 2>/dev/null \
      | grep -aoE 'Epoch [0-9]+ \[[a-z]+\]: +[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ \[[^]]*\]' \
      | tail -1 | tr -s ' ')
  printf "  %s\n" "$(basename "$CK")"
  [ -n "$cur" ] && printf "     %s\n" "$cur"
  python3 - "$CK" <<'PY'
import json, glob, re, sys
fs = sorted(glob.glob(sys.argv[1] + "/metrics_epoch_*.json"),
            key=lambda p: int(re.findall(r'\d+', p)[-1]))
rows = []
for f in fs:
    try: d = json.load(open(f))
    except Exception: continue
    rows.append((d.get("actual_epoch", -1), d.get("real_mean_iou", float("nan"))))
rows.sort()
if rows:
    best = max(r[1] for r in rows)
    tail = rows[-6:]
    s = "  ".join(f"e{e}:{i:.4f}" + ("*" if i == best else "") for e, i in tail)
    more = f"(+{len(rows)-len(tail)} earlier) " if len(rows) > len(tail) else ""
    print(f"     {len(rows)} ep  {more}{s}   best {best:.4f}")
else:
    print("     no epochs yet")
PY
  local V; V=$(ls data/evaluations/val_$(basename "$CK").json 2>/dev/null | head -1)
  [ -n "$V" ] && printf "     VAL-SELECTED: %s\n" "$(tr -d ' \n' < "$V" | head -c 300)"
  return 0
}

snap() {
  printf "%s   GPU %s\n" "$(date '+%H:%M:%S')" \
    "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
       --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
  local found=0 seen=""
  echo; echo "── TRAINING ─────────────────────────────────────────────"
  while read -r pid rest; do
    [ -z "$pid" ] && continue
    local ck log
    ck=$(sed -n 's/.*--checkpoint-dir[ =]\([^ ]*\).*/\1/p' <<<"$rest")
    [ -z "$ck" ] && continue
    case " $seen " in *" $ck "*) continue;; esac   # dataloader workers share the cmdline
    seen="$seen $ck"
    log=$(readlink -f "/proc/$pid/fd/1" 2>/dev/null)
    [[ "$log" == /dev/* || "$log" == *"pipe:"* ]] && log=""
    fmt_run "$ck" "$log"; found=1
  done < <(ps -eo pid,args | grep -a "train_refune[t].py" | grep -av grep)
  [ "$found" = 0 ] && echo "  (nothing training)"

  # non-training jobs worth seeing, since they gate the next training
  local other
  other=$(ps -eo pid,args | grep -aE "augment_local_datase[t].py|generate_synthetic_v[6].py|select_epoch_on_va[l].py" \
          | sed 's/^ *//' | cut -c1-110)
  [ -n "$other" ] && { echo; echo "── OTHER JOBS ───────────────────────────────────────────"; echo "$other" | sed 's/^/  /'; }

  echo; echo "── FINISHED (last 12h) ──────────────────────────────────"
  local dirs live any=0 d
  live=$(ps -eo args 2>/dev/null | grep -a "train_refune[t].py" || true)
  dirs=$(find data/runs -maxdepth 1 -type d -name 'ck_*' -mmin -720 2>/dev/null | sort)
  for d in $dirs; do
    case "$live" in *"--checkpoint-dir $d "*|*"--checkpoint-dir $d") continue;; esac
    fmt_run "$d" ""
    any=1
  done
  [ "$any" = 0 ] && echo "  (none)"
  return 0
}

if [ "${1:-}" = "-w" ]; then while true; do clear; snap; sleep 15; done; else snap; fi
