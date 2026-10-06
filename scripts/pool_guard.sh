# Source from a run script: `. scripts/pool_guard.sh`, then `if need_pool $POOL; then <build>; fi`.
# need_pool: 0 = not built yet (build it), 1 = built by the current generator (reuse it). A pool from
# an older GEN_VERSION stops the script -- rename / delete it, or set ALLOW_STALE_POOLS=1 to reuse
# it on purpose (e.g. reproducing an old run).
need_pool() {
  local s=0
  python3 scripts/pool_version.py "$1" || s=$?
  case $s in
    0) return 1 ;;
    10) return 0 ;;
    *) if [ "${ALLOW_STALE_POOLS:-0}" = 1 ]; then echo "reusing stale pool $1 (ALLOW_STALE_POOLS=1)"; return 1; fi
       echo "STOP: $1 was built by an older generator (scripts/pool_version.py); delete or rename it," \
            "or ALLOW_STALE_POOLS=1 to reuse it" >&2
       exit 3 ;;
  esac
}
