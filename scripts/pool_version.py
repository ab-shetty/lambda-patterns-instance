#!/usr/bin/env python3
"""Is a synthetic pool built by the CURRENT generator defaults?

    python3 scripts/pool_version.py POOL        exit 0 current, 10 not built yet, 20 stale
    python3 scripts/pool_version.py --current   print GEN_VERSION

The version is read from generation_manifest.json (generate_synthetic_fc.py), hard_pool_manifest.json
(build_hard_pool.py, copied from its fresh pool) or merge_manifest.json (merge_local_datasets.py,
the synthetic sources' version). A directory with files but no version is stale. Used by
scripts/pool_guard.sh so a run never silently trains on a pool from older defaults.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def current():
    src = (ROOT / "generate_synthetic_fc.py").read_text()
    return re.search(r'^GEN_VERSION = "([^"]+)"', src, re.M).group(1)


def version(pool):
    p = Path(pool)
    for name in ("generation_manifest.json", "hard_pool_manifest.json", "merge_manifest.json"):
        f = p / name
        if f.exists():
            return json.loads(f.read_text()).get("generator_version", "none")
    if p.exists() and any(p.iterdir()):
        return "none"
    return None


def main():
    if sys.argv[1] == "--current":
        print(current())
        return 0
    v, cur = version(sys.argv[1]), current()
    if v is None:
        return 10
    if v == cur:
        return 0
    print(f"{sys.argv[1]}: built by generator version {v}, current is {cur}")
    return 20


if __name__ == "__main__":
    sys.exit(main())
