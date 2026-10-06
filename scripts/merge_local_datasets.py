#!/usr/bin/env python3
"""Merge local-data directories with deterministic collision-free names."""

import argparse
import json
import sys
from pathlib import Path

from build_mix import merge_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sources = [str(Path(source).resolve()) for source in args.sources]
    count = merge_dirs(sources, args.out)
    manifest = {"sources": sources, "images": count,
                "ordering": "source order, then sorted annotation filename"}
    # synthetic sources' generator version (scripts/pool_version.py); real / Gemini have none
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pool_version import version
    vs = {v for v in (version(src) for src in sources) if v not in (None, "none")}
    if vs:
        manifest["generator_version"] = vs.pop() if len(vs) == 1 else "mixed"
    path = Path(args.out) / "merge_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
