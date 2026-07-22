#!/usr/bin/env python3
"""Merge local-data directories with deterministic collision-free names."""

import argparse
import json
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
    path = Path(args.out) / "merge_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
