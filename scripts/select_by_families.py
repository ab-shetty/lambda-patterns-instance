#!/usr/bin/env python3
"""Hard-link the sheets of a local pool that have at least N labelled families.

    python3 scripts/select_by_families.py --src POOL --out SUBSET --min-families 3 [--max 500]

Sheets are taken in sorted id order, so the subset is deterministic. Prints the count.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-families", type=int, default=3)
    ap.add_argument("--max", type=int, default=0, help="stop after this many sheets (0 = all)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((src / "annotations").glob("*.json")):
        ann = json.loads(f.read_text())
        if len({x["family"] for x in ann["annotations"]}) < a.min_families:
            continue
        for s, d in ((f, out / "annotations" / f.name),
                     (src / "images" / ann["image"]["file_name"], out / "images" / ann["image"]["file_name"])):
            if not d.exists():
                os.link(s, d)
        n += 1
        if a.max and n >= a.max:
            break
    print(f"{out}: {n} sheets with >= {a.min_families} families")


if __name__ == "__main__":
    main()
