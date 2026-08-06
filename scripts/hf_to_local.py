#!/usr/bin/env python3
"""Export a `floz-synth-v5` parquet split into the repo's local-data format.

The generator pools the selection scripts were built against (`faintcad2500`,
`cadneg2500`) only ever existed on an earlier VM and their generator flags were
never committed, so they cannot be rebuilt. The 20k-row default config on
HuggingFace is the same schema and is the reproducible stand-in: this writes it
out as `images/*.png` + `annotations/*.json` so `select_toparea_local.py`,
`merge_local_datasets.py`, and `--local-data` all consume it unchanged.

One schema gap is filled here: the parquet annotations carry no `area`, but
`select_toparea_local.py` ranks candidates by summed annotation area over image
area. Area is therefore computed the way `generate_synthetic_v5.py` defines it —
outer ring area minus the area of every subsequent (hole) ring.

Usage:
    python scripts/hf_to_local.py --out data/synthetic/hf20k --workers 32
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

_DS = None


def _shoelace(flat):
    """Absolute polygon area from a flat [x,y,x,y,...] ring."""
    n = len(flat) // 2
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x0, y0 = flat[2 * i], flat[2 * i + 1]
        j = (i + 1) % n
        x1, y1 = flat[2 * j], flat[2 * j + 1]
        acc += x0 * y1 - x1 * y0
    return abs(acc) / 2.0


def _instance_area(segmentation):
    """Outer ring minus holes, matching the generator's `area` semantics."""
    if not segmentation:
        return 0.0
    area = _shoelace(segmentation[0])
    for hole in segmentation[1:]:
        area -= _shoelace(hole)
    return max(area, 0.0)


def _init(repo, config, split, cache_dir):
    global _DS
    from datasets import load_dataset
    _DS = load_dataset(repo, config, split=split, cache_dir=cache_dir)


def _write_one(job):
    idx, out = job
    row = _DS[idx]
    anns = row["annotations"]
    if isinstance(anns, str):
        anns = json.loads(anns)
    for a in anns:
        if a.get("area") is None:
            a["area"] = _instance_area(a.get("segmentation") or [])

    stem = f"synth_{idx:06d}"
    with open(f"{out}/images/{stem}.png", "wb") as f:
        f.write(row["image"])
    doc = {
        "image": {"file_name": f"{stem}.png",
                  "width": int(row["width"]), "height": int(row["height"])},
        "mode": row.get("mode") or "unknown",
        "source_dataset": "hf:floz-synth-v5",
        "source_image": row.get("filename") or stem,
        "annotations": anns,
    }
    with open(f"{out}/annotations/{stem}.json", "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    return doc["mode"], len(anns)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="abshetty/floz-synth-v5")
    ap.add_argument("--config", default=None, help="None = default 20k train config")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/images", exist_ok=True)
    os.makedirs(f"{args.out}/annotations", exist_ok=True)

    from datasets import load_dataset
    n = len(load_dataset(args.repo, args.config, split=args.split,
                         cache_dir=args.cache_dir))
    if args.limit:
        n = min(n, args.limit)
    print(f"exporting {n} rows -> {args.out}")

    from collections import Counter
    modes, n_inst = Counter(), 0
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init,
            initargs=(args.repo, args.config, args.split, args.cache_dir)) as ex:
        for i, (mode, k) in enumerate(
                ex.map(_write_one, ((j, args.out) for j in range(n)),
                       chunksize=32)):
            modes[mode] += 1
            n_inst += k
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{n}")

    print(f"wrote {sum(modes.values())} images, {n_inst} instances -> {args.out}")
    print("modes:", dict(modes))


if __name__ == "__main__":
    main()
