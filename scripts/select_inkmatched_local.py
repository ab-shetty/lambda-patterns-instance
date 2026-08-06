#!/usr/bin/env python3
"""Select synthetic images whose ink density matches real plans.

`select_toparea_local.py` ranks by labelled-area fraction, which also selects for
drawing density: the resulting pool has ~13.9% median ink coverage against ~5.5%
for real plans. The model therefore trains mostly on dense sheets and
over-predicts on sparse, faint ones -- the largest remaining HF14 failures
(images 14, 12, 7) all sit at 1.7-4.7% ink.

This selector instead quantile-matches the synthetic ink distribution to a
reference set of real plans, while keeping the mode quotas and a floor on
labelled area so the examples stay useful.

The reference set must NOT be the acceptance holdout. Pass the validation plans
and/or the real training pools; `--reference-hf-indices` defaults to the 14
validation indices, never the HF14 fourteen.

Usage:
    python3 scripts/select_inkmatched_local.py \
        --source data/synthetic/hf20k \
        --reference-local data/roboflow/floz-real-pool-v2-clean \
                          data/roboflow/floz-genreal-v1-clean \
        --out data/synthetic/inkmatched1600 --n 1600 \
        --mode-quotas elevation=889,roof_plan=540,freeform=171
"""

import argparse
import io
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

VALIDATION_INDICES = "4,5,6,8,9,10,13,15,17,19,20,21,22,26"
HF14 = {12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24}


def ink_of(gray):
    """Fraction of clearly-inked pixels. Scale-insensitive enough to downsample."""
    g = np.asarray(gray, dtype=np.float32) / 255.0
    return float((g < 0.6).mean())


def _ink_path(path):
    im = Image.open(path).convert("L")
    im.thumbnail((640, 640), Image.BILINEAR)
    return ink_of(im)


def _probe(ann_path):
    """(ink, mode, area_frac, ann_path, img_path) for one local-data record."""
    doc = json.load(open(ann_path))
    root = Path(ann_path).parent.parent
    img = root / "images" / doc["image"]["file_name"]
    if not img.exists():
        return None
    w = float(doc["image"].get("width") or 0)
    h = float(doc["image"].get("height") or 0)
    if w <= 0 or h <= 0:
        return None
    area = sum(float(a.get("area") or 0.0) for a in doc.get("annotations") or [])
    return (_ink_path(img), str(doc.get("mode") or "unknown"),
            area / (w * h), str(ann_path), str(img))


def parse_quotas(text):
    if not text:
        return {}
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="synthetic local-data pool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--mode-quotas", default="")
    ap.add_argument("--reference-local", nargs="*", default=[],
                    help="local-data dirs of real plans to match")
    ap.add_argument("--reference-hf-indices", default=VALIDATION_INDICES,
                    help="HF real-world-test indices to match (validation only)")
    ap.add_argument("--min-area-frac", type=float, default=0.05,
                    help="drop synthetic with almost nothing labelled")
    ap.add_argument("--max-ink", type=float, default=0.0,
                    help="if >0, keep only records under this ink fraction and "
                         "then rank by labelled area instead of quantile-matching. "
                         "Pure quantile matching also halves labelled area "
                         "(0.387 -> 0.286 median), which costs more than the "
                         "density match gains; this keeps faint AND well-labelled.")
    ap.add_argument("--probe-cache", default="",
                    help="JSON path to cache/reuse the expensive ink probe")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    args = ap.parse_args()

    idxs = [int(v) for v in args.reference_hf_indices.split(",") if v.strip()]
    leaked = sorted(set(idxs) & HF14)
    if leaked:
        raise SystemExit(f"refusing to match against HF14 acceptance images: {leaked}")

    # ---- reference ink distribution -------------------------------------
    ref = []
    if idxs:
        from refmask2former import load_parquet_records
        ds = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                  config="real-world-test", split="test")
        for i in idxs:
            im = Image.open(io.BytesIO(ds[i]["image"])).convert("L")
            im.thumbnail((640, 640), Image.BILINEAR)
            ref.append(ink_of(im))
    for d in args.reference_local:
        for p in sorted(glob(f"{d}/images/*.png")):
            ref.append(_ink_path(p))
    if not ref:
        raise SystemExit("no reference images")
    ref = np.array(ref)
    print(f"reference: n={len(ref)} ink median={np.median(ref)*100:.2f}% "
          f"p10={np.percentile(ref,10)*100:.2f}% p90={np.percentile(ref,90)*100:.2f}%")

    # ---- probe the synthetic pool ---------------------------------------
    anns = sorted(glob(f"{args.source}/annotations/*.json"))
    if args.probe_cache and os.path.exists(args.probe_cache):
        probed = [tuple(x) for x in json.load(open(args.probe_cache))]
        print(f"loaded probe cache: {len(probed)} records")
    else:
        print(f"probing {len(anns)} synthetic records with {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            probed = [p for p in ex.map(_probe, anns, chunksize=64) if p]
        if args.probe_cache:
            json.dump(probed, open(args.probe_cache, "w"))
    probed = [p for p in probed if p[2] >= args.min_area_frac]
    print(f"usable after area floor {args.min_area_frac}: {len(probed)}")

    by_mode = defaultdict(list)
    for p in probed:
        by_mode[p[1]].append(p)
    for m in by_mode:
        by_mode[m].sort(key=lambda p: p[0])

    quotas = parse_quotas(args.mode_quotas) or {None: args.n}

    # ---- quantile match --------------------------------------------------
    chosen = []
    for mode, count in quotas.items():
        pool = by_mode.get(mode, []) if mode else sorted(probed, key=lambda p: p[0])
        if not pool:
            print(f"  WARN: no candidates for mode {mode}")
            continue

        if args.max_ink > 0:
            faint = [p for p in pool if p[0] <= args.max_ink]
            if len(faint) < count:
                print(f"  WARN: mode {mode} only {len(faint)} under ink "
                      f"{args.max_ink}; backfilling by ink then area")
                rest = sorted((p for p in pool if p[0] > args.max_ink),
                              key=lambda p: p[0])
                faint = faint + rest[:count - len(faint)]
            faint.sort(key=lambda p: -p[2])          # highest labelled area first
            take = faint[:count]
            chosen.extend(take)
            print(f"  mode {mode}: took {len(take)} of {len(pool)} "
                  f"(ink<={args.max_ink}, ranked by area)")
            continue

        inks = np.array([p[0] for p in pool])
        used = set()
        targets = np.quantile(ref, np.linspace(0.02, 0.98, count))
        for t in targets:
            order = np.argsort(np.abs(inks - t))
            for j in order:
                if j not in used:
                    used.add(j)
                    chosen.append(pool[j])
                    break
        print(f"  mode {mode}: took {len(used)} of {len(pool)}")

    # ---- write -----------------------------------------------------------
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    modes = Counter()
    got = []
    for i, (ink, mode, area, ann_path, img_path) in enumerate(chosen):
        stem = f"synth_{i:06d}"
        suffix = Path(img_path).suffix.lower()
        shutil.copy2(img_path, out / "images" / f"{stem}{suffix}")
        doc = json.load(open(ann_path))
        doc.setdefault("image", {})["file_name"] = f"{stem}{suffix}"
        doc["source_ink"] = ink
        doc["source_image"] = Path(img_path).name
        json.dump(doc, open(out / "annotations" / f"{stem}.json", "w"),
                  separators=(",", ":"))
        modes[mode] += 1
        got.append(ink)
    got = np.array(got)
    print(json.dumps({"n": len(got), "modes": dict(modes),
                      "ink_median_pct": round(float(np.median(got)) * 100, 2),
                      "ink_p10_pct": round(float(np.percentile(got, 10)) * 100, 2),
                      "ink_p90_pct": round(float(np.percentile(got, 90)) * 100, 2),
                      "reference_median_pct": round(float(np.median(ref)) * 100, 2)},
                     indent=2))


if __name__ == "__main__":
    main()
