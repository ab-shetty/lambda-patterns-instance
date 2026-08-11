#!/usr/bin/env python3
"""Measure the disconnected-repetition gap between a pool and the real plans.

Per *selection* (every labelled instance is a possible user rectangle, exactly as
`evaluate_refunet_selection.py` enumerates them), compute:

  share_local  the fraction of the target union lying in the connected component
               that contains the user's rectangle. 1.0 means the correct answer
               is entirely "outline the blob you are pointing at".
  single       share_local >= 0.999, i.e. the reference component IS the answer.

Reproduces the table in `codex_doc.md` (synthetic 1600: 0.893 / 84%; HF14: 0.537
/ 27%) so a new selector can be checked against the same definition it targets.

  PYTHONPATH=. python3 scripts/connectivity_stats.py --hf14
  PYTHONPATH=. python3 scripts/connectivity_stats.py --local data/synthetic/hf20k
"""

import argparse
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from refmask2former.dataset import render_instance_mask, sample_reference_box

HOLDOUT = [12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24]
VALIDATION = [4, 5, 6, 8, 9, 10, 13, 15, 17, 19, 20, 21, 22, 26]


def family_components(union, res=2048, shape=None):
    """Connected components of a family union, measured at reduced resolution.

    res=2048 reproduces `codex_doc.md` exactly (HF14 0.537 / 27%); at 1024 nearby
    components merge and the gap looks smaller than it is (0.558 / 33%)."""
    h, w = union.shape
    s = res / max(h, w)
    if s < 1.0:
        small = cv2.resize(union.astype(np.uint8), (max(1, round(w * s)),
                                                    max(1, round(h * s))),
                           interpolation=cv2.INTER_NEAREST)
    else:
        small = union.astype(np.uint8)
    n, lab = cv2.connectedComponents(small, connectivity=8)
    return n - 1, lab, small


def record_rows(image_idx, anns, h0, w0):
    """One row per selection: share_local + component count for its family."""
    masks = [render_instance_mask(a["segmentation"], h0, w0).astype(bool) for a in anns]
    cats = [a.get("category_name", "pattern") for a in anns]
    unions, comps = {}, {}
    for c in set(cats):
        u = np.zeros((h0, w0), bool)
        for m, cc in zip(masks, cats):
            if cc == c:
                u |= m
        unions[c] = u
        comps[c] = family_components(u)

    rows = []
    for ref_idx, (rm, c) in enumerate(zip(masks, cats)):
        if not rm.any():
            continue
        rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
        x, y, w, h = sample_reference_box(rm, 128, 512, rng=rng)
        n_comp, lab, small = comps[c]
        sh, sw = lab.shape
        # Map the rectangle's centre into the reduced-resolution label image.
        cy = int(np.clip(round((y + h / 2) * sh / h0), 0, sh - 1))
        cx = int(np.clip(round((x + w / 2) * sw / w0), 0, sw - 1))
        lid = lab[cy, cx]
        if lid == 0:                      # centre fell off the union after resize
            ys, xs = np.nonzero(lab)
            if not len(ys):
                continue
            d = (ys - cy) ** 2 + (xs - cx) ** 2
            lid = lab[ys[d.argmin()], xs[d.argmin()]]
        total = int(small.sum())
        local = int((lab == lid).sum())
        rows.append({"image": image_idx, "ref": ref_idx, "category": c,
                     "n_components": int(n_comp),
                     "share_local": local / max(1, total)})
    return rows


def summarize(rows, name):
    if not rows:
        print(f"{name}: no selections")
        return
    sl = np.array([r["share_local"] for r in rows])
    single = (sl >= 0.999).mean()
    nc = np.array([r["n_components"] for r in rows])
    print(f"{name:<38}{len(rows):>7}{sl.mean():>14.3f}{100 * single:>13.0f}%"
          f"{np.median(nc):>13.0f}")
    return sl.mean(), single


def iter_local(root, limit=None):
    root = Path(root)
    files = sorted((root / "annotations").glob("*.json"))
    if limit:
        files = files[:limit]
    for i, f in enumerate(files):
        ann = json.load(f.open())
        img = ann.get("image", {})
        yield i, ann.get("annotations", []), int(img["height"]), int(img["width"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf14", action="store_true", help="the acceptance holdout")
    ap.add_argument("--hf-val", action="store_true", help="the validation complement")
    ap.add_argument("--local", nargs="*", default=[], help="local pool dirs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-dir", default="./data")
    args = ap.parse_args()

    print(f"\n{'pool':<38}{'selections':>7}{'share_local':>14}{'single':>14}"
          f"{'med comps':>13}")

    if args.hf14 or args.hf_val:
        from refmask2former import load_parquet_records
        ds = load_parquet_records(cache_dir=args.cache_dir,
                                  config="real-world-test", split="test")
        for flag, idxs, nm in ((args.hf14, HOLDOUT, "HF14 (acceptance)"),
                               (args.hf_val, VALIDATION, "HF-val (tuning)")):
            if not flag:
                continue
            rows = []
            for i in idxs:
                rec = ds[i]
                a = rec["annotations"]
                a = json.loads(a) if isinstance(a, str) else a
                with Image.open(io.BytesIO(rec["image"])) as im:
                    w0, h0 = im.size
                rows += record_rows(i, a, h0, w0)
            summarize(rows, nm)

    for root in args.local:
        rows = []
        for i, a, h0, w0 in iter_local(root, args.limit):
            rows += record_rows(i, a, h0, w0)
        summarize(rows, Path(root).name)
    print()


if __name__ == "__main__":
    main()
