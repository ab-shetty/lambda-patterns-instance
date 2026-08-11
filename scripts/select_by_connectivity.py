#!/usr/bin/env python3
"""Split one generated pool into a CONNECTED and a DISCONNECTED arm.

Selecting disconnected records out of `hf20k` is impossible without also swapping
the mode mix (99.8% of freeform is disconnected, 10.9% of elevation), which would
confound disconnection with mode. Generating one large single-mode pool and
splitting it by connectivity keeps the generator, recipe and mode identical, so
`share_local` is the only thing that differs between the arms.

Ranks by per-image mean `share_local` (the same quantity `connectivity_stats.py`
reports) and takes from the bottom (`--take low`, disconnected) or the top
(`--take high`, connected). `--match-area` then trims both arms toward a common
labelled-area distribution, because selection otherwise trades area away and area
has moved results before.

    python3 scripts/select_by_connectivity.py --source data/synthetic/elevpool4000 \
        --out-low data/synthetic/elev889_disc --out-high data/synthetic/elev889_conn \
        --n 889 --workers 48
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.connectivity_stats import record_rows  # noqa: E402


def _score(ann_path: str):
    """(path, mean share_local, area_frac, mode, n_selections) or None."""
    try:
        with open(ann_path) as f:
            ann = json.load(f)
    except Exception:
        return None
    im = ann.get("image") or {}
    h, w = int(im.get("height") or 0), int(im.get("width") or 0)
    anns = ann.get("annotations") or []
    if not anns or h <= 0 or w <= 0:
        return None
    rows = record_rows(0, anns, h, w)
    if not rows:
        return None
    area = sum(float(a.get("area") or 0.0) for a in anns) / (h * w)
    return (ann_path, float(np.mean([r["share_local"] for r in rows])), area,
            str(ann.get("mode") or "unknown"), len(rows))


def _write(rows, out: Path, overwrite: bool):
    if out.exists() and overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for i, (ap_, share, area, mode, _n) in enumerate(rows):
        src_ann = Path(ap_)
        with src_ann.open() as f:
            ann = json.load(f)
        src_img = src_ann.parent.parent / "images" / ann["image"]["file_name"]
        stem = f"synth_{i:06d}"
        dst_img = out / "images" / f"{stem}{src_img.suffix.lower()}"
        shutil.copy2(src_img, dst_img)
        ann["source_dataset"] = str(src_ann.parent.parent)
        ann["source_image"] = ann["image"]["file_name"]
        ann["image"]["file_name"] = dst_img.name
        ann["selection_share_local"] = share
        ann["selection_score"] = area
        with (out / "annotations" / f"{stem}.json").open("w") as f:
            json.dump(ann, f, separators=(",", ":"))
    sh = np.array([r[1] for r in rows])
    ar = np.array([r[2] for r in rows])
    print(f"  {out.name:<26} n={len(rows):<5} share_local mean={sh.mean():.3f} "
          f"median={np.median(sh):.3f} | area mean={ar.mean():.3f} median={np.median(ar):.3f}")


def _match_area(low, high, n, bins=12):
    """Pick n from each arm under a COMMON labelled-area histogram.

    Both arms get the same per-bin counts, so the two pools carry the same area
    distribution and `share_local` is left as the only systematic difference. The
    per-bin quota is decided here and never re-trimmed afterwards -- taking a
    global top-n by share_local after matching would silently undo the match.
    """
    allo = np.array([r[2] for r in low + high])
    edges = np.quantile(allo, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    lo_bins, hi_bins, cap = [], [], []
    for b in range(bins):
        lo_b = sorted([r for r in low if edges[b] <= r[2] < edges[b + 1]],
                      key=lambda r: r[1])        # most disconnected first
        hi_b = sorted([r for r in high if edges[b] <= r[2] < edges[b + 1]],
                      key=lambda r: -r[1])       # most connected first
        lo_bins.append(lo_b)
        hi_bins.append(hi_b)
        cap.append(min(len(lo_b), len(hi_b)))

    total = sum(cap)
    if total < n:
        raise SystemExit(f"area matching can only supply {total} per arm "
                         f"(need {n}); generate a larger pool or --no-match-area")
    # Scale every bin down proportionally so both arms keep the same shape.
    take = [int(c * n / total) for c in cap]
    b = 0
    while sum(take) < n:                          # hand out the rounding remainder
        if take[b % bins] < cap[b % bins]:
            take[b % bins] += 1
        b += 1
    return ([r for bl, k in zip(lo_bins, take) for r in bl[:k]],
            [r for bh, k in zip(hi_bins, take) for r in bh[:k]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-low", required=True, help="disconnected arm")
    ap.add_argument("--out-high", required=True, help="connected arm")
    ap.add_argument("--n", type=int, default=889)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--match-area", action="store_true", default=True)
    ap.add_argument("--no-match-area", dest="match_area", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    files = [str(p) for p in sorted((Path(args.source) / "annotations").glob("*.json"))]
    print(f"scoring {len(files)} records ...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(_score, files, chunksize=32) if r]
    rows.sort(key=lambda r: r[1])
    sh = np.array([r[1] for r in rows])
    print(f"pool share_local: mean {sh.mean():.3f}  p10 {np.quantile(sh,0.1):.3f}  "
          f"p90 {np.quantile(sh,0.9):.3f}")

    if args.match_area:
        # Take a generous candidate slice from each end, then area-match down to n.
        span = min(len(rows) // 2, max(args.n * 2, args.n + 200))
        low, high = _match_area(rows[:span], rows[-span:], args.n)
    else:
        low, high = rows[:args.n], rows[-args.n:]

    print("writing arms:")
    _write(low, Path(args.out_low), args.overwrite)
    _write(high, Path(args.out_high), args.overwrite)


if __name__ == "__main__":
    main()
