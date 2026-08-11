#!/usr/bin/env python3
"""Select synthetic images whose pattern families repeat in DISCONNECTED places.

`toparea1600_balanced` teaches the wrong task: measured per selection, 84% of its
targets sit entirely in the connected component holding the user's rectangle
(share_local 0.893) versus 27% / 0.537 on HF14. For 44% of records the correct
answer is "outline the blob you are pointing at", and the `--anchor` probe showed
the model takes that shortcut when it is available.

This ranks the same way `select_toparea_local.py` does -- by labelled-area
fraction, with the same mode quotas -- but first keeps only images where enough
pattern families are genuinely disconnected, so the constraint is the only
difference between the two pools.

A family is disconnected when the union of its instances has more than one
8-connected component, measured with the longest side at --res (2048 reproduces
the numbers in codex_doc.md; at 1024 nearby components merge).

    python3 scripts/select_disconnected_local.py \
      --sources data/synthetic/hf20k --out data/synthetic/disc1600_balanced \
      --n 1600 --mode-quotas elevation=889,roof_plan=540,freeform=171 --workers 48
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

_RES = 2048


def _family_stats(ann_path: str):
    """(path, mode, area_score, n_families, n_disconnected, ann_count) or None."""
    try:
        with open(ann_path) as f:
            ann = json.load(f)
    except Exception:
        return None
    image = ann.get("image") or {}
    fn = image.get("file_name")
    w0, h0 = float(image.get("width") or 0), float(image.get("height") or 0)
    if not fn or w0 <= 0 or h0 <= 0:
        return None
    anns = ann.get("annotations") or []
    if not anns:
        return None

    s = _RES / max(h0, w0)
    nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))

    fams = {}
    for a in anns:
        fams.setdefault(a.get("category_name", "pattern"), []).append(a)

    n_disc = 0
    for _cat, group in fams.items():
        u = np.zeros((nh, nw), np.uint8)
        for a in group:
            seg = a.get("segmentation") or []
            if not seg:
                continue
            outer = (np.asarray(seg[0], np.float32).reshape(-1, 2) * s).astype(np.int32)
            cv2.fillPoly(u, [outer], 1)
            for hole in seg[1:]:
                hp = (np.asarray(hole, np.float32).reshape(-1, 2) * s).astype(np.int32)
                cv2.fillPoly(u, [hp], 0)
        n, _ = cv2.connectedComponents(u, connectivity=8)
        if n - 1 > 1:
            n_disc += 1

    area = sum(float(a.get("area") or 0.0) for a in anns)
    return (ann_path, str(ann.get("mode") or "unknown"), area / (w0 * h0),
            len(fams), n_disc, len(anns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--mode-quotas", default=None)
    ap.add_argument("--min-disconnected-frac", type=float, default=1.0,
                    help="fraction of an image's families that must be "
                         "disconnected (1.0 = all of them)")
    ap.add_argument("--min-families", type=int, default=1)
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stats-only", action="store_true",
                    help="report the pool's disconnection distribution and exit")
    args = ap.parse_args()

    global _RES
    _RES = args.res

    ann_files = []
    for src in args.sources:
        ann_files += [str(p) for p in sorted((Path(src) / "annotations").glob("*.json"))]
    print(f"scanning {len(ann_files)} annotations with {args.workers} workers ...")

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_set_res, initargs=(args.res,)) as ex:
        rows = [r for r in ex.map(_family_stats, ann_files, chunksize=64) if r]

    fracs = np.array([r[4] / max(1, r[3]) for r in rows])
    print(f"images scanned: {len(rows)}")
    for t in (0.0, 0.5, 1.0):
        print(f"  images with >= {t:.0%} of families disconnected: "
              f"{(fracs >= t).sum():6d}  ({100 * (fracs >= t).mean():.1f}%)")
    if args.stats_only:
        return

    keep = [r for r, fr in zip(rows, fracs)
            if fr >= args.min_disconnected_frac and r[3] >= args.min_families]
    print(f"eligible after constraint: {len(keep)}")
    keep.sort(key=lambda r: r[2], reverse=True)

    quotas = None
    if args.mode_quotas:
        quotas = {}
        for part in args.mode_quotas.split(","):
            if part.strip():
                m, c = part.split("=")
                quotas[m.strip()] = int(c)
        if sum(quotas.values()) != args.n:
            raise SystemExit(f"--mode-quotas must sum to --n ({args.n})")

    if quotas:
        selected = []
        for mode, count in quotas.items():
            pool = [r for r in keep if r[1] == mode]
            if len(pool) < count:
                raise SystemExit(f"only {len(pool)} eligible for mode={mode}, "
                                 f"need {count}. Lower --min-disconnected-frac "
                                 f"or relax the quotas.")
            selected += pool[:count]
        selected.sort(key=lambda r: r[2], reverse=True)
    else:
        selected = keep[:args.n]
    if len(selected) < args.n:
        raise SystemExit(f"only found {len(selected)} candidates for n={args.n}")

    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for i, (ap_, mode, score, nfam, ndisc, nann) in enumerate(selected):
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
        ann["selection_score"] = score
        ann["selection_families"] = nfam
        ann["selection_disconnected_families"] = ndisc
        with (out / "annotations" / f"{stem}.json").open("w") as f:
            json.dump(ann, f, separators=(",", ":"))
        manifest_rows.append({"rank": i, "score": score, "mode": mode,
                              "families": nfam, "disconnected_families": ndisc,
                              "ann_count": nann, "source_ann": str(src_ann)})

    scores = [r["score"] for r in manifest_rows]
    modes = Counter(r["mode"] for r in manifest_rows)
    manifest = {"n": len(manifest_rows), "sources": args.sources,
                "min_disconnected_frac": args.min_disconnected_frac,
                "res": args.res,
                "score_min": min(scores), "score_median": sorted(scores)[len(scores) // 2],
                "score_mean": sum(scores) / len(scores), "score_max": max(scores),
                "modes": dict(modes), "rows": manifest_rows}
    with (out / "selection_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: manifest[k] for k in
                      ("n", "score_min", "score_median", "score_mean",
                       "score_max", "modes")}, indent=2))


def _set_res(res):
    global _RES
    _RES = res


if __name__ == "__main__":
    main()
