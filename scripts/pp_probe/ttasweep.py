"""Flip TTA vs extra scales at equal compute, on the full product v3 pipeline (2026-11-11). Map dirs: {split}_{size} and
{split}_{size}t{n} (scripts/gpu_maps_tta.sh: n dihedral views averaged). Compute in 4096^2 units (a view at size S costs
(S/4096)^2): v1 4096+2048 = 1.25, v2/v3 5120+4096+3072 = 3.1.
Each blend -> threshold 0.35, gap fills, colour cut, cell snap, eps 6, 1 px offset (PRODUCT_V3); IoU and vertices / label.
Usage: python3 scripts/pp_probe/ttasweep.py SPLIT MAPS [SHARD N];  python3 scripts/pp_probe/ttasweep.py --summary TAG SPLIT"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2

BLENDS = {  # name: ({map dir suffix: weight}, compute units)
    "v3 5120+4096+3072":       ({"5120": 1 / 3, "4096": 1 / 3, "3072": 1 / 3}, 3.13),
    "v1 4096+2048":            ({"4096": 0.7, "2048": 0.3}, 1.25),
    "4096t2+2048t2":           ({"4096t2": 0.7, "2048t2": 0.3}, 2.5),
    "4096t2+2048":             ({"4096t2": 0.7, "2048": 0.3}, 2.25),
    "4096t4+2048t2":           ({"4096t4": 0.7, "2048t2": 0.3}, 4.5),
    "4096t2+3072":             ({"4096t2": 0.6, "3072": 0.4}, 2.56),
    "5120+4096t2+3072":        ({"5120": 1 / 3, "4096t2": 1 / 3, "3072": 1 / 3}, 4.13),
    "4096t4":                  ({"4096t4": 1.0}, 4.0),
}


def summary(tag, split):
    R = [r for f in sorted(glob.glob(f"scripts/pp_probe/rows_ttasweep_{tag}_{split}*.json")) for r in json.load(open(f))]
    L = [r for r in R if r["label_v"] > 0]
    b = "v3 5120+4096+3072"
    print(f"{split} {tag} {len(R)} q   product v3 pipeline; change vs the v3 blend; compute (4096^2 = 1)")
    for k, (_, c) in BLENDS.items():
        if k not in R[0]:
            continue
        d = np.array([r[k] - r[b] for r in R])
        print(f"  {k:20s} {np.mean([r[k] for r in R]):.4f}  {d.mean():+.4f} ({(d > 1e-3).sum()} up / {(d < -1e-3).sum()} down)"
              f"  {np.median([r['v_' + k] / r['label_v'] for r in L]):.2f}x  compute {c:.2f}")


if __name__ == "__main__" and sys.argv[1] == "--summary":
    summary(sys.argv[2], sys.argv[3]); sys.exit()

import common
from stack import colour_cut, iou, CUR, nv
from curve import label_vertices
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, rasterize, offset_polys, snap_to_cells
common.WITH_BOX[0] = True

if __name__ == "__main__":
    split, maps = sys.argv[1], Path(sys.argv[2])
    shard, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    common.MAPS = maps
    ms = common.map_split(split)
    have = {d.name[len(ms) + 1:] for d in maps.glob(f"{ms}_*") if (d / "metrics.json").exists()}
    blends = {k: v for k, v in BLENDS.items() if set(v[0]) <= have}
    rows = []
    for job in list(common.jobs(split))[shard::nsh]:
        idx, img, anyann, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            CUR[0] = (idx, ref)
            s = p4.shape[1] / w
            cache = {"4096": cv2.resize(p4, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255}

            def get(k):
                if k not in cache:
                    a = np.load(maps / f"{ms}_{k}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
                    cache[k] = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                return cache[k]
            r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat)}
            for name, (wts, _) in blends.items():
                P = sum(wt * get(k) for k, wt in wts.items())
                m = snap_to_cells(colour_cut(img, fill_gaps(_fill_holes(P > 0.35, 4096 / s / s), P), box), img, s)
                pl = offset_polys(mask_to_polygons(m, 0.005, 6.0), 1.0)
                r[name], r["v_" + name] = iou(rasterize(pl, (h, w)), tgt), nv(pl)
            rows.append(r)
    suf = f"_{shard}" if nsh > 1 else ""
    json.dump(rows, open(f"scripts/pp_probe/rows_ttasweep_{maps.name}_{split}{suf}.json", "w"))
    print("done", split, shard, len(rows), sorted(blends))
