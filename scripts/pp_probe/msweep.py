"""Multi-scale blend sweep on the product output (2026-11-10). Production blends 0.7 x 4096 + 0.3 x 2048; raw single-scale
gem2s-x peaks at 4096 (HF14 0.924, 5120 0.922, 6144 0.917; Gemini30 6144 0.859). Does another blend help the product output?
Each blend through the eps-6 low-vertex pipeline (fills, colour cut, eps 6, pieces < 0.5%), with and without the 1 px offset.
Usage: python3 scripts/pp_probe/msweep.py SPLIT MAPS [SHARD NSHARDS];  python3 scripts/pp_probe/msweep.py --summary TAG SPLIT"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2

BLENDS = {
    "4096+2048 (prod)": {4096: 0.7, 2048: 0.3},
    "4096":             {4096: 1.0},
    "4096+3072":        {4096: 0.6, 3072: 0.4},
    "4096+3072+2048":   {4096: 0.5, 3072: 0.25, 2048: 0.25},
    "5120+4096+2048":   {5120: 0.3, 4096: 0.4, 2048: 0.3},
    "5120+4096+3072":   {5120: 1 / 3, 4096: 1 / 3, 3072: 1 / 3},
    "6144+4096+2048":   {6144: 0.3, 4096: 0.4, 2048: 0.3},
}


def summary(tag, split):
    R = [r for f in sorted(glob.glob(f"scripts/pp_probe/rows_msweep_{tag}_{split}*.json")) for r in json.load(open(f))]
    b = "4096+2048 (prod)"
    print(f"{split} {tag} {len(R)} q   eps6 / eps6+off1, change vs prod blend")
    for k in BLENDS:
        if k not in R[0]:
            continue
        for o in ("", " off1"):
            d = np.array([r[k + o] - r[b + o] for r in R])
            L = [r for r in R if r.get("label_v", 0) > 0 and "v_" + k in r]
            vr = f"  {np.median([r['v_' + k] / r['label_v'] for r in L]):.2f}x" if L else ""
            print(f"  {k + o:24s} {np.mean([r[k + o] for r in R]):.4f}  {d.mean():+.4f} ({(d > 1e-3).sum()} up / {(d < -1e-3).sum()} down){vr}")


if __name__ == "__main__" and sys.argv[1] == "--summary":
    summary(sys.argv[2], sys.argv[3]); sys.exit()

import common
from stack import colour_cut, iou, CUR, nv
from curve import label_vertices
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, rasterize
common.WITH_BOX[0] = True

if __name__ == "__main__":
    split, maps = sys.argv[1], Path(sys.argv[2])
    shard, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    common.MAPS = maps
    have = {sz for sz in (2048, 3072, 4096, 5120, 6144) if (maps / f"{common.map_split(split)}_{sz}" / "metrics.json").exists()}
    blends = {k: v for k, v in BLENDS.items() if set(v) <= have}
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rows = []
    for job in list(common.jobs(split))[shard::nsh]:
        idx, img, anyann, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            CUR[0] = (idx, ref)
            cache = {4096: p4}
            for sz in have - {4096}:
                cache[sz] = np.load(maps / f"{common.map_split(split)}_{sz}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
            s = p4.shape[1] / w
            r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat)}
            for name, wts in blends.items():
                P = sum(wt * cv2.resize(cache[sz], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                        for sz, wt in wts.items())
                m = colour_cut(img, fill_gaps(_fill_holes(P > 0.35, 4096 / s / s), P), box)
                base = rasterize(mask_to_polygons(m, 0.005, 6, 6.0), (h, w))
                r[name] = iou(base, tgt)
                r["v_" + name] = nv(mask_to_polygons(m, 0.005, 6, 6.0))
                r[name + " off1"] = iou(cv2.dilate(base.astype(np.uint8), k3).astype(bool), tgt)
            rows.append(r)
    suf = f"_{shard}" if nsh > 1 else ""
    json.dump(rows, open(f"scripts/pp_probe/rows_msweep_{maps.name}_{split}{suf}.json", "w"))
    print("done", split, shard, len(rows))
