"""Threshold / eps on the 3-scale blend (5120+4096+3072, equal) with and without the 1 px offset (2026-11-11). Tune on valfix3;
hf14fix2 / Gemini30 / gemdev are checks.  Usage: python3 scripts/pp_probe/tune3s.py SPLIT MAPS [SHARD N];  --summary TAG SPLIT"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
CFG = [(t, e) for t in (0.30, 0.35, 0.40, 0.45) for e in (5, 6, 8)]

if sys.argv[1] == "--summary":
    R = [r for f in sorted(glob.glob(f"scripts/pp_probe/rows_tune3s_{sys.argv[2]}_{sys.argv[3]}*.json")) for r in json.load(open(f))]
    L = [r for r in R if r["label_v"] > 0]
    print(f"{sys.argv[3]} {sys.argv[2]} {len(R)} q   IoU (+off1) / vertices-label median; change vs thr .35 eps 6")
    b = "t0.35 e6"
    for t, e in CFG:
        k = f"t{t:.2f} e{e}"
        print(f"  {k:10s} {np.mean([r[k] for r in R]):.4f} ({np.mean([r[k] - r[b] for r in R]):+.4f})  off1 {np.mean([r[k + ' o1'] for r in R]):.4f} "
              f"({np.mean([r[k + ' o1'] - r[b + ' o1'] for r in R]):+.4f})  {np.median([r['v_' + k] / r['label_v'] for r in L]):.2f}x")
    sys.exit()

import common
from stack import colour_cut, iou, CUR, nv
from curve import label_vertices
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, rasterize
common.WITH_BOX[0] = True
split, maps = sys.argv[1], Path(sys.argv[2])
shard, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
common.MAPS = maps
ms = common.map_split(split)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
rows = []
for job in list(common.jobs(split))[shard::nsh]:
    idx, img, anyann, qs = common.load(job)
    h, w = img.shape[:2]
    for ref, cat, tgt, p4, box in qs:
        CUR[0] = (idx, ref)
        s = p4.shape[1] / w
        P = cv2.resize(p4, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 / 3
        for sz in (5120, 3072):
            a = np.load(maps / f"{ms}_{sz}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
            P += cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 / 3
        r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat)}
        for t in sorted({t for t, _ in CFG}):
            m = colour_cut(img, fill_gaps(_fill_holes(P > t, 4096 / s / s), P), box)
            for tt, e in CFG:
                if tt != t:
                    continue
                k = f"t{t:.2f} e{e}"
                pl = mask_to_polygons(m, 0.005, e, 6.0)
                base = rasterize(pl, (h, w))
                r[k], r["v_" + k] = iou(base, tgt), nv(pl)
                r[k + " o1"] = iou(cv2.dilate(base.astype(np.uint8), k3).astype(bool), tgt)
        rows.append(r)
suf = f"_{shard}" if nsh > 1 else ""
json.dump(rows, open(f"scripts/pp_probe/rows_tune3s_{maps.name}_{split}{suf}.json", "w"))
print("done", split, shard, len(rows))
