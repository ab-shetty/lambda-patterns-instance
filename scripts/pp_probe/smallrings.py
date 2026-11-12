"""Where HF14's extra vertices are (2026-11-12, data/diag/worstv_hf14_*.jpg): wiggly holes cut by cast shadows / furniture, small
false islands, and collinear runs -- small irregular rings that cost tens of vertices for little area. Variants on the
product v3 mask (same polygons otherwise):
  base       product v3 (eps 6, axis snap, +1 px)
  ringK      a ring (outer or hole) under 5% of the question's mask area is re-simplified at eps = K x sqrt(its area)
  dropF      holes and islands under F of the mask area are dropped (holes filled, islands removed)
  eps8       eps 8 everywhere, for reference
Prints IoU, vertices / label (median, share <= 1.2x, total / total).   Usage: python3 scripts/pp_probe/smallrings.py SPLIT [MAPS]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from vertcount import label_vertices
from polygonize import LOW_VERTEX, PRODUCT_V2, mask_to_polygons, offset_polys, rasterize, snap_ring, clean_ring


def area(r):
    return abs(cv2.contourArea(np.asarray(r, np.float32)))


def resimplify(polys, total, k, max_frac=0.05):
    out = []
    for p in polys:
        rings = [p["outer"]] + p["holes"]
        new = []
        for r in rings:
            a = area(r)
            if a < max_frac * total and len(r) > 4:
                q = cv2.approxPolyDP(np.asarray(r, np.int32).reshape(-1, 1, 2), max(PRODUCT_V2["eps"], k * np.sqrt(a)), True)[:, 0, :]
                q = np.round(clean_ring(snap_ring(q, 6.0))).astype(int)
                r = q.tolist() if len(q) >= 3 else r
            new.append(r)
        out.append({"outer": new[0], "holes": new[1:]})
    return out


def drop_small(polys, total, f):
    return [{"outer": p["outer"], "holes": [h for h in p["holes"] if area(h) >= f * total]}
            for p in polys if area(p["outer"]) >= f * total]


VARIANTS = ("base", "ring0.1", "ring0.2", "ring0.3", "drop0.01", "drop0.02", "ring0.2+drop0.01", "eps8", "eps7",
            "eps7+ring0.1", "eps8+ring0.1")

if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/gem2s_x")
    common.MAPS = maps
    rows = []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            mp = {z: (p4 if z == 4096 else np.load(maps / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
                  for z in product.SCALES}
            m = product.product_mask(product.blend(mp, (h, w)), img, box)
            total = max(1, int(m.sum()))
            base = offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], PRODUCT_V2["eps"]), PRODUCT_V2["offset"])
            V = {"base": base,
                 "eps8": offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], 8.0), PRODUCT_V2["offset"])}
            for k in (0.1, 0.2, 0.3):
                V[f"ring{k}"] = resimplify(base, total, k)
            for f in (0.01, 0.02):
                V[f"drop{f}"] = drop_small(base, total, f)
            V["ring0.2+drop0.01"] = resimplify(drop_small(base, total, 0.01), total, 0.2)
            V["eps7"] = offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], 7.0), PRODUCT_V2["offset"])
            V["eps7+ring0.1"] = resimplify(V["eps7"], total, 0.1)
            V["eps8+ring0.1"] = resimplify(V["eps8"], total, 0.1)
            r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat)}
            for k, polys in V.items():
                r[k] = (common.masked_iou(rasterize(polys, (h, w)), tgt, (idx, ref)), product.nverts(polys))
            rows.append(r)
    json.dump(rows, open(f"scripts/pp_probe/rows_smallrings_{split}.json", "w"))
    lab = np.array([r["label_v"] for r in rows], float)
    for k in VARIANTS:
        a = np.array([r[k] for r in rows], float); ratio = a[:, 1] / lab
        print(f"{split} {k:18s} IoU {a[:, 0].mean():.4f}  vertices/label median {np.median(ratio):.2f}x  "
              f"<= 1.2x {100 * (ratio <= 1.2).mean():.0f}%  total {a[:, 1].sum() / lab.sum():.2f}x")
