"""Vertices: our polygons (polygonize.py defaults) vs the vertices the labeller actually placed (the annotation polygons of the
target family), per question. Also where ours go: on pieces that overlap the target vs spill pieces, and per-vertex edge length.
Usage: python3 scripts/pp_probe/vertcount.py SPLIT MAPS_DIR"""
import sys, json
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from pathlib import Path
import common
from polygonize import probs_to_polygons, rasterize
common.WITH_BOX[0] = True


def label_vertices(rec, cat):
    anns = rec["annotations"]; anns = json.loads(anns) if isinstance(anns, str) else anns
    n = 0
    for a in anns:
        if a.get("category_name", a.get("category")) != cat:
            continue
        seg = a["segmentation"]
        seg = json.loads(seg) if isinstance(seg, str) else seg
        rings = seg if isinstance(seg, list) else seg.get("polygons", [])
        for r in rings:
            n += len(r) // 2 if r and not isinstance(r[0], (list, tuple)) else len(r)
    return n


if __name__ == "__main__":
    split, maps = sys.argv[1], sys.argv[2]
    common.MAPS = Path(maps)
    rows = []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        rec = job[1]; h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            polys = probs_to_polygons(p4, (h, w))
            on = spill = 0
            for p in polys:
                k = len(p["outer"]) + sum(len(x) for x in p["holes"])
                m = rasterize([p], (h, w)).astype(bool)
                if (m & tgt).sum() > 0.5 * m.sum(): on += k
                else: spill += k
            rows.append({"sheet": idx, "q": ref, "label": label_vertices(rec, cat), "ours": on + spill, "spill": spill})
    lab = np.array([r["label"] for r in rows]); ours = np.array([r["ours"] for r in rows]); sp = np.array([r["spill"] for r in rows])
    ok = lab > 0
    r = ours[ok] / lab[ok]
    print(split, "%d q (%d with label polygons) | label vertices median %d, ours median %d | ours/label median %.2f, p25 %.2f, p75 %.2f | "
          "share of our vertices on spill pieces %.0f%%" % (len(rows), ok.sum(), np.median(lab[ok]), np.median(ours), np.median(r),
                                                             np.percentile(r, 25), np.percentile(r, 75), 100 * sp.sum() / ours.sum()))
    json.dump(rows, open(f"scripts/pp_probe/rows_vert_{split}.json", "w"))
