import sys
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from pathlib import Path
import common
from polygonize import probs_to_polygons, rasterize
from snap import simplify, nverts
common.WITH_BOX[0] = True
split = sys.argv[1]; common.MAPS = Path("data/floz-diag/gem2s_x")
R = []
for job in common.jobs(split):
    idx, img, _, qs = common.load(job); h, w = img.shape[:2]
    for ref, cat, tgt, p4, box in qs:
        n_t = cv2.connectedComponents(tgt.astype(np.uint8), connectivity=8)[0] - 1
        cs, hier = cv2.findContours(tgt.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes_t = 0 if hier is None else int((hier[0][:, 3] != -1).sum())
        row = {"pt": n_t, "ht": holes_t}
        for mf in (0.002, 0.01, 0.03):
            polys = simplify(probs_to_polygons(p4, (h, w), eps=3.0, min_frac=mf, snap_deg=None), 6.0)
            m = rasterize(polys, (h, w)).astype(bool)
            row[mf] = ((m & tgt).sum() / max(1, (m | tgt).sum()), len(polys), sum(len(p["holes"]) for p in polys), nverts(polys))
        R.append(row)
pt = np.array([r["pt"] for r in R]); ht = np.array([r["ht"] for r in R])
print(split, "label pieces mean %.1f holes %.1f" % (pt.mean(), ht.mean()))
for mf in (0.002, 0.01, 0.03):
    a = np.array([r[mf] for r in R], float)
    print(split, "min_frac %.3f iou %.4f pieces %.1f holes %.1f vertices %.0f" % (mf, a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean()))
