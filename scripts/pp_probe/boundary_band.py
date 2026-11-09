"""How much IoU loss is at the boundary (vertex placement) vs whole wrong / missing regions.
Per question, production polygons (poly). IoU recomputed ignoring a band of k px (native) around the TRUE boundary:
the gain is the loss a perfect boundary/vertex fit could recover at most. Also vertex counts: prediction vs label."""
import sys, json
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from pathlib import Path
import common
from polygonize import probs_to_polygons, rasterize
common.WITH_BOX[0] = True
split = sys.argv[1]
common.MAPS = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/xblend-kq30-kms30")
KS = tuple(int(k) for k in sys.argv[3].split(",")) if len(sys.argv) > 3 else (2, 4, 8, 16)
rows = []
for job in common.jobs(split):
    idx, img, _, qs = common.load(job)
    h, w = img.shape[:2]
    for ref, cat, tgt, p4, box in qs:
        polys = probs_to_polygons(p4, (h, w))
        m = rasterize(polys, (h, w)).astype(bool)
        t8 = tgt.astype(np.uint8)
        edge = cv2.morphologyEx(t8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)
        r = {"sheet": idx, "q": ref, "iou": float((m & tgt).sum() / max(1, (m | tgt).sum()))}
        for k in KS:
            keep = dist > k
            r[f"band{k}"] = float((m & tgt & keep).sum() / max(1, ((m | tgt) & keep).sum()))
        cs, _ = cv2.findContours(t8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        r["v_pred"] = int(sum(len(p["outer"]) + sum(len(x) for x in p["holes"]) for p in polys))
        r["v_true"] = int(sum(len(cv2.approxPolyDP(c, 1.0, True)) for c in cs))
        r["px_fp"] = int((m & ~tgt).sum()); r["px_fn"] = int((~m & tgt).sum())
        r["fp_band4"] = int((m & ~tgt & (dist <= 4)).sum()); r["fn_band4"] = int((~m & tgt & (dist <= 4)).sum())
        rows.append(r)
json.dump(rows, open(f"scripts/pp_probe/rows_band_{split}.json", "w"))
mean = lambda k: np.mean([r[k] for r in rows])
err = sum(r["px_fp"] + r["px_fn"] for r in rows); b4 = sum(r["fp_band4"] + r["fn_band4"] for r in rows)
print(split, len(rows), "q | iou %.4f" % mean("iou"), " ".join("band%d %.4f" % (k, mean(f"band{k}")) for k in KS),
      "| error px within 4px of true edge %.1f%%" % (100 * b4 / max(1, err)),
      "| vertices pred/true median %.2f" % np.median([r["v_pred"] / max(1, r["v_true"]) for r in rows]))
