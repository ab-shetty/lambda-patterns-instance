#!/usr/bin/env python3
"""Edit-cost proxy for the polygon output (2026-10-20). The user fixes a wrong answer in the frontend by deleting wrong
polygons, tracing missed sections and moving / cutting vertices, so per question, beside the polygons' IoU:
  delete  predicted polygons on the target for < 10% of their area (one delete each)
  cut     the rest that also cover > 10% of their area AND > 0.05% of the sheet outside the target dilated by 4 native px
          (spill attached to a found region: vertex surgery, the expensive fix)
  add     target pieces (8-connected, >= 0.2% of the target) recalled < 25% by the polygons (trace a new polygon)
  verts   vertices over all polygons, outer rings and holes
  actions = delete + cut + add
Polygons from scripts/polygonize.py on the best model's maps (data/floz-diag/xblend-kq30-kms30/<split>_4096/probs);
--eps sets its Douglas-Peucker tolerance (native px; 1 = the default output). IoU stays the gate; this is a diagnostic.
  python3 scripts/editcost.py val hf14 gemini30 --eps 1,2,4 [--maps DIR]"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts")
import common
from polygonize import probs_to_polygons, rasterize

BAND = 4


def raster_one(p, hw):
    m = np.zeros(hw, np.uint8)
    cv2.fillPoly(m, [np.array(p["outer"], np.int32)], 1)
    for hole in p["holes"]:
        cv2.fillPoly(m, [np.array(hole, np.int32)], 0)
    return m.astype(bool)


def edit_cost(polys, tgt):
    hw = tgt.shape
    near = cv2.dilate(tgt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1,) * 2)).astype(bool)
    pred = rasterize(polys, hw)
    inter, union = (pred & tgt).sum(), (pred | tgt).sum()
    delete = cut = 0
    for p in polys:
        m = raster_one(p, hw); a = m.sum()
        if a == 0:
            continue
        if (m & tgt).sum() < 0.1 * a:
            delete += 1
        else:
            out = (m & ~near).sum()
            cut += int(out > 0.1 * a and out > 0.0005 * tgt.size)
    n, lab, st, _ = cv2.connectedComponentsWithStats(tgt.astype(np.uint8), connectivity=8)
    add = 0
    for i in range(1, n):
        if st[i, 4] >= 0.002 * tgt.sum():
            piece = lab == i
            add += int((piece & pred).sum() < 0.25 * st[i, 4])
    verts = sum(len(p["outer"]) + sum(len(h) for h in p["holes"]) for p in polys)
    return {"iou": float(inter / union) if union else 1.0, "delete": delete, "cut": cut, "add": add,
            "actions": delete + cut + add, "verts": verts, "polys": len(polys)}


def sheet(args):
    job, epss = args
    idx, img, anyann, qs = common.load(job)
    h0, w0 = img.shape[:2]; rows = []
    for ref, cat, tgt, p in qs:
        for eps in epss:
            rows.append({"i": idx, "q": ref, "eps": eps, **edit_cost(probs_to_polygons(p, (h0, w0), eps=eps), tgt)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splits", nargs="+")
    ap.add_argument("--eps", default="1")
    ap.add_argument("--maps", default=str(common.MAPS))
    ap.add_argument("--out", default="data/diag")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    common.MAPS = Path(a.maps); epss = [float(e) for e in a.eps.split(",")]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        with Pool(a.workers) as pool:
            rows = [r for rs in pool.map(sheet, [(j, epss) for j in common.jobs(split)]) for r in rs]
        json.dump(rows, open(f"{a.out}/rows_editcost_{split}.json", "w"))
        print(f"{split}: {len({(r['i'], r['q']) for r in rows})} questions")
        print("  eps   IoU     verts med / p90   polys/q  delete/q  cut/q  add/q  actions/q  q with any action")
        for eps in epss:
            R = [r for r in rows if r["eps"] == eps]
            v = np.array([r["verts"] for r in R])
            m = lambda k: np.mean([r[k] for r in R])
            print(f"  {eps:<4}  {m('iou'):.4f}  {np.median(v):6.0f} / {np.percentile(v, 90):6.0f}   {m('polys'):6.2f}"
                  f"   {m('delete'):6.2f}   {m('cut'):5.2f}  {m('add'):5.2f}   {m('actions'):6.2f}"
                  f"     {np.mean([r['actions'] > 0 for r in R]):.2f}")


if __name__ == "__main__":
    main()
