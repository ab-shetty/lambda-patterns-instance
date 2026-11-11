"""Where the low-vertex output loses IoU, and whether the "prediction falls short" edge error is a cheap fix (2026-11-10).
On the product stack (ms map 0.7 x 4096 + 0.3 x 2048, colour cut), per question:
  thr   threshold on the ms map (0.35 production; lower = wider: the prediction falls short of the label at edges)
  fill  polygonize's low_vertex gap fill (irregular holes / inlets, P >= .05) before the colour cut
  eps   Douglas-Peucker tolerance, native px (2 production, 12 low_vertex); axis snap on
  minf  drop pieces under this share of the answer (0.002 production, 0.005 low_vertex)
  off   grow the rasterised polygons by this many native px (disc dilation) -- an outward polygon offset
Usage: python3 scripts/pp_probe/lvdecomp.py SPLIT MAPS -> rows_lvdecomp_<tag>_<split>.json + a table."""
import sys, json
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common
from stack import colour_cut, nv, iou, CUR
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, rasterize
from curve import label_vertices
common.WITH_BOX[0] = True

CFG = {  # name: (thr, fill, eps, minf, off)
    "prod ms+cc":            (0.35, False, 2, 0.002, 0),
    "ms+cc_lv":              (0.35, True, 12, 0.005, 0),
    "lv no fill":            (0.35, False, 12, 0.005, 0),
    "lv minf.002":           (0.35, True, 12, 0.002, 0),
    "lv eps8":               (0.35, True, 8, 0.005, 0),
    "lv eps6":               (0.35, True, 6, 0.005, 0),
    "lv eps4":               (0.35, True, 4, 0.005, 0),
    "lv off1":               (0.35, True, 12, 0.005, 1),
    "lv off2":               (0.35, True, 12, 0.005, 2),
    "lv off3":               (0.35, True, 12, 0.005, 3),
    "lv thr.25":             (0.25, True, 12, 0.005, 0),
    "lv thr.15":             (0.15, True, 12, 0.005, 0),
    "lv eps8 off2":          (0.35, True, 8, 0.005, 2),
    "lv eps6 off2":          (0.35, True, 6, 0.005, 2),
    "lv eps6 off1":          (0.35, True, 6, 0.005, 1),
    "lv eps5 off1":          (0.35, True, 5, 0.005, 1),
    "lv eps8 off1":          (0.35, True, 8, 0.005, 1),
    "prod off1":             (0.35, False, 2, 0.002, 1),
    "prod off2":             (0.35, False, 2, 0.002, 2),
    "prod thr.25":           (0.25, False, 2, 0.002, 0),
}


def grow(m, k):
    if k <= 0:
        return m
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))).astype(bool)


def sheet(job, maps, split):
    idx, img, anyann, qs = common.load(job)
    h, w = img.shape[:2]
    rows = []
    for ref, cat, tgt, p4, box in qs:
        CUR[0] = (idx, ref)
        f2 = Path(maps) / f"{common.map_split(split)}_2048" / "probs" / f"{idx:03d}_{ref:03d}.npz"
        if not f2.exists():
            continue
        s = p4.shape[1] / w
        P4 = cv2.resize(p4, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        P = 0.7 * P4 + 0.3 * cv2.resize(np.load(f2)["prob"], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat)}
        cut = {}
        for name, (thr, fill, eps, minf, off) in CFG.items():
            if (thr, fill) not in cut:
                m = _fill_holes(P > thr, 4096 / s / s)
                if fill:
                    m = fill_gaps(m, P)
                cut[(thr, fill)] = colour_cut(img, m, box)
            polys = mask_to_polygons(cut[(thr, fill)], minf, eps, 6.0)
            r[name] = iou(grow(rasterize(polys, (h, w)), off), tgt)
            r["v_" + name] = nv(polys)
        rows.append(r)
    return rows


if __name__ == "__main__":
    split, maps = sys.argv[1], sys.argv[2]
    common.MAPS = Path(maps)
    R = [r for job in common.jobs(split) for r in sheet(job, maps, split)]
    tag = Path(maps).name
    json.dump(R, open(f"scripts/pp_probe/rows_lvdecomp_{tag}_{split}.json", "w"))
    L = [r for r in R if r["label_v"] > 0]
    print(f"{split} {tag} {len(R)} q   IoU / vertices-label median (p90), change vs ms+cc_lv")
    for k in CFG:
        v = np.array([r["v_" + k] / r["label_v"] for r in L])
        d = np.array([r[k] - r["ms+cc_lv"] for r in R])
        print(f"  {k:16s} {np.mean([r[k] for r in R]):.4f}  {np.median(v):.2f}x ({np.percentile(v, 90):.1f}x)"
              f"  {d.mean():+.4f} ({(d > 0.001).sum()} up / {(d < -0.001).sum()} down)")
