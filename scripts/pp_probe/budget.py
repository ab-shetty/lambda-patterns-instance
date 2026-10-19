"""Oracle error budget of the default mask (lin_fill) per question: how much IoU each kind of fix could buy at most.
  band{k}      pixels within k native px of the mask's edge set to the label
  dropfp       remove predicted components overlapping the label < 10%
  addmiss      add label components the prediction misses (< 10% covered)
  comp         both
  cell         every ink-bounded cell (ink v2) set by label majority (ink follows nearest cell) - cell-voting ceiling"""
import sys, json
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load, ink_map_v2
from methods import fill_holes, iou
T = 0.35

def comps(m):
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8); return n, lab

def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]
    ink = ink_map_v2(img)
    nc, cells = cv2.connectedComponents((~ink).astype(np.uint8), connectivity=4)
    _, lbl = cv2.distanceTransformWithLabels(ink.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(~ink); im = np.zeros(len(ys) + 1, np.int64); im[1:] = ys * w0 + xs; near = im[lbl]
    rows = []
    for ref, cat, G, p in qs:
        s = p.shape[1] / w0
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        M = fill_holes(P > T, 4096 / s / s); r = {"i": idx, "q": ref, "ref": iou(M, G)}
        din = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 5); dout = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 5)
        for k in (2, 4, 8, 16):
            b = (din <= k) & (dout <= k); r[f"band{k}"] = iou(np.where(b, G, M), G)
        n, lab = comps(M); ov = np.bincount(lab.ravel(), weights=G.ravel(), minlength=n) / np.maximum(np.bincount(lab.ravel(), minlength=n), 1)
        fp = (ov < 0.1); fp[0] = False; Md = M & ~fp[lab]; r["dropfp"] = iou(Md, G)
        n, lab = comps(G); cv_ = np.bincount(lab.ravel(), weights=M.ravel(), minlength=n) / np.maximum(np.bincount(lab.ravel(), minlength=n), 1)
        ms = cv_ < 0.1; ms[0] = False; r["addmiss"] = iou(M | ms[lab], G); r["comp"] = iou(Md | ms[lab], G)
        maj = np.bincount(cells.ravel(), weights=G.ravel(), minlength=nc) / np.maximum(np.bincount(cells.ravel(), minlength=nc), 1) > 0.5
        cm = maj[cells]; cm = np.where(ink, cm.ravel()[near].reshape(h0, w0), cm); r["cell"] = iou(cm, G)
        rows.append(r)
    return rows

if __name__ == "__main__":
    split = sys.argv[1]
    with Pool(1) as pool: rows = [r for rs in pool.imap(sheet, jobs(split)) for r in rs]
    json.dump(rows, open(f"data/diag/rows_budget_{split}.json", "w"))
    by = {}
    for r in rows: by.setdefault(r["i"], []).append(r)
    for k in [k for k in rows[0] if k not in ("i", "q")]:
        print(f"{k:8s} q-mean {np.mean([r[k] for r in rows]):.4f} per-sheet {np.mean([np.mean([r[k] for r in v]) for v in by.values()]):.4f}")
