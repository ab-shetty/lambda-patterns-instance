"""Cell consistency at the mask's edge (working backwards from HF14 12 q00, val 13 q08, Gemini30 16 q05, HF14 18 q03):
errors there are whole drawn cells -- a blank band beyond the last hatch line, a trim strip, a window sill (wrongly in) or the
last board row before the outline (wrongly out). A person decides those by comparing the cell with the region's own cells.
Cells = connected components of non-ink (ink v2) pixels. Per question, from the default mask M (lin_fill):
  interior cells: cells inside M that do not touch the band of B native px around M's edge -> reference colour (median Lab,
                  area-weighted) and reference thickness (median of 2 x the cell's max inner distance = row height)
  edge cells:     cells meeting that band, smaller than 8x the median interior cell (big blank areas keep the pixel decision)
                  in  <=> |Lab - ref| < tc  and  thickness within [ref / rho, ref * rho]
                  out otherwise
  ink pixels in the band follow their nearest non-ink pixel. Skipped (default kept) when < 6 interior cells (no texture).
Variants cs_b{B}_r{rho}_c{tc}; rows vs ref (= lin_fill+poly)."""
import sys, json, itertools
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
import common
from common import jobs, load, ink_map_v2
common.WITH_BOX[0] = True
from methods import fill_holes, polygons, iou
T = 0.35
GRID = []   # the edge-cell similarity rule (cs_*) lost on its own motivating cases; vote_* only
VOTE = [(t, c) for t in (0.3, 0.4, 0.5) for c in (0.002, 0.01)]


def sheet_cells(img, T=None, min_len=12):
    """T: black-hat threshold; None = ink v2's per-sheet rule."""
    if T is None: ink = ink_map_v2(img)
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        ink = bh > T
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool); keep[1:] = np.maximum(st[1:, 2], st[1:, 3]) >= min_len; ink = keep[lab]
    n, cells = cv2.connectedComponents((~ink).astype(np.uint8), connectivity=4)
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    area = np.bincount(cells.ravel(), minlength=n).astype(float)
    mean = np.stack([np.bincount(cells.ravel(), weights=lab[..., c].ravel(), minlength=n) for c in range(3)], 1) / np.maximum(area, 1)[:, None]
    dt = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    thick = np.zeros(n); np.maximum.at(thick, cells.ravel(), dt.ravel()); thick *= 2
    h0, w0 = ink.shape
    _, lbl = cv2.distanceTransformWithLabels(ink.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(~ink); im = np.zeros(len(ys) + 1, np.int64); im[1:] = ys * w0 + xs
    return dict(ink=ink, cells=cells, n=n, area=area, mean=mean, thick=thick, near=im[lbl])


def refine(M, C, B, rho, tc):
    din = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 5); dout = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 5)
    band = (din <= B) & (dout <= B)
    cells, n, ink = C["cells"], C["n"], C["ink"]
    inM = np.bincount(cells.ravel(), weights=M.ravel(), minlength=n) / np.maximum(C["area"], 1)
    touch = np.bincount(cells.ravel(), weights=band.ravel(), minlength=n) > 0
    touch[0] = False   # label 0 = ink
    interior = (inM > 0.98) & ~touch; interior[0] = False
    if interior.sum() < 6: return None
    a = C["area"][interior]; medA = np.median(a)
    w = a / a.sum()
    refc = np.array([np.average(C["mean"][interior, c], weights=w) for c in range(3)])
    reft = np.median(C["thick"][interior])
    edge = touch & (C["area"] <= 8 * medA)
    dc = np.linalg.norm(C["mean"] - refc, axis=1); tr = C["thick"] / max(reft, 1e-6)
    good = (dc < tc) & (tr > 1 / rho) & (tr < rho)
    out = M.copy()
    sel = edge[cells] & ~ink
    out[sel] = good[cells][sel]
    inkb = ink & band
    out[inkb] = out.ravel()[C["near"][inkb]]
    return out


def ref_threshold(g, box):
    """Half the stroke contrast inside the reference box (its 95th black-hat percentile), clipped to 10..60: the reference's
    own hatch lines count as ink even when they are faint grey."""
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    x, y, w, h = [int(round(v)) for v in box]
    v = bh[max(y, 0):y + h, max(x, 0):x + w]
    return float(np.clip(0.5 * np.percentile(v, 95), 10, 60)) if v.size else 60.0


def vote(M, P, C, t, cap):
    """Cell voting with the reference-adaptive ink: every cell smaller than cap px is in iff its mean probability > t; bigger
    cells keep the pixel decision; ink pixels follow their nearest cell pixel."""
    cells, n, ink = C["cells"], C["n"], C["ink"]
    mp = np.bincount(cells.ravel(), weights=P.ravel(), minlength=n) / np.maximum(C["area"], 1)
    small = C["area"] <= cap; small[0] = False
    out = np.where(small[cells], (mp > t)[cells], M)
    out[ink] = out.ravel()[C["near"][ink]]
    return out


def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); rows = []
    for ref, cat, tgt, p, box in qs:
        s = p.shape[1] / w0; ha = 4096 / s / s
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lf = fill_holes(P > T, ha)
        r = {"i": idx, "q": ref, "ref": iou(polygons(lf)[0], tgt)}
        ys, xs = np.nonzero(lf)
        if not len(ys):
            for B, rho, tc in GRID: r[f"cs_b{B}_r{rho}_c{tc:g}"] = r["ref"]
            for t, capf in VOTE: r[f"vote_t{t}_c{capf}"] = r["ref"]
            rows.append(r); continue
        y1, y2, x1, x2 = max(ys.min() - 40, 0), min(ys.max() + 41, h0), max(xs.min() - 40, 0), min(xs.max() + 41, w0)
        C = sheet_cells(img[y1:y2, x1:x2], ref_threshold(g, box)); lfc = lf[y1:y2, x1:x2]
        Pc = P[y1:y2, x1:x2]
        for t, capf in VOTE:
                m = lf.copy(); m[y1:y2, x1:x2] = vote(lfc, Pc, C, t, capf * h0 * w0)
                r[f"vote_t{t}_c{capf}"] = iou(polygons(fill_holes(m, ha))[0], tgt)
        for B, rho, tc in GRID:
            o = refine(lfc, C, B, rho, tc)
            if o is None: r[f"cs_b{B}_r{rho}_c{tc:g}"] = r["ref"]; continue
            m = lf.copy(); m[y1:y2, x1:x2] = o
            r[f"cs_b{B}_r{rho}_c{tc:g}"] = iou(polygons(fill_holes(m, ha))[0], tgt)
        rows.append(r)
    print(idx, flush=True, file=sys.stderr)
    return rows


if __name__ == "__main__":
    for split in sys.argv[1:]:
        with Pool(2) as pool: rows = [r for rs in pool.imap(sheet, jobs(split)) for r in rs]
        json.dump(rows, open(f"data/diag/rows_cellsim_{split}.json", "w"))
        b = np.array([r["ref"] for r in rows]); by = {}
        for r in rows: by.setdefault(r["i"], []).append(r)
        print(f"== {split}: n={len(rows)} ref {b.mean():.4f}")
        for k in [k for k in rows[0] if k.startswith(("cs_", "vote_"))]:
            d = np.array([r[k] for r in rows]) - b; sd = np.array([np.mean([r[k] - r["ref"] for r in v]) for v in by.values()])
            print(f"  {k:20s} {d.mean():+.4f} (up {(d > .005).sum()} / down {(d < -.005).sum()}; worst {d.min():+.3f}; sheets down >0.02 {(sd < -.02).sum()})", flush=True)
