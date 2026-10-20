"""Two follow-ups to cellsim.py (2026-10-20), tuned on Gemini30 only.
  vg_c{tau}            reference-adaptive cell voting (t 0.4, cells <= 1% of the sheet: the Gemini30 pick) with a colour guard:
                       a cell may be ADDED only if its mean Lab is within tau of the mask's interior colour (area-weighted mean
                       over cells > 98% inside the mask). Removals unchanged. Only ink within 2 px of a cell follows it;
                       thicker ink keeps the pixel decision. (Designed after seeing val 17, where thin red door
                       trims beside a green plinth band were added: val is not a clean test for this one.)
  bite_r{r}_p{rho}_c{tau}  clutter fill: pieces a closing (ellipse, radius r native px) would add are added if they are mostly
                       enclosed by the mask (share >= rho of the piece's outer ring lies in the mask: a bite or hole, not the
                       gap between two separate regions) and their non-ink colour is within tau of the interior colour.
  vg_c{a}+bite_...     the two in sequence.
Rows vs ref (= lin_fill+poly)."""
import sys, json, itertools
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
import common
from common import jobs, load
common.WITH_BOX[0] = True
from methods import fill_holes, polygons, iou
from cellsim import sheet_cells, ref_threshold
T = 0.35
VG = (6.0, 12.0)
BITE = list(itertools.product((8, 16), (0.6, 0.8), (8.0, 15.0)))


def interior_colour(M, C):
    inM = np.bincount(C["cells"].ravel(), weights=M.ravel(), minlength=C["n"]) / np.maximum(C["area"], 1)
    sel = inM > 0.98; sel[0] = False
    if not sel.any(): return None
    return np.average(C["mean"][sel], axis=0, weights=C["area"][sel])


def vote_guard(M, P, C, refc, tau, t=0.4, cap=None):
    cells, n, ink = C["cells"], C["n"], C["ink"]
    mp = np.bincount(cells.ravel(), weights=P.ravel(), minlength=n) / np.maximum(C["area"], 1)
    inM = np.bincount(cells.ravel(), weights=M.ravel(), minlength=n) / np.maximum(C["area"], 1)
    small = C["area"] <= cap; small[0] = False
    dec = mp > t
    if refc is not None:
        dc = np.linalg.norm(C["mean"] - refc, axis=1)
        dec = np.where((inM < 0.5) & dec & (dc >= tau), False, dec)          # additions need the interior colour
    out = np.where(small[cells], dec[cells], M)
    # only thin strokes follow their nearest cell; ink further than 2 px from any cell (thick dark bands, coloured trims
    # read as ink) keeps the pixel decision
    thin = ink & (cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3) <= 2)
    out[thin] = out.ravel()[C["near"][thin]]
    out[ink & ~thin] = M[ink & ~thin]
    # leak guard: additions only within 8 px of the mask (mortar networks of brick / stone textures form "cells" that run
    # far outside the region; val 17)
    near = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 3) <= 8
    out = np.where(~M & ~near, False, out)
    return out


def bite_fill(M, lab, ink, refc, r, rho, tau):
    if refc is None: return M
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    cand = cv2.morphologyEx(M.astype(np.uint8), cv2.MORPH_CLOSE, K).astype(bool) & ~M
    n, lab_c, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    if n <= 1: return M
    ring = cv2.dilate(cand.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~cand
    _, ring_lab = cv2.distanceTransformWithLabels((~cand).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)
    # ring pixel -> nearest candidate piece: use the piece label of the nearest candidate pixel
    ys, xs = np.nonzero(cand); comp_of_zero = np.zeros(ring_lab.max() + 1, np.int64)
    comp_of_zero[ring_lab[ys, xs]] = lab_c[ys, xs]
    rp = comp_of_zero[ring_lab[ring]]
    tot = np.bincount(rp, minlength=n).astype(float); inm = np.bincount(rp, weights=M[ring].astype(float), minlength=n)
    contact = inm / np.maximum(tot, 1)
    ni = ~ink
    cnt = np.bincount(lab_c[ni], minlength=n).astype(float)
    col = np.stack([np.bincount(lab_c[ni], weights=lab[..., c][ni], minlength=n) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    dc = np.linalg.norm(col - refc, axis=1)
    ok = (contact >= rho) & ((dc < tau) | (cnt == 0)); ok[0] = False
    return M | ok[lab_c]


def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); rows = []
    keys = [f"vg_c{a:g}" for a in VG] + [f"bite_r{r}_p{p}_c{c:g}" for r, p, c in BITE] + \
           [f"vg_c{a:g}+bite_r{r}_p{p}_c{c:g}" for a in VG for r, p, c in BITE]
    for ref, cat, tgt, p, box in qs:
        s = p.shape[1] / w0; ha = 4096 / s / s
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lf = fill_holes(P > T, ha)
        r = {"i": idx, "q": ref, "ref": iou(polygons(lf)[0], tgt)}
        ys, xs = np.nonzero(lf)
        if not len(ys):
            r.update({k: r["ref"] for k in keys}); rows.append(r); continue
        y1, y2, x1, x2 = max(ys.min() - 40, 0), min(ys.max() + 41, h0), max(xs.min() - 40, 0), min(xs.max() + 41, w0)
        crop = img[y1:y2, x1:x2]
        C = sheet_cells(crop, ref_threshold(g, box)); lfc = lf[y1:y2, x1:x2]; Pc = P[y1:y2, x1:x2]
        lab = cv2.cvtColor(crop.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        refc = interior_colour(lfc, C)
        def score(mc):
            m = lf.copy(); m[y1:y2, x1:x2] = mc
            return iou(polygons(fill_holes(m, ha))[0], tgt)
        for a in VG:
            v = vote_guard(lfc, Pc, C, refc, a, cap=0.01 * h0 * w0); r[f"vg_c{a:g}"] = score(v)
            for rr, pp, cc in BITE:
                r[f"vg_c{a:g}+bite_r{rr}_p{pp}_c{cc:g}"] = score(bite_fill(v, lab, C["ink"], refc, rr, pp, cc))
        for rr, pp, cc in BITE:
            r[f"bite_r{rr}_p{pp}_c{cc:g}"] = score(bite_fill(lfc, lab, C["ink"], refc, rr, pp, cc))
        rows.append(r)
    print(idx, flush=True, file=sys.stderr)
    return rows


if __name__ == "__main__":
    for split in sys.argv[1:]:
        with Pool(2) as pool: rows = [r for rs in pool.imap(sheet, jobs(split)) for r in rs]
        json.dump(rows, open(f"data/diag/rows_guard_{split}.json", "w"))
        print(f"== {split}: n={len(rows)} ref {np.mean([r['ref'] for r in rows]):.4f}", flush=True)
