"""Fewer vertices without losing IoU (2026-11-07). On top of polygonize.py's defaults (eps 2 + axis snap):
  merge    drop a vertex whose neighbours' line passes within max(2, frac x that line's length) px of it, capped at cap px,
           smallest offset first (straightens diagonal hips / ridges, any angle)
  notch    close the mask with a disc of radius r; keep an added piece only if it holds drawing ink (text, symbols: the
           labeller draws straight through them) and is under amax of the sheet
  hole     fill holes under hmax of the sheet that hold ink (slope text inside a region); skylights are bigger
Vertices are compared with the ones the labeller placed (vertcount.py). Usage: python3 scripts/pp_probe/vertfix.py SPLIT MAPS [CFG1,CFG2,...]"""
import sys, json
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from pathlib import Path
import common
from polygonize import probs_to_polygons, rasterize, T, _fill_holes
from vertcount import label_vertices
import textcache   # text_mask.py masks, cached by textcache.py
common.WITH_BOX[0] = True


def merge_ring(p, frac=0.01, cap=4.0, lo=2.0):
    p = np.asarray(p, np.float64)
    while len(p) > 3:
        a, c = np.roll(p, 1, 0), np.roll(p, -1, 0)
        ac = c - a
        L = np.hypot(ac[:, 0], ac[:, 1])
        off = np.abs((p - a)[:, 0] * ac[:, 1] - (p - a)[:, 1] * ac[:, 0]) / np.maximum(L, 1e-6)
        tol = np.minimum(np.maximum(lo, frac * L), cap)
        i = int(np.argmin(off - tol))
        if off[i] >= tol[i]:
            break
        p = np.delete(p, i, 0)
    return np.round(p).astype(int)


def merge(polys, **kw):
    out = []
    for q in polys:
        o = merge_ring(q["outer"], **kw)
        if len(o) >= 3:
            out.append({"outer": o.tolist(), "holes": [h for h in (merge_ring(x, **kw).tolist() for x in q["holes"]) if len(h) >= 3]})
    return out


def ink_fill(m, ink, r=0, amax=0.0, hmax=0.0):
    n_px = m.size
    if r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        add = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & ~m
        n, cc, st, _ = cv2.connectedComponentsWithStats(add.astype(np.uint8), connectivity=8)
        if n > 1:
            inkn = np.bincount(cc[ink], minlength=n)
            ok = (inkn > 0.02 * st[:, 4]) & (st[:, 4] < amax * n_px); ok[0] = False
            m = m | ok[cc]
    if hmax:
        n, cc, st, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
        h, w = m.shape
        inkn = np.bincount(cc[ink], minlength=n)
        ok = np.zeros(n, bool)
        for i in range(1, n):
            x, y, ww, hh, a = st[i]
            ok[i] = not (x == 0 or y == 0 or x + ww == w or y + hh == h) and a < hmax * n_px and inkn[i] > 0.02 * a
        m = m | ok[cc]
    return m


def mask_polys(m, hw):
    return probs_to_polygons(np.where(m, 255, 0).astype(np.uint8), hw)


def nv(polys):
    return sum(len(p["outer"]) + sum(len(h) for h in p["holes"]) for p in polys)


CFG = {
    "base": {},
    "merge1": {"merge": dict(frac=0.01, cap=4.0)},
    "merge2": {"merge": dict(frac=0.02, cap=6.0)},
    "notch": {"fill": dict(r=10, amax=0.002)},
    "hole": {"fill": dict(hmax=0.002)},
    "notch+hole": {"fill": dict(r=10, amax=0.002, hmax=0.002)},
    "all1": {"fill": dict(r=10, amax=0.002, hmax=0.002), "merge": dict(frac=0.01, cap=4.0)},
    "all2": {"fill": dict(r=10, amax=0.002, hmax=0.002), "merge": dict(frac=0.02, cap=6.0)},
    # text-only ink (scripts/text_mask.py, Tesseract words): drawing ink also fires on pattern lines (Gemini30 -0.011)
    "notchT": {"fill": dict(r=10, amax=0.002), "src": "text"},
    "notchT+holeT": {"fill": dict(r=10, amax=0.002, hmax=0.002), "src": "text"},
    "allT1": {"fill": dict(r=10, amax=0.002, hmax=0.002), "src": "text", "merge": dict(frac=0.01, cap=4.0)},
}
ONLY = None

if __name__ == "__main__":
    split, maps = sys.argv[1], sys.argv[2]
    ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    common.MAPS = Path(maps)
    rows = []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        ink = common.ink_map(img)
        text = textcache.load(split, idx, (h, w)) & ink if any(c.get("src") == "text" for n, c in CFG.items() if ONLY is None or n in ONLY) else None
        for ref, cat, tgt, p4, box in qs:
            P = cv2.resize(p4.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_LINEAR)
            m0 = _fill_holes(P > T, 4096 / (p4.shape[1] / w) ** 2)
            r = {"sheet": idx, "q": ref, "label": label_vertices(job[1], cat)}
            for name, c in CFG.items():
                if ONLY and name not in ONLY:
                    continue
                m = ink_fill(m0, text if c.get("src") == "text" else ink, **c["fill"]) if "fill" in c else m0
                polys = mask_polys(m, (h, w)) if "fill" in c else probs_to_polygons(p4, (h, w))
                if "merge" in c:
                    polys = merge(polys, **c["merge"])
                M = rasterize(polys, (h, w)).astype(bool)
                r[name] = (float((M & tgt).sum() / max(1, (M | tgt).sum())), nv(polys))
            rows.append(r)
    json.dump(rows, open(f"scripts/pp_probe/rows_vertfix_{split}{'_' + sys.argv[3].replace(',', '-') if ONLY else ''}.json", "w"))
    rows = [r for r in rows if r["label"] > 0]
    lab = np.array([r["label"] for r in rows])
    for name in (ONLY or CFG):
        a = np.array([r[name] for r in rows], float)
        print(split, "%-11s iou %.4f  vertices/label median %.2f p75 %.2f  total %.2fx" %
              (name, a[:, 0].mean(), np.median(a[:, 1] / lab), np.percentile(a[:, 1] / lab, 75), a[:, 1].sum() / lab.sum()))
