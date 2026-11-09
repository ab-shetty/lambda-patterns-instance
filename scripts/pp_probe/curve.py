"""Vertex / IoU trade-off curve (2026-11-09). How much IoU does a labeller-like polygon cost? Every lever that removes vertices,
alone and stacked, on the saved @4096 maps; per question IoU (rasterised polygons) and our vertices vs the labeller's
(vertcount.label_vertices). Levers, on top of polygonize.py's pipeline (threshold 0.35, fill holes, drop small pieces, trace,
Douglas-Peucker eps, snap near-axis edges within 6 deg):
  eps     Douglas-Peucker tolerance (native px; default 2)
  merge   vertfix.merge_ring: drop vertices within max(2, frac x span) px of their neighbours' line, capped -- any angle
  hole    fill holes up to this area (inference px at 4096; default 4096)
  close   morphological closing, disc radius r (4096-scale px): fills notches around text / symbols, may bridge to neighbours
  minf    drop pieces under this share of the answer (default 0.002): spill pieces
Usage: python3 scripts/pp_probe/curve.py SPLIT MAPS [SHARD NSHARDS] -> scripts/pp_probe/rows_curve_<model>_<split>[_<shard>].json
       python3 scripts/pp_probe/curve.py --summary MODEL   (merges shards, prints the table)"""
import sys, json, glob
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from pathlib import Path

B = dict(eps=2.0, merge=None, hole=4096, close=0, minf=0.002, blur=0)
CFG = {  # cumulative ladder L0..L7, then single levers off the default
    "L0 default":            {},
    "L1 eps3":               dict(eps=3.0),
    "L2 eps4":               dict(eps=4.0),
    "L3 +merge":             dict(eps=4.0, merge=(0.02, 6.0)),
    "L4 +hole16k":           dict(eps=4.0, merge=(0.02, 6.0), hole=16384),
    "L5 +close6":            dict(eps=4.0, merge=(0.02, 6.0), hole=16384, close=6),
    "L6 +minf.01":           dict(eps=4.0, merge=(0.02, 6.0), hole=16384, close=6, minf=0.01),
    "L7 all strong":         dict(eps=6.0, merge=(0.03, 10.0), hole=65536, close=10, minf=0.02),
    "s eps4":                dict(eps=4.0),
    "s eps8":                dict(eps=8.0),
    "s merge":               dict(merge=(0.02, 6.0)),
    "s hole16k":             dict(hole=16384),
    "s hole64k":             dict(hole=65536),
    "s close6":              dict(close=6),
    "s minf.01":             dict(minf=0.01),
    "s minf.03":             dict(minf=0.03),
    # blur: Gaussian sigma (4096-scale px) on the probability map before thresholding -- removes the texture sawtooth itself
    "b blur2":               dict(blur=2),
    "b blur4":               dict(blur=4),
    "b blur8":               dict(blur=8),
    "b blur16":              dict(blur=16),
    "b eps4+blur4":          dict(eps=4.0, blur=4),
    "b eps4+blur8":          dict(eps=4.0, blur=8),
    "b eps8+blur8":          dict(eps=8.0, blur=8),
    "b eps4+blur8+minf.01":  dict(eps=4.0, blur=8, minf=0.01),
    # adapt: per-outline eps, coarsest keeping the ring's own IoU >= tau
    "a tau.99":              dict(adapt=0.99),
    "a tau.98":              dict(adapt=0.98),
    "a tau.97":              dict(adapt=0.97),
    "a tau.95":              dict(adapt=0.95),
    "a tau.98+blur4":        dict(adapt=0.98, blur=4),
    "a tau.97+minf.01":      dict(adapt=0.97, minf=0.01),
    # rhole: fill irregular holes (area / min-area rectangle < r), keep rectangle-like ones (skylights); f = max hole share
    "r rect.8":              dict(rhole=(0.8, 1.0)),
    "r rect.7":              dict(rhole=(0.7, 1.0)),
    "r rect.8 f.05":         dict(rhole=(0.8, 0.05)),
    "r rect.8+tau.97":       dict(rhole=(0.8, 1.0), adapt=0.97),
    "r rect.8+tau.95":       dict(rhole=(0.8, 1.0), adapt=0.95),
    "r rect.8 f.05+tau.97":  dict(rhole=(0.8, 0.05), adapt=0.97),
    "r rect.8+tau.97+minf.01": dict(rhole=(0.8, 1.0), adapt=0.97, minf=0.01),
    # inlet: irregular gaps between each piece and its convex hull (dropouts reaching the edge), on top of the hole rule
    "i inlet.8 f.05":        dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05)),
    "i inlet.8 f.02":        dict(rhole=(0.8, 0.05), inlet=(0.8, 0.02)),
    "i inlet.7 f.05":        dict(rhole=(0.8, 0.05), inlet=(0.7, 0.05)),
    "i inlet.8 f.05+tau.97": dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05), adapt=0.97),
    "i inlet.8 f.02+tau.97": dict(rhole=(0.8, 0.05), inlet=(0.8, 0.02), adapt=0.97),
    "i inlet.8 f.05+tau.97+minf.01": dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05), adapt=0.97, minf=0.01),
    "i inlet.8 f.05+tau.95+minf.01": dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05), adapt=0.95, minf=0.01),
    # pmin: fill a gap only if its mean probability >= pmin (a dropout the model was unsure about, not a true gap)
    "p .05 inlet+tau.95+minf.01": dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05), adapt=0.95, minf=0.01, pmin=0.05),
    "p .10 inlet+tau.95+minf.01": dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05), adapt=0.95, minf=0.01, pmin=0.10),
    "p .15 inlet+tau.95+minf.01": dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05), adapt=0.95, minf=0.01, pmin=0.15),
    "p .20 inlet+tau.95+minf.01": dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05), adapt=0.95, minf=0.01, pmin=0.20),
    "p .10 inlet+tau.97+minf.01": dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05), adapt=0.97, minf=0.01, pmin=0.10),
    "p .10 inlet f.1+tau.95+minf.01": dict(rhole=(0.8, 0.1), inlet=(0.8, 0.1), adapt=0.95, minf=0.01, pmin=0.10),
    "p .10 inlet":           dict(rhole=(0.8, 0.05), inlet=(0.8, 0.05), pmin=0.10),
}
ONLY = None


# label_vertices / merge_ring copied from vertcount.py / vertfix.py: both run their whole script on import
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


def merge(polys, frac, cap):
    out = []
    for q in polys:
        o = merge_ring(q["outer"], frac, cap)
        if len(o) >= 3:
            out.append({"outer": o.tolist(), "holes": [h for h in (merge_ring(x, frac, cap).tolist() for x in q["holes"]) if len(h) >= 3]})
    return out


ADAPT_EPS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)


def adaptive_ring(q, tau, eps0):
    """Coarsest Douglas-Peucker eps (then axis snap) whose filled ring keeps IoU >= tau with the ring's own filled contour:
    per-outline tolerance instead of one eps for the sheet. Rectangles collapse to 4 vertices; textured edges take what they can."""
    from polygonize import snap_ring, clean_ring
    x, y, w, h = cv2.boundingRect(q)
    pad = 2; off = np.array([x - pad, y - pad])
    ref = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(ref, [q - off], -1, 1, -1)
    ra = ref.sum()
    best = np.round(clean_ring(snap_ring(cv2.approxPolyDP(q, eps0, True)[:, 0, :], 6.0))).astype(int)
    for e in ADAPT_EPS:
        if e <= eps0:
            continue
        cand = np.round(clean_ring(snap_ring(cv2.approxPolyDP(q, e, True)[:, 0, :], 6.0))).astype(int)
        if len(cand) < 3:
            break
        m = np.zeros_like(ref); cv2.fillPoly(m, [(cand - off).astype(np.int32)], 1)
        inter = (m & ref).sum(); uni = (m | ref).sum()
        if uni == 0 or inter / uni < tau:
            break
        best = cand
    return best


def fill_irregular_holes(m, rect_min=0.8, max_frac=1.0, P=None, pmin=0.0):
    """Fill a hole (background component not touching the border) unless it is rectangle-like: area / its min-area rotated
    rectangle >= rect_min keeps it (skylights, insets of another material). max_frac: never fill a hole larger than this
    share of the mask's area. Blotchy confidence dropouts inside a correct region are irregular and get filled."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
    h, w = m.shape
    tot = max(1, int(m.sum()))
    fill = np.zeros(n, bool)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if x == 0 or y == 0 or x + ww == w or y + hh == h or a > max_frac * tot:
            continue
        sub = (lab[y:y + hh, x:x + ww] == i).astype(np.uint8)
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        rw, rh = cv2.minAreaRect(max(cs, key=cv2.contourArea))[1]
        fill[i] = a / max(1.0, rw * rh) < rect_min
        if fill[i] and pmin > 0:
            fill[i] = P[y:y + hh, x:x + ww][sub.astype(bool)].mean() >= pmin
    return m | fill[lab]


def fill_irregular_inlets(m, rect_min=0.8, max_frac=0.05, P=None, pmin=0.0):
    """Per piece: the gap between its convex hull and the mask, split into components; fill a component unless it is
    rectangle-like (an L-shaped region's notch, an inset of another material) or larger than max_frac of the piece. Catches
    dropouts that reach the piece's edge (inlets), which a hole rule cannot see."""
    out = m.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if a < 64:
            continue
        piece = (lab[y:y + hh, x:x + ww] == i).astype(np.uint8)
        cs, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = np.zeros_like(piece); cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cs))], 1)
        gap = (hull & (1 - piece)).astype(np.uint8)
        k, gl, gs, _ = cv2.connectedComponentsWithStats(gap, connectivity=4)
        sub = out[y:y + hh, x:x + ww]
        for j in range(1, k):
            ga = gs[j, 4]
            if ga > max_frac * a or ga < 4:
                continue
            gx, gy, gw, gh = gs[j, :4]
            g = (gl[gy:gy + gh, gx:gx + gw] == j).astype(np.uint8)
            gc, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rw, rh = cv2.minAreaRect(max(gc, key=cv2.contourArea))[1]
            if ga / max(1.0, rw * rh) < rect_min:
                sel = gl == j
                if pmin > 0 and P[y:y + hh, x:x + ww][sel].mean() < pmin:
                    continue
                sub[sel] = True
    return out


def polys_from(P, s, c):
    from polygonize import _fill_holes, snap_ring, clean_ring
    if c.get("blur"):
        sg = c["blur"] / s
        P = cv2.GaussianBlur(P, (0, 0), sg)
    m = P > 0.35
    m = _fill_holes(m, c["hole"] / s / s)
    if c["close"]:
        r = max(1, int(round(c["close"] / s)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        m = _fill_holes(m, c["hole"] / s / s)
    if c.get("rhole"):
        m = fill_irregular_holes(m, *c["rhole"], P=P, pmin=c.get("pmin", 0.0))
    if c.get("inlet"):
        m = fill_irregular_inlets(m, *c["inlet"], P=P, pmin=c.get("pmin", 0.0))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n > 1:
        keep = np.zeros(n, bool); keep[1:] = st[1:, 4] >= c["minf"] * st[1:, 4].sum(); m = keep[lab]
    cs, hier = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    if c.get("adapt"):
        ps = [adaptive_ring(q, c["adapt"], c["eps"]) for q in cs]
    else:
        ps = [np.round(clean_ring(snap_ring(cv2.approxPolyDP(q, c["eps"], True)[:, 0, :], 6.0))).astype(int) for q in cs]
    out, index = [], {}
    for j, hh in enumerate(hier[0]):
        if hh[3] == -1 and len(ps[j]) >= 3:
            index[j] = len(out); out.append({"outer": ps[j].tolist(), "holes": []})
    for j, hh in enumerate(hier[0]):
        if hh[3] != -1 and hh[3] in index and len(ps[j]) >= 3:
            out[index[hh[3]]]["holes"].append(ps[j].tolist())
    if c["merge"]:
        out = merge(out, frac=c["merge"][0], cap=c["merge"][1])
    return out


def summary(model, pre=""):
    print(f"model {model} | vertices = ours / labeller's, per question | IoU = rasterised polygons, mean")
    for split in ("gemini30", "val", "hf14"):
        rows = sum((json.load(open(f)) for f in sorted(glob.glob(f"scripts/pp_probe/rows_curve{pre}_{model}_{split}*.json"))), [])
        rows = [r for r in rows if r["label"] > 0]
        if not rows:
            continue
        lab = np.array([r["label"] for r in rows], float)
        base = np.array([r["L0 default"][0] for r in rows])
        print(f"\n{split} ({len(rows)} q)  {'config':<15} {'IoU':>7} {'dIoU':>8} {'v/l med':>8} {'p75':>6} {'p90':>6} {'<=1.5x':>7} {'<=2x':>6}")
        for name in CFG:
            if name not in rows[0]:
                continue
            a = np.array([r[name] for r in rows], float); x = a[:, 1] / lab
            print(f"{'':>{len(split) + 9}}{name:<15} {a[:, 0].mean():7.4f} {a[:, 0].mean() - base.mean():+8.4f} {np.median(x):8.2f} "
                  f"{np.percentile(x, 75):6.2f} {np.percentile(x, 90):6.2f} {100 * np.mean(x <= 1.5):6.0f}% {100 * np.mean(x <= 2):5.0f}%")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""); sys.exit()
    import common
    from polygonize import rasterize
    common.WITH_BOX[0] = True
    import os
    ONLY = [n for n in CFG if os.environ.get("CURVE_ONLY") and any(n.startswith(x) for x in os.environ["CURVE_ONLY"].split(","))] or None
    split, maps = sys.argv[1], sys.argv[2]
    shard, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    common.MAPS = Path(maps)
    rows = []
    for k, job in enumerate(common.jobs(split)):
        if k % nsh != shard:
            continue
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            s = p4.shape[1] / w
            P = cv2.resize(p4.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_LINEAR)
            r = {"sheet": idx, "q": ref, "label": label_vertices(job[1], cat)}
            for name, c in CFG.items():
                if ONLY and name not in ONLY:
                    continue
                polys = polys_from(P, s, {**B, **c})
                M = rasterize(polys, (h, w)).astype(bool)
                r[name] = (float((M & tgt).sum() / max(1, (M | tgt).sum())),
                           int(sum(len(p["outer"]) + sum(len(x) for x in p["holes"]) for p in polys)))
            rows.append(r)
    tag = Path(maps).name
    pre = os.environ.get("CURVE_TAG", "")
    json.dump(rows, open(f"scripts/pp_probe/rows_curve{pre}_{tag}_{split}{'_%d' % shard if nsh > 1 else ''}.json", "w"))
    print("done", split, tag, shard, len(rows))
