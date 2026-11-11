"""Snap the product mask to the drawing's cells (2026-11-11). Architectural sheets are closed cells bounded by ink lines and the
labels follow them; the remaining real-sheet loss is partial coverage (HF14 14's band at 37%) and the 1-4 px edge band.
Cells = 4-connected components of non-ink (common.ink_map A: darker than the 9 px local max by > T; or ink_map_v2 B), once per
sheet. Per question, on product v2's mask before tracing (5120+4096+3072 equal blend, threshold 0.35, gap fills, colour cut):
  coverage c = share of the cell inside the mask;  c >= hi -> take the whole cell;  c <= lo -> drop the cell's pixels;  else keep.
Cells over `big` of the sheet (paper, page background) or under 30 px are left as they are. Then strokes: ink pixels whose
3x3 neighbourhood is mostly selected join (internal lines between taken cells). Traced as product v2 (eps 6, pieces < 0.5%),
with and without the 1 px offset.
Usage: python3 scripts/pp_probe/cellsnap.py SPLIT MAPS [SHARD N];  python3 scripts/pp_probe/cellsnap.py --summary TAG SPLIT"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2

# name: (ink, T, hi, lo, big[, min cell px, refill holes])
CFG_ROUND1 = {
    "A60 hi.5":        ("A", 60, 0.5, -1, 0.15),
    "A60 hi.5 lo.2":   ("A", 60, 0.5, 0.2, 0.15),
    "A60 hi.3":        ("A", 60, 0.3, -1, 0.15),
    "A60 hi.7":        ("A", 60, 0.7, -1, 0.15),
    "A60 hi.7 lo.3":   ("A", 60, 0.7, 0.3, 0.15),
    "A40 hi.5":        ("A", 40, 0.5, -1, 0.15),
    "A40 hi.5 lo.2":   ("A", 40, 0.5, 0.2, 0.15),
    "B hi.5":          ("B", 0, 0.5, -1, 0.15),
    "B hi.5 lo.2":     ("B", 0, 0.5, 0.2, 0.15),
    "A60 hi.5 big.05": ("A", 60, 0.5, -1, 0.05),
}
# round 2 (2026-11-11): round 1's take+drop pushed hf14fix4 to 1.3-1.45x vertices -- dropping / taking tiny cells (text,
# hatch fragments) notches the outline. Minimum cell size, and holes refilled after the snap.
CFG_ROUND2 = {
    "A40 hi.5 lo.2":            ("A", 40, 0.5, 0.2, 0.15, 30, False),
    "A40 hi.5 lo.2 min500":     ("A", 40, 0.5, 0.2, 0.15, 500, False),
    "A40 hi.5 lo.2 min2k":      ("A", 40, 0.5, 0.2, 0.15, 2000, False),
    "A40 hi.5 lo.2 min500 f":   ("A", 40, 0.5, 0.2, 0.15, 500, True),
    "A40 hi.5 lo.2 min2k f":    ("A", 40, 0.5, 0.2, 0.15, 2000, True),
    "A40 hi.5 min500 f":        ("A", 40, 0.5, -1, 0.15, 500, True),
    "A50 hi.5 lo.2 min500 f":   ("A", 50, 0.5, 0.2, 0.15, 500, True),
    "A40 hi.6 lo.2 min500 f":   ("A", 40, 0.6, 0.2, 0.15, 500, True),
    "A40 hi.5 lo.1 min500 f":   ("A", 40, 0.5, 0.1, 0.15, 500, True),
}
# round 3: round 2's take-only rule fails Gemini (-0.006 Gemini30, -0.003 gemdev); hi .6 + drop holds there (+0.0005 gemdev,
# -0.0024 Gemini30) but hf14fix4 is 1.23x. Stricter take / bigger minimum cell for the vertex budget.
CFG = {
    "A40 hi.6 lo.2 min500 f":   ("A", 40, 0.6, 0.2, 0.15, 500, True),
    "A40 hi.6 lo.2 min1k f":    ("A", 40, 0.6, 0.2, 0.15, 1000, True),
    "A40 hi.6 lo.2 min2k f":    ("A", 40, 0.6, 0.2, 0.15, 2000, True),
    "A40 hi.7 lo.2 min500 f":   ("A", 40, 0.7, 0.2, 0.15, 500, True),
    "A40 hi.7 lo.2 min2k f":    ("A", 40, 0.7, 0.2, 0.15, 2000, True),
    "A40 hi.8 lo.2 min1k f":    ("A", 40, 0.8, 0.2, 0.15, 1000, True),
}


def summary(tag, split):
    R = [r for f in sorted(glob.glob(f"scripts/pp_probe/rows_cellsnap3_{tag}_{split}*.json")) for r in json.load(open(f))]
    L = [r for r in R if r["label_v"] > 0]
    print(f"{split} {tag} {len(R)} q   IoU / vertices-label median; change vs product v2 (off1 = with the 1 px offset)")
    for k in ["v2"] + list(CFG):
        for o in ("", " off1"):
            d = np.array([r[k + o] - r["v2" + o] for r in R])
            print(f"  {k + o:22s} {np.mean([r[k + o] for r in R]):.4f}  {d.mean():+.4f} ({(d > 1e-3).sum()} up / "
                  f"{(d < -1e-3).sum()} down)  {np.median([r['v_' + k] / r['label_v'] for r in L]):.2f}x")


if __name__ == "__main__" and sys.argv[1] == "--summary":
    summary(sys.argv[2], sys.argv[3]); sys.exit()

import common
from stack import colour_cut, iou, CUR, nv
from curve import label_vertices
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, rasterize, offset_polys
common.WITH_BOX[0] = True


def cells_of(ink):
    n, lab = cv2.connectedComponents((~ink).astype(np.uint8), connectivity=4)
    return n, lab


def snap(m, ink, n, lab, hi, lo, big, minpx=30, refill=False, s=1.0):
    tot = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    inm = np.bincount(lab.ravel(), weights=m.ravel().astype(np.float64), minlength=n)
    c = inm / np.maximum(tot, 1)
    ok = (tot >= minpx) & (tot <= big * lab.size)
    ok[0] = False                                     # label 0 = the ink itself
    take, drop = ok & (c >= hi), ok & (c <= lo) & (c > 0)
    out = m.copy()
    out[take[lab]] = True
    out[drop[lab]] = False
    out[ink] = m[ink]
    # strokes between taken cells join: an ink pixel whose 3x3 neighbourhood is mostly selected
    nb = cv2.blur(out.astype(np.float32), (3, 3))
    out |= ink & (nb >= 0.5)
    if refill:   # the production hole fill again (4096 px at inference scale): the snap can open small holes
        out = _fill_holes(out, 4096 / s / s)
    return out


if __name__ == "__main__":
    split, maps = sys.argv[1], Path(sys.argv[2])
    shard, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    common.MAPS = maps
    ms = common.map_split(split)
    rows = []
    for job in list(common.jobs(split))[shard::nsh]:
        idx, img, anyann, qs = common.load(job)
        h, w = img.shape[:2]
        inks = {("A", t): common.ink_map(img, T=t) for t in sorted({c[1] for c in CFG.values() if c[0] == "A"})}
        if any(c[0] == "B" for c in CFG.values()):
            inks[("B", 0)] = common.ink_map_v2(img)
        cells = {k: cells_of(v) for k, v in inks.items()}
        for ref, cat, tgt, p4, box in qs:
            CUR[0] = (idx, ref)
            s = p4.shape[1] / w
            P = cv2.resize(p4, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 / 3
            for z in (5120, 3072):
                P += cv2.resize(np.load(maps / f"{ms}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"], (w, h),
                                interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 / 3
            m = colour_cut(img, fill_gaps(_fill_holes(P > 0.35, 4096 / s / s), P), box)
            r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat)}
            variants = {"v2": m}
            for k, c in CFG.items():
                ik, t, hi, lo, big = c[:5]
                minpx, refill = (c[5], c[6]) if len(c) > 5 else (30, False)
                n, lab = cells[(ik, t)]
                variants[k] = snap(m, inks[(ik, t)], n, lab, hi, lo, big, minpx, refill, s)
            for k, mk in variants.items():
                pl = mask_to_polygons(mk, 0.005, 6.0)
                r[k], r["v_" + k] = iou(rasterize(pl, (h, w)), tgt), nv(pl)
                r[k + " off1"] = iou(rasterize(offset_polys(pl, 1.0), (h, w)), tgt)
            rows.append(r)
    suf = f"_{shard}" if nsh > 1 else ""
    json.dump(rows, open(f"scripts/pp_probe/rows_cellsnap3_{maps.name}_{split}{suf}.json", "w"))
    print("done", split, shard, len(rows))
