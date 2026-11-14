#!/usr/bin/env python3
"""The product output, end to end (2026-11-11, PRODUCT_V3): probability maps at three scales -> polygons for the frontend.

    blend         equal mean of the maps at SCALES (long side 5120 / 4096 / 3072), each linearly upsampled to native size
    product_mask  threshold 0.35, holes <= 4096 px at 4096 scale filled, confidence gaps filled (polygonize.fill_gaps), the
                  chromaticity colour cut against the reference box, snapped to the drawing's cells (polygonize.snap_to_cells)
    polygons      traced, pieces < 0.5% dropped, Douglas-Peucker 6 native px + axis snap, every polygon grown 1 native px
                  outward (the real-sheet labels put the edge on the outer side of the stroke); rings under 5% of the mask
                  re-simplified at 0.1 x sqrt(their area) (simplify_small_rings, 2026-11-12); pieces under 5% of the mask whose
                  mean blended probability is < 0.6 dropped (drop_islands, 2026-11-14)
Numbers (gem2s-x, synth_progress 2026-11-11): valfix4 0.9511 / 1.06x the labeller's vertices, hf14fix4 0.9473 / 1.24x. Tuned on
the real eval sheets; the scales and cell snap hold on gemdev, the 1 px offset is the opposite of Gemini's label convention.

Library: product_polygons(maps, img, box) with maps = {scale: prob_u8}. CLI: score saved maps (the evaluator's --save-probs
layout, MAPS/{split}_{scale}/probs/IIII_QQQ.npz) on a pp_probe split; `evaluate_refunet_selection.py --product` runs the model
at the three scales and then this.
    python3 scripts/product.py valfix4 data/floz-diag/gem2s_x [--out rows.json]"""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polygonize import (T, LOW_VERTEX, PRODUCT_V2, _fill_holes, fill_gaps, snap_to_cells, mask_to_polygons, offset_polys,
                        rasterize, snap_ring, clean_ring)

SCALES = (5120, 4096, 3072)
REF_SCALE = 4096           # the hole-fill size is set at this inference scale, as in every probe
EPS_FLOOR = 4.0            # scaled eps never below 4 native px: without it a 1000-2000 px sheet simplifies at 1.5-3 px and its
                           # outlines stay jagged (valfix4 1.31x, hf14fix4 1.38x vertices / label)


def colour_cut(img, m, box, t=6.0):
    """Chromaticity colour cut (synth_progress 2026-10-21; pp_probe/stack.py `cc`): on a colour sheet (reference fill > 12 Lab
    from the paper), selected pieces whose shadow-invariant chromaticity is > t from the reference's are removed."""
    k = max(5, int(max(img.shape[:2]) / 160) | 1)
    blur = cv2.medianBlur(img, k)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32) * [100 / 255, 1, 1] - [0, 128, 128]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = np.median(lab[g >= np.percentile(g, 90)], 0)
    x, y, bw, bh = box
    if np.linalg.norm(np.median(lab[y:y + bh, x:x + bw].reshape(-1, 3), 0) - paper) <= 12:
        return m
    f = blur.astype(np.float32) + 1
    ch = f / f.sum(-1, keepdims=True) * 100
    d = np.linalg.norm(ch - np.median(ch[y:y + bh, x:x + bw].reshape(-1, 3), 0), axis=-1)
    far = (m & (d > t)).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(far, connectivity=8)
    big = np.zeros(n, bool); big[1:] = st[1:, 4] > 0.0002 * m.size
    cut = big[cc]
    keep = m & ~cv2.dilate(cut.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return cv2.morphologyEx((m & ~cut).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool) | keep


def blend(maps, hw):
    """Equal mean of the probability maps ({scale: uint8 map}), each linearly upsampled to native (h, w)."""
    h, w = hw
    return sum(cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 for p in maps.values()) / len(maps)


SNAP_GUARD = 0.95


def product_mask(P, img, box, guard=SNAP_GUARD):
    """guard (2026-11-14, pp_probe/snapguard.py): the cell snap may not delete pixels whose blended probability is >= guard
    (when they are > 0.1% of the mask). A drawn cell that leaks through a gap into a large unlabelled area has low coverage
    and was dropped whole, confident pattern and all (HF14 11 q0: a hatched triangle with P ~ 1 left as hatch-line spikes).
    Chosen on valfix4 (neutral); hf14fix4 +0.0006 / +0.0004 (fg / gem2s-x), Gemini30 +0.0012 (15 up / 6 down)."""
    h, w = P.shape
    s = REF_SCALE / max(h, w)
    pre = colour_cut(img, fill_gaps(_fill_holes(P > T, 4096 / s / s), P), box)
    snap = snap_to_cells(pre, img, s)
    if guard:
        g = pre & ~snap & (P >= guard)
        if g.sum() > 0.001 * max(1, snap.sum()):
            snap = snap | g
    return snap


def product_eps(hw, scaled=False):
    """Douglas-Peucker tolerance in native px: 6 (v3). scaled (tried 2026-11-11, not the default): PRODUCT_V2's 6 px is at the 4096 inference
    scale, so a 6000 px sheet simplifies at ~9 native px; never below EPS_FLOOR. Worse than 6 px everywhere (valfix4 1.18x,
    hf14fix4 1.37x, Gemini30 1.61x vertices / label vs 1.06x / 1.24x / 1.50x at the same IoU)."""
    return max(EPS_FLOOR, PRODUCT_V2["eps"] * max(hw) / REF_SCALE) if scaled else PRODUCT_V2["eps"]


SMALL_RING = dict(k=0.1, max_frac=0.05)


def simplify_small_rings(polys, total, k=SMALL_RING["k"], max_frac=SMALL_RING["max_frac"]):
    """A ring (outer or hole) under max_frac of the question's mask area is re-simplified at eps = k x sqrt(its area) (then
    axis snap): the holes cast shadows / furniture cut and small false islands are traced with tens of vertices for little
    area (2026-11-12, pp_probe/smallrings.py: total vertices / label hf14fix4 1.77x -> 1.62x, valfix4 1.15x -> 1.02x, Gemini30
    1.76x -> 1.64x for IoU -0.0002 / -0.0005 / +0.0003)."""
    out = []
    for p in polys:
        rings = []
        for r in [p["outer"]] + p["holes"]:
            a = abs(cv2.contourArea(np.asarray(r, np.float32)))
            if a < max_frac * total and len(r) > 4:
                q = cv2.approxPolyDP(np.asarray(r, np.int32).reshape(-1, 1, 2), max(PRODUCT_V2["eps"], k * np.sqrt(a)), True)[:, 0, :]
                q = np.round(clean_ring(snap_ring(q, 6.0))).astype(int)
                r = q.tolist() if len(q) >= 3 else r
            rings.append(r)
        out.append({"outer": rings[0], "holes": rings[1:]})
    return out


ISLANDS = dict(max_frac=0.05, p=0.6)


def drop_islands(polys, P, total, max_frac=ISLANDS["max_frac"], pmin=ISLANDS["p"]):
    """Drop a piece (outer ring with its holes) under max_frac of the question's mask area whose mean blended probability is
    under pmin; never the last piece. The extra HF14 vertices of the fgab model were mostly small low-confidence spill islands
    (2026-11-14, pp_probe/vertsweep.py, chosen on valfix4 + Gemini30, change vs v3 on fgab_fg / fgab_ctl: valfix4 +0.0001 /
    +0.0001 (1 up / 0 down), Gemini30 +0.0025 / +0.0025 (28 up / 7 down; 26 / 10), hf14fix4 +0.0009 / +0.0008; vertices / label
    median hf14fix4 1.46x -> 1.26x / 1.17x -> 1.12x, Gemini30 1.27x -> 1.19x / 1.32x -> 1.19x)."""
    keep = []
    for p in polys:
        if len(polys) > 1 and abs(cv2.contourArea(np.asarray(p["outer"], np.float32))) < max_frac * total:
            m = rasterize([p], P.shape)
            if m.any() and P[m].mean() < pmin:
                continue
        keep.append(p)
    return keep or polys


def product_polygons(maps, img, box, scaled_eps=False, small_rings=True, islands=True):
    """maps: {scale: prob_u8 at that inference scale}; img: native BGR sheet; box: the reference (x, y, w, h), native px."""
    P = blend(maps, img.shape[:2])
    m = product_mask(P, img, box)
    total = max(1, int(m.sum()))
    polys = offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], product_eps(img.shape[:2], scaled_eps)), PRODUCT_V2["offset"])
    polys = simplify_small_rings(polys, total) if small_rings else polys
    return drop_islands(polys, P, total) if islands else polys


def nverts(polys):
    return sum(len(p["outer"]) + sum(len(x) for x in p["holes"]) for p in polys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split")
    ap.add_argument("maps")
    ap.add_argument("--out", default="")
    ap.add_argument("--scaled-eps", action="store_true", help="eps scaled with the sheet's size (product_eps; not the default)")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "pp_probe"))
    import common
    from vertcount import label_vertices
    common.WITH_BOX[0] = True
    common.MAPS = Path(a.maps)
    rows = []
    for job in common.jobs(a.split):
        idx, img, _, qs = common.load(job)
        for ref, cat, tgt, p4, box in qs:
            maps = {z: (p4 if z == 4096 else np.load(Path(a.maps) / f"{common.map_split(a.split)}_{z}" / "probs" /
                                                       f"{idx:03d}_{ref:03d}.npz")["prob"]) for z in SCALES}
            polys = product_polygons(maps, img, box, scaled_eps=a.scaled_eps)
            m = rasterize(polys, img.shape[:2])
            rows.append({"sheet": idx, "q": ref, "iou": common.masked_iou(m, tgt, (idx, ref)), "vertices": nverts(polys),
                         "label_vertices": label_vertices(job[1], cat)})
    r = np.array([x["vertices"] / x["label_vertices"] for x in rows if x["label_vertices"] > 0])
    print(f"{a.split} {Path(a.maps).name}: {len(rows)} q, product mIoU {np.mean([x['iou'] for x in rows]):.4f}, "
          f"vertices / label median {np.median(r):.2f}x (p90 {np.percentile(r, 90):.1f}x)")
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
