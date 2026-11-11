#!/usr/bin/env python3
"""The product output, end to end (2026-11-11, PRODUCT_V3): probability maps at three scales -> polygons for the frontend.

    blend         equal mean of the maps at SCALES (long side 5120 / 4096 / 3072), each linearly upsampled to native size
    product_mask  threshold 0.35, holes <= 4096 px at 4096 scale filled, confidence gaps filled (polygonize.fill_gaps), the
                  chromaticity colour cut against the reference box, snapped to the drawing's cells (polygonize.snap_to_cells)
    polygons      traced, pieces < 0.5% dropped, Douglas-Peucker 6 px + axis snap, every polygon grown 1 native px outward
                  (the real-sheet labels put the edge on the outer side of the stroke)
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
                        rasterize)

SCALES = (5120, 4096, 3072)
REF_SCALE = 4096           # the hole-fill size is set at this inference scale, as in every probe


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


def product_mask(P, img, box):
    h, w = P.shape
    s = REF_SCALE / max(h, w)
    m = fill_gaps(_fill_holes(P > T, 4096 / s / s), P)
    return snap_to_cells(colour_cut(img, m, box), img, s)


def product_polygons(maps, img, box):
    """maps: {scale: prob_u8 at that inference scale}; img: native BGR sheet; box: the reference (x, y, w, h), native px."""
    m = product_mask(blend(maps, img.shape[:2]), img, box)
    return offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], PRODUCT_V2["eps"]), PRODUCT_V2["offset"])


def nverts(polys):
    return sum(len(p["outer"]) + sum(len(x) for x in p["holes"]) for p in polys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split")
    ap.add_argument("maps")
    ap.add_argument("--out", default="")
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
            polys = product_polygons(maps, img, box)
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
