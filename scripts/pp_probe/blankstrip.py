"""Blank-strip cut (2026-10-22; rule fixed before scoring). Spill attached to a correct region is mostly 12-20 px thick (@2048):
trim boards, casings, rakes, sills -- but correct area of that thickness (piers between windows, bands under eaves) is 2-3x
larger, so shape cannot decide. Content can: a trim board is BLANK (no hatch), a pier carries the reference's lines. Rule:
thin parts of the mask (local thickness < 20 px @2048) are cut when their stroke density (ink v2, 15 px blur) is < 0.4 x the
reference box's (the taxonomy probe's 'blank' definition) -- only when the reference itself is textured (density > 0.02).
Rows: per question IoU, base (lin_fill) vs bs."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from common import ink_map_v2
from methods import fill_holes
common.WITH_BOX[0] = True
T = 0.35


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def sheet(job):
    idx, img, anyann, qs = common.load(job)
    h0, w0 = img.shape[:2]; s = 2048 / max(h0, w0); W, H = round(w0 * s), round(h0 * s)
    small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    dens = cv2.blur(ink_map_v2(small).astype(np.float32), (15, 15))
    rows = []
    for ref, cat, tgt, p, box in qs:
        P = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        t = cv2.resize(tgt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        sp = p.shape[1] / W
        M = fill_holes(P > T, int(4096 / (sp * sp)))
        x, y, bw, bh = [int(round(v * s)) for v in box]
        rd = float(dens[y:y + bh, x:x + bw].mean()) if bw > 0 and bh > 0 else 0.0
        out = M.copy()
        if rd > 0.02:
            dt = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 3)
            thick = np.where(M, np.minimum(2 * cv2.dilate(dt, np.ones((25, 25), np.uint8)),
                                           2 * cv2.dilate(dt, np.ones((9, 9), np.uint8)) + 8), 0)
            thin = (M & (thick < 20)).astype(np.uint8)
            n, cc, st, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
            if n > 1:
                md = np.bincount(cc.ravel(), weights=dens.ravel(), minlength=n) / np.maximum(1, st[:, 4])
                cut = np.zeros(n, bool); cut[1:] = md[1:] < 0.4 * rd
                out = M & ~cut[cc]
        rows.append({"sheet": idx, "q": ref, "base": iou(M, t), "bs": iou(out, t), "removed": int((M & ~out).sum())})
    return rows


if __name__ == "__main__":
    split = sys.argv[1]
    R = [r for job in common.jobs(split) for r in sheet(job)]
    json.dump(R, open(f"scripts/pp_probe/rows_blankstrip_{split}.json", "w"))
    d = np.array([r["bs"] - r["base"] for r in R])
    print(f"{split}: base {np.mean([r['base'] for r in R]):.4f}  blank-strip {d.mean():+.4f}  ({(d > 0.002).sum()} up / {(d < -0.002).sum()} down)")
