"""Holes of the answer (2026-10-21, descriptive): every connected non-mask region enclosed by the mask (>= 80% of its outer ring
in the mask), mask = lin_fill @2048-equivalent. Per hole (no labels): area / label-free features rect (area / min-area-rect
area: windows ~1), solidity, pmean, and truth: share of the hole inside the label. A hole that is mostly label is a miss
(fill it); one that is not is a real opening (window, door) or another material (keep it)."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from methods import fill_holes
common.WITH_BOX[0] = True
T = 0.35


def sheet(job):
    idx, img, anyann, qs = common.load(job)
    h0, w0 = img.shape[:2]; s = 2048 / max(h0, w0); W, H = round(w0 * s), round(h0 * s)
    rows = []
    for ref, cat, tgt, p, box in qs:
        P = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        t = cv2.resize(tgt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        sp = p.shape[1] / W
        M = fill_holes(P > T, int(4096 / (sp * sp)))
        inv = (~M).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        for i in range(1, n):
            x, y, bw, bh, a = st[i]
            if x == 0 or y == 0 or x + bw >= W or y + bh >= H or a < 20:
                continue
            sel = cc[y:y + bh, x:x + bw] == i
            ring = cv2.dilate(sel.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT).astype(bool) & ~sel
            if M[y:y + bh, x:x + bw][ring].mean() < 0.8:
                continue
            cs, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea)
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            hull = cv2.contourArea(cv2.convexHull(c)) or 1
            rows.append({"sheet": idx, "q": ref, "area": int(a), "label": int(t.sum()), "mask": int(M.sum()),
                         "rect": float(a / max(1, rw * rh)), "solidity": float(a / hull),
                         "pmean": float(P[y:y + bh, x:x + bw][sel].mean()),
                         "tshare": float(t[y:y + bh, x:x + bw][sel].mean())})
    return rows


if __name__ == "__main__":
    split = sys.argv[1]
    R = [r for job in common.jobs(split) for r in sheet(job)]
    json.dump(R, open(f"scripts/pp_probe/rows_holes_{split}.json", "w"))
    print(split, len(R), "holes")
