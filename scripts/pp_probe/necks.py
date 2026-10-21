"""Spill attached through necks? (2026-10-21, descriptive.) mask M = lin_fill @2048-equivalent. Opening with a disc of radius r
cuts narrow necks; M's pixels then belong to the opened piece they are geodesically nearest (reconstruction by dilation inside
M) or to nothing (slivers the opening removed). Fragments = reconstructed pieces smaller than f x the largest. Report, per r:
target / non-target area in those fragments and in the removed slivers (of label area): dropping them nets target - non."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from methods import fill_holes
T = 0.35
RS = (4, 8, 16)


def sheet(job):
    idx, img, anyann, qs = common.load(job)
    h0, w0 = img.shape[:2]; s = 2048 / max(h0, w0); W, H = round(w0 * s), round(h0 * s)
    out = {r: np.zeros(4) for r in RS}   # frag target, frag non, sliver target, sliver non
    lab = 0
    for q in qs:
        ref, cat, tgt, p = q[:4]
        P = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        t = cv2.resize(tgt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        sp = p.shape[1] / W
        M = fill_holes(P > T, int(4096 / (sp * sp))); lab += t.sum()
        for r in RS:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            O = cv2.morphologyEx(M.astype(np.uint8), cv2.MORPH_OPEN, k)
            n, cc, st, _ = cv2.connectedComponentsWithStats(O, connectivity=8)
            if n <= 1:
                continue
            big = st[1:, 4].max()
            small = np.zeros(n, bool); small[1:] = st[1:, 4] < 0.05 * big
            # reconstruct each opened piece inside M (labels spread by dilation, 3 rounds of r)
            L = cc.astype(np.int32)
            for _ in range(3):
                L = np.where((L == 0) & M, cv2.dilate(L.astype(np.float32), k).astype(np.int32), L)
            frag = small[L] & M; sliver = M & (L == 0)
            out[r] += [(frag & t).sum(), (frag & ~t).sum(), (sliver & t).sum(), (sliver & ~t).sum()]
    return out, lab


if __name__ == "__main__":
    split = sys.argv[1]
    tot = {r: np.zeros(4) for r in RS}; lab = 0
    for job in common.jobs(split):
        o, l = sheet(job); lab += l
        for r in RS:
            tot[r] += o[r]
    for r in RS:
        a = tot[r] / lab
        print(f"{split} r={r:2d}: small fragments target {a[0]:.4f} non {a[1]:.4f} (drop: {a[1]-a[0]:+.4f}) | "
              f"slivers target {a[2]:.4f} non {a[3]:.4f} (drop: {a[3]-a[2]:+.4f})")
