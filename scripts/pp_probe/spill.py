"""Spill shapes (2026-10-22, descriptive). mask M = lin_fill @2048-equivalent. Spill = connected pieces of M & ~label that touch
M & label (attached wrong area). Per spill piece: area, local thickness (2 x max inscribed radius), and how it sits:
  band   thickness < 12 px and long along the correct region's edge (a strip: trim, casing, rake, sill)
  blob   thicker (an adjacent region / part of one)
Area-weighted by thickness bucket, against the CORRECT area of M that has the same local thickness (what a thickness rule
would remove alongside): local thickness of a pixel = 2 x distance transform of M at its nearest medial point (approximated
by the max of the distance transform in a (2r+1) window)."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from methods import fill_holes
T = 0.35
BK = [0, 4, 8, 12, 20, 40, 10 ** 6]


def sheet(job):
    idx, img, anyann, qs = common.load(job)
    h0, w0 = img.shape[:2]; s = 2048 / max(h0, w0); W, H = round(w0 * s), round(h0 * s)
    spill = np.zeros(len(BK) - 1); corr = np.zeros(len(BK) - 1); lab = 0; detached = 0
    for q in qs:
        ref, cat, tgt, p = q[:4]
        P = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        t = cv2.resize(tgt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        sp = p.shape[1] / W
        M = fill_holes(P > T, int(4096 / (sp * sp))); lab += t.sum()
        dt = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 3)
        thick = 2 * cv2.dilate(dt, np.ones((25, 25), np.uint8))           # local thickness (window max of the DT)
        thick = np.where(M, np.minimum(thick, 2 * cv2.dilate(dt, np.ones((9, 9), np.uint8)) + 8), 0)
        fp = (M & ~t).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(fp, connectivity=8)
        tpd = cv2.dilate((M & t).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        att = np.zeros(n, bool)
        if n > 1:
            hit = np.bincount(cc[tpd & (cc > 0)], minlength=n); att = hit > 0
        a = att[cc] & (cc > 0)
        detached += (fp.astype(bool) & ~a).sum()
        b = np.digitize(thick, BK) - 1
        spill += np.bincount(b[a], minlength=len(BK) - 1)[:len(BK) - 1]
        corr += np.bincount(b[M & t], minlength=len(BK) - 1)[:len(BK) - 1]
    return spill, corr, lab, detached


if __name__ == "__main__":
    split = sys.argv[1]
    S = np.zeros(len(BK) - 1); C = np.zeros(len(BK) - 1); L = 0; D = 0
    for job in common.jobs(split):
        s_, c_, l_, d_ = sheet(job); S += s_; C += c_; L += l_; D += d_
    print(f"{split}: attached spill {S.sum()/L:.4f}, detached wrong pieces {D/L:.4f} (of label area)")
    for i in range(len(BK) - 1):
        hi = "inf" if BK[i + 1] > 10 ** 5 else BK[i + 1]
        print(f"  local thickness {BK[i]:>2}-{hi:<4} px: spill {S[i]/L:.4f}   correct area of that thickness {C[i]/L:.4f}")
