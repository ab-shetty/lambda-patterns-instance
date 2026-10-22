"""Is the reference's pattern actually there? (2026-10-22; the user's suggestion: shape picks the doubtful parts, a classical
pattern check decides them. Settings fixed before scoring.)
Template = the central square of the reference box (side min(32, box) px @2048, grey); normalised cross-correlation over the
sheet (cv2.TM_CCOEFF_NORMED), then the max over a 9 px neighbourhood (a periodic pattern matches somewhere within a period
whatever the phase). Same spacing + direction + texture -> high; blank, other hatch, other spacing -> low.
Doubtful parts of the mask M (lin_fill @2048): thin parts (local thickness < 20 px) and every piece but the largest.
ref score = median match inside the reference box. A doubtful part is cut when its median match < 0.5 x ref score
(only when ref score > 0.3: the pattern is matchable at all). Rows: base vs pc."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from methods import fill_holes
common.WITH_BOX[0] = True
T = 0.35


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def match_map(gray, x, y, bw, bh):
    side = int(min(32, bw, bh))
    if side < 12:
        return None
    cx, cy = x + bw // 2, y + bh // 2
    tpl = gray[cy - side // 2:cy - side // 2 + side, cx - side // 2:cx - side // 2 + side]
    if tpl.std() < 3:
        return None
    r = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    full = np.full(gray.shape, -1, np.float32)
    full[side // 2:side // 2 + r.shape[0], side // 2:side // 2 + r.shape[1]] = r
    return cv2.dilate(full, np.ones((9, 9), np.uint8))


def sheet(job):
    idx, img, anyann, qs = common.load(job)
    h0, w0 = img.shape[:2]; s = 2048 / max(h0, w0); W, H = round(w0 * s), round(h0 * s)
    gray = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    rows = []
    for ref, cat, tgt, p, box in qs:
        P = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        t = cv2.resize(tgt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        sp = p.shape[1] / W
        M = fill_holes(P > T, int(4096 / (sp * sp)))
        x, y, bw, bh = [int(round(v * s)) for v in box]
        out = M.copy(); refs = None
        mm = match_map(gray, x, y, bw, bh)
        if mm is not None:
            refs = float(np.median(mm[y:y + bh, x:x + bw]))
        if refs is not None and refs > 0.3:
            dt = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 3)
            thick = np.where(M, np.minimum(2 * cv2.dilate(dt, np.ones((25, 25), np.uint8)),
                                           2 * cv2.dilate(dt, np.ones((9, 9), np.uint8)) + 8), 0)
            n, cc, st, _ = cv2.connectedComponentsWithStats(M.astype(np.uint8), connectivity=8)
            big = 1 + int(np.argmax(st[1:, 4])) if n > 1 else 0
            doubt = (M & (thick < 20)) | (M & (cc != big))
            n2, c2, st2, _ = cv2.connectedComponentsWithStats(doubt.astype(np.uint8), connectivity=8)
            for i in range(1, n2):
                sel = c2 == i
                if np.median(mm[sel]) < 0.5 * refs:
                    out &= ~sel
        rows.append({"sheet": idx, "q": ref, "base": iou(M, t), "pc": iou(out, t), "refs": refs})
    return rows


if __name__ == "__main__":
    split = sys.argv[1]
    R = [r for job in common.jobs(split) for r in sheet(job)]
    json.dump(R, open(f"scripts/pp_probe/rows_patcheck_{split}.json", "w"))
    d = np.array([r["pc"] - r["base"] for r in R])
    print(f"{split}: base {np.mean([r['base'] for r in R]):.4f}  pattern-check {d.mean():+.4f}  ({(d > 0.002).sum()} up / "
          f"{(d < -0.002).sum()} down; matchable refs {sum(1 for r in R if r['refs'] and r['refs'] > 0.3)}/{len(R)})")
