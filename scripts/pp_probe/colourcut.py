"""Colour consistency (2026-10-21, thresholds fixed in advance, never tuned on val / HF14 / Gemini30).
On a colour-filled sheet the answer is the reference's paint: a selected piece whose FILL colour (median-blurred so hatch
lines wash out) has a different CHROMATICITY (rgb / (r+g+b): a shadow multiplies the colour and cancels out; a different
paint does not -- raw Lab distance cut val 17's lit wall against a shaded reference) from the reference box's fill: another material, however confident the model is (Gemini30
26: tan and olive lap, roofs). Line-only sheets (reference fill within DE 12 of the paper) are left alone.
  base   lin_fill (linear upsample, threshold 0.35, holes <= 4096 inference px filled)
  cc{T}  base minus the pieces of (base & fill-DE > T) larger than 0.02% of the sheet, then a 3 px opening of what was cut
Rows: per question IoU for base and cc{T}."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from methods import fill_holes
common.WITH_BOX[0] = True
T = 0.35
TS = (4.0, 6.0)        # chromaticity distance (x100): same family incl. shadow p90 <= 2.4 (val 17, Gemini30 26 / 1),
                        # Gemini30 26's wrong material ~6 -- set from those two design cases, not tuned on scores


def fill_lab(img):
    k = max(5, int(max(img.shape[:2]) / 160) | 1)
    return cv2.cvtColor(cv2.medianBlur(img, k), cv2.COLOR_BGR2LAB).astype(np.float32) * [100 / 255, 1, 1] - [0, 128, 128]


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def sheet(job):
    idx, img, anyann, qs = common.load(job)
    h, w = img.shape[:2]
    lab = fill_lab(img)
    k = max(5, int(max(img.shape[:2]) / 160) | 1)
    f = cv2.medianBlur(img, k).astype(np.float32) + 1
    chrom = f / f.sum(-1, keepdims=True) * 100                # rgb chromaticity: shade / shadow (a multiply) cancels
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = np.median(lab[g >= np.percentile(g, 90)], 0)
    rows = []
    for ref, cat, tgt, p, (x, y, bw, bh) in qs:
        P = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        s = p.shape[1] / w
        base = fill_holes(P > T, int(4096 / (s * s)))
        r = {"sheet": idx, "q": ref, "base": iou(base, tgt)}
        rc = np.median(lab[y:y + bh, x:x + bw].reshape(-1, 3), 0)
        colour = np.linalg.norm(rc - paper) > 12
        cr = np.median(chrom[y:y + bh, x:x + bw].reshape(-1, 3), 0)
        d = np.linalg.norm(chrom - cr, axis=-1)          # shadow-invariant: brightness divided out
        for t in TS:
            m = base
            if colour:
                far = (base & (d > t)).astype(np.uint8)
                n, cc, st, _ = cv2.connectedComponentsWithStats(far, connectivity=8)
                big = np.zeros(n, bool); big[1:] = st[1:, 4] > 0.0002 * h * w
                cut = big[cc]
                m = base & ~cut
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool) | (base & ~cv2.dilate(cut.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool))
            r[f"cc{int(t)}"] = iou(m, tgt)
        r["colour"] = bool(colour)
        rows.append(r)
    return rows


if __name__ == "__main__":
    split = sys.argv[1]
    rows = [r for job in common.jobs(split) for r in sheet(job)]
    json.dump(rows, open(f"scripts/pp_probe/rows_colourcut_{split}.json", "w"))
    keys = [k for k in rows[0] if k.startswith("cc")]
    print(split, len(rows), "q; colour q", sum(r["colour"] for r in rows), "; base", round(np.mean([r["base"] for r in rows]), 4),
          " ".join(f"{k} {np.mean([r[k] for r in rows]) - np.mean([r['base'] for r in rows]):+.4f} "
                   f"({sum(r[k] > r['base'] + 0.005 for r in rows)}/{sum(r[k] < r['base'] - 0.005 for r in rows)})" for k in keys))
