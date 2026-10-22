"""Close-out post-processing stack (2026-10-22), scored as the frontend sees it (rasterised polygons).
  base    evaluator mask (threshold 0.35 at inference res, nearest upsample)
  poly    production default: scripts/polygonize.py on the 4096 map
  ms      multi-scale map 0.7 x 4096 + 0.3 x 2048 (synth_progress 2026-10-20: never hurts), then polygons
  cc      chromaticity colour cut (pp_probe/colourcut.py, threshold 6, colour sheets only), then polygons
  ms+cc   both, then polygons
Usage: python3 scripts/pp_probe/stack.py SPLIT [MAPS_DIR]   (MAPS_DIR holds {split}_4096/probs and {split}_2048/probs;
default the current best's maps). Rows -> scripts/pp_probe/rows_stack_{tag}_{split}.json."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from polygonize import probs_to_polygons, rasterize, _fill_holes
common.WITH_BOX[0] = True
T = 0.35


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def colour_cut(img, m, box, t=6.0, adapt=False):
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
    # a flat, light reference colour earns a tighter cut (Gemini30 26: tan vs olive siding at 5.97); chromaticity is
    # noisy on darker fills (HF14 3, val 19 at L 52-54 lose with it), so those keep t
    if adapt and np.median(lab[y:y + bh, x:x + bw, 0]) >= 60:
        t = float(np.clip(4 * np.percentile(d[y:y + bh, x:x + bw], 90), 3.5, t))
    far = (m & (d > t)).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(far, connectivity=8)
    big = np.zeros(n, bool); big[1:] = st[1:, 4] > 0.0002 * m.size
    cut = big[cc]
    out = m & ~cut
    keep = m & ~cv2.dilate(cut.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool) | keep


def paper_cut(img, m, box, ref_min=10.0, paper_max=4.0):
    """Filled vs bare paper (Gemini30 19: grey-filled joist areas labelled, the same lines on white paper not). When the
    reference's fill differs from the paper by > ref_min (Lab, median-blurred so hatch lines wash out), selected pieces
    whose fill is paper (within paper_max) are cut: a shadow never turns paint into paper-white. Neutral greys included --
    the chromaticity cut cannot see grey vs white. Skipped when the reference box itself holds paper."""
    k = max(5, int(max(img.shape[:2]) / 160) | 1)
    lab = cv2.cvtColor(cv2.medianBlur(img, k), cv2.COLOR_BGR2LAB).astype(np.float32) * [100 / 255, 1, 1] - [0, 128, 128]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = np.median(lab[g >= np.percentile(g, 90)], 0)
    x, y, bw, bh = box
    d = np.linalg.norm(lab[y:y + bh, x:x + bw].reshape(-1, 3) - paper, axis=1)
    # the reference must be filled throughout: one whose own box shows paper (val 9/10, a pale roof whose strips are
    # bare between the stripes) may own paper elsewhere too
    if np.median(d) <= ref_min or (d < paper_max).mean() > 0.02:
        return m
    bare = (m & (np.linalg.norm(lab - paper, axis=-1) < paper_max)).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(bare, connectivity=8)
    big = np.zeros(n, bool); big[1:] = st[1:, 4] > 0.0005 * m.size
    cut = big[cc]
    return cv2.morphologyEx((m & ~cut).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool) | \
        (m & ~cv2.dilate(cut.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool))


def colour_add(img, m, box, t=2.0):
    """Same paint inside the answer's outline: on colour sheets, pixels whose chromaticity is within t of the reference's
    (same paint, shadow included: same-family p50 ~0.3, p90 <= 2.4) and that lie inside a closing of the mask (radius ~1% of
    the sheet's long side) are added -- the paint between battens / in blotches the model left out, never a new region."""
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
    r = max(3, int(max(img.shape[:2]) / 100))
    closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    return m | (closed & (d < t))


def to_mask(P_native, s, img=None, box=None, add=False, paper=False, adapt=False):
    m = _fill_holes(P_native > T, 4096 / s / s)
    if img is not None:
        m = colour_cut(img, m, box, adapt=adapt)
        if paper:
            m = paper_cut(img, m, box)
        if add:
            m = colour_add(img, m, box)
    u8 = (np.clip(P_native, 0, 1) * 255).astype(np.uint8)
    # polygons from the (possibly cut) mask: run the production step on a map that is 1 inside, 0 outside
    return rasterize(probs_to_polygons(np.where(m, 255, 0).astype(np.uint8), m.shape), m.shape)


def sheet(job, maps, split):
    idx, img, anyann, qs = common.load(job)
    h, w = img.shape[:2]
    rows = []
    for ref, cat, tgt, p4, box in qs:
        # fix v2 uses the HF14 maps (common.jobs does the same for the 4096 ones)
        f2 = Path(maps) / f"{'hf14' if split.startswith('hf14') else split}_2048" / "probs" / f"{idx:03d}_{ref:03d}.npz"
        P4 = cv2.resize(p4, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        s = p4.shape[1] / w
        r = {"sheet": idx, "q": ref,
             "base": iou(cv2.resize(p4, (w, h), interpolation=cv2.INTER_NEAREST) > int(T * 255), tgt),
             "poly": iou(rasterize(probs_to_polygons(p4, (h, w)), (h, w)), tgt),
             "cc": iou(to_mask(P4, s, img, box), tgt)}
        if f2.exists():
            p2 = np.load(f2)["prob"]
            P = 0.7 * P4 + 0.3 * cv2.resize(p2, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            r["ms"] = iou(to_mask(P, s), tgt)
            r["ms+cc"] = iou(to_mask(P, s, img, box), tgt)
            r["ms+cc+paper"] = iou(to_mask(P, s, img, box, paper=True), tgt)
            r["ms+acc+paper"] = iou(to_mask(P, s, img, box, paper=True, adapt=True), tgt)
        rows.append(r)
    return rows


if __name__ == "__main__":
    split = sys.argv[1]
    maps = sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/xblend-kq30-kms30"
    common.MAPS = Path(maps)
    R = [r for job in common.jobs(split) for r in sheet(job, maps, split)]
    tag = Path(maps).name
    json.dump(R, open(f"scripts/pp_probe/rows_stack_{tag}_{split}.json", "w"))
    keys = [k for k in ("base", "poly", "ms", "cc", "ms+cc", "ms+cc+paper", "ms+acc+paper") if k in R[0]]
    print(split, tag, " ".join(f"{k} {np.mean([r[k] for r in R]):.4f}" for k in keys),
          "| vs poly:", " ".join(f"{k} {np.mean([r[k] - r['poly'] for r in R]):+.4f}" for k in keys if k not in ("base", "poly")))
