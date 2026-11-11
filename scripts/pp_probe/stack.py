"""Close-out post-processing stack (2026-10-22), scored as the frontend sees it (rasterised polygons).
  base    evaluator mask (threshold 0.35 at inference res, nearest upsample)
  poly    production default: scripts/polygonize.py on the 4096 map
  ms      multi-scale map 0.7 x 4096 + 0.3 x 2048 (synth_progress 2026-10-20: never hurts), then polygons
  cc      chromaticity colour cut (pp_probe/colourcut.py, threshold 6, colour sheets only), then polygons
  ms+cc   both, then polygons
  ms3+cc_v2  product v2 (2026-11-11): equal 5120 + 4096 + 3072 blend, colour cut, PRODUCT_V2 polygons (low_vertex at eps 6,
          offset 1 px); only when the 5120 / 3072 maps exist (scripts/gpu_maps_sizes.sh)
  ms3+cc_v3  product v3: v2 after polygonize.snap_to_cells (the drawing's cells, CELL_SNAP)
  *_lv    the same with polygonize's low_vertex option (2026-11-09: irregular hole / inlet fill gated on P >= 0.05, before the
          colour cut; Douglas-Peucker 12 px; pieces < 0.5%). v_<key>: polygon vertices; label_v: the labeller's.
Usage: python3 scripts/pp_probe/stack.py SPLIT [MAPS_DIR]   (MAPS_DIR holds {split}_4096/probs and {split}_2048/probs;
default the current best's maps). Rows -> scripts/pp_probe/rows_stack_{tag}_{split}.json."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from polygonize import probs_to_polygons, rasterize, _fill_holes, fill_gaps, LOW_VERTEX, offset_polys, snap_to_cells
from curve import label_vertices
common.WITH_BOX[0] = True
T = 0.35


CUR = [None]      # (sheet, question) being scored: its label-fix v3 ignore pixels are left out of the prediction


def iou(a, b):
    return common.masked_iou(a, b, CUR[0])


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


def nv(polys):
    return sum(len(p["outer"]) + sum(len(x) for x in p["holes"]) for p in polys)


def to_mask(P_native, s, img=None, box=None, add=False, paper=False, adapt=False, lv=False, out=None, key=None, v2=False,
            v3=False):
    m = _fill_holes(P_native > T, 4096 / s / s)
    if lv or v2 or v3:
        m = fill_gaps(m, P_native)
    if img is not None:
        m = colour_cut(img, m, box, adapt=adapt)
        if paper:
            m = paper_cut(img, m, box)
        if add:
            m = colour_add(img, m, box)
    # polygons from the (possibly cut) mask: run the production step on a map that is 1 inside, 0 outside
    if v3:   # PRODUCT_V3 (2026-11-11): v2 after snapping to the drawing's cells
        m = snap_to_cells(m, img, s)
    if v2 or v3:   # PRODUCT_V2 (2026-11-11): low-vertex settings at eps 6, polygons offset 1 native px
        from polygonize import mask_to_polygons
        polys = offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], 6.0), 1.0)
    else:
        polys = probs_to_polygons(np.where(m, 255, 0).astype(np.uint8), m.shape, **(LOW_VERTEX if lv else {}))
    if out is not None:
        out["v_" + key] = nv(polys)
    return rasterize(polys, m.shape)


def sheet(job, maps, split):
    idx, img, anyann, qs = common.load(job)
    h, w = img.shape[:2]
    rows = []
    for ref, cat, tgt, p4, box in qs:
        CUR[0] = (idx, ref)
        # label-fix splits use their base split's maps (common.jobs does the same for the 4096 ones)
        f2 = Path(maps) / f"{common.map_split(split)}_2048" / "probs" / f"{idx:03d}_{ref:03d}.npz"
        P4 = cv2.resize(p4, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        s = p4.shape[1] / w
        pp, pl = probs_to_polygons(p4, (h, w)), probs_to_polygons(p4, (h, w), low_vertex=True)
        r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat),
             "base": iou(cv2.resize(p4, (w, h), interpolation=cv2.INTER_NEAREST) > int(T * 255), tgt),
             "poly": iou(rasterize(pp, (h, w)), tgt), "v_poly": nv(pp),
             "poly_lv": iou(rasterize(pl, (h, w)), tgt), "v_poly_lv": nv(pl),
             "cc": iou(to_mask(P4, s, img, box), tgt)}
        if f2.exists():
            p2 = np.load(f2)["prob"]
            P = 0.7 * P4 + 0.3 * cv2.resize(p2, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            r["ms"] = iou(to_mask(P, s), tgt)
            r["ms+cc"] = iou(to_mask(P, s, img, box, out=r, key="ms+cc"), tgt)
            r["ms+cc_lv"] = iou(to_mask(P, s, img, box, lv=True, out=r, key="ms+cc_lv"), tgt)
            f5, f3 = (Path(maps) / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz" for z in (5120, 3072))
            if f5.exists() and f3.exists():   # product v2: equal 5120 + 4096 + 3072 blend, PRODUCT_V2 polygons
                P3 = (P4 + sum(cv2.resize(np.load(f)["prob"], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                               for f in (f5, f3))) / 3
                r["ms3+cc_v2"] = iou(to_mask(P3, s, img, box, v2=True, out=r, key="ms3+cc_v2"), tgt)
                r["ms3+cc_v3"] = iou(to_mask(P3, s, img, box, v3=True, out=r, key="ms3+cc_v3"), tgt)
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
    keys = [k for k in ("base", "poly", "poly_lv", "ms", "cc", "ms+cc", "ms+cc_lv", "ms3+cc_v2", "ms3+cc_v3", "ms+cc+paper", "ms+acc+paper") if k in R[0]]
    print(split, tag, " ".join(f"{k} {np.mean([r[k] for r in R]):.4f}" for k in keys),
          "| vs poly:", " ".join(f"{k} {np.mean([r[k] - r['poly'] for r in R]):+.4f}" for k in keys if k not in ("base", "poly")))
    L = [r for r in R if r["label_v"] > 0]
    def ratio(k):
        return np.array([r["v_" + k] / r["label_v"] for r in L])
    print("   vertices / label, median (p90):", " ".join(
        f"{k} {np.median(ratio(k)):.2f}x ({np.percentile(ratio(k), 90):.1f}x)" for k in ("poly", "poly_lv", "ms+cc", "ms+cc_lv", "ms3+cc_v2", "ms3+cc_v3") if "v_" + k in L[0]))
