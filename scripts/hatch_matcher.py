#!/usr/bin/env python3
"""Reference-conditioned matcher for CAD hatch, using drafting parameters.

Measures orientation (structure tensor), spacing (distance transform of the
PAPER -- inside a hatch of pitch d it peaks at d/2, so its histogram is a spacing
spectrum) and ink density, pooled over a few pitches, then compares by
Bhattacharyya. max(R,G,B) as the working channel suppresses highlighter markup.
`--contrast margin` subtracts the best rival material family, since pattern IDs
are image-local and therefore mutually exclusive.

HF14, native input, best config (validation-tuned, one confirmation run):

    generic Gabor bank   AUC 0.727  IoU 0.308   4% of selections >0.90
    this matcher         AUC 0.897  IoU 0.505  63%

IoU is oracle-thresholded per selection for both -- an upper bound, not a product
metric. Not complementary to RefUNet: on identical selections the two correlate
+0.45 (AUC) / +0.65 (IoU), and this matcher's IoU falls to 0.252 on RefUNet's
worst 13, so it is not worth feeding in as a feature. Resolution matters here
where it did not for the Gabor bank: 1280 gives 0.775, native 0.897.

Measured and rejected: per-region decisions via ink closing (`--mode cell`; GT
regions are mostly white inside, so covering them claims 30-80% of the sheet) and
via planar subdivision (oracle ceiling IoU 0.257 -- scanned line networks have
gaps and one gap merges two faces); stroke weight (`--w-wid`, 0.877->0.852);
relevance feedback (`--feedback`, 0.791->0.783, drifts); blank-paper gating
(`--min-ink`, 0.791->0.736, deletes white interiors). 74% of remaining false
positives are real linework labelled as a different material.

`cv2.kmeans` is unseeded, so cluster-based arms vary run to run by an unmeasured
amount; the margin's +0.041 IoU is clear of it, `--ref-parts 3`'s +0.011 is not.

Usage:
    PYTHONPATH=. python3 scripts/hatch_matcher.py --mode pixel --input-size 0 \
        --clusters 8 --contrast margin --contrast-w 0.5 --ref-parts 3
"""

import argparse
import csv
import io
import json
import os
import random
import time

import cv2
import numpy as np
from PIL import Image

from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask, sample_reference_box

VALIDATION = "4,5,6,8,9,10,13,15,17,19,20,21,22,26"
HF14 = "12,16,27,7,11,25,23,1,18,2,0,3,14,24"

N_ORI = 18                       # orientation bins over [0, pi)
DT_EDGES = np.geomspace(0.5, 48.0, 17)   # spacing spectrum, log-spaced
WID_EDGES = np.geomspace(0.5, 16.0, 13)  # stroke-weight spectrum


# ---------------------------------------------------------------- preprocessing

def paper_gray(rgb):
    """max(R,G,B). Ink stays dark; saturated highlighter lifts toward paper."""
    return rgb.max(axis=2).astype(np.float32) / 255.0


def flat_field(gray, ksize=51):
    """Divide out slowly varying paper tone (scan shading, JPEG blotches)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    bg = cv2.GaussianBlur(bg, (0, 0), ksize / 6.0)
    return np.clip(gray / np.maximum(bg, 1e-3), 0.0, 1.5)


def sauvola(gray, w=25, k=0.2, R=0.25):
    """Local adaptive threshold. Robust to faint linework and uneven scans."""
    w = int(w) | 1
    m = cv2.boxFilter(gray, cv2.CV_32F, (w, w))
    m2 = cv2.boxFilter(gray * gray, cv2.CV_32F, (w, w))
    sd = np.sqrt(np.maximum(m2 - m * m, 0.0))
    return (gray < m * (1.0 + k * (sd / R - 1.0))).astype(np.uint8)


# ------------------------------------------------------------------- structure

def structure_tensor(gray, sigma=2.0):
    """Local orientation and coherence of the linework.

    Returns theta in [0, pi) (gradient direction, so hatch stroke direction + 90),
    a coherence in [0,1] that is high on clean parallel strokes and low on noise,
    and the gradient energy used to weight the histogram.
    """
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    j11 = cv2.GaussianBlur(ix * ix, (0, 0), sigma)
    j22 = cv2.GaussianBlur(iy * iy, (0, 0), sigma)
    j12 = cv2.GaussianBlur(ix * iy, (0, 0), sigma)
    theta = 0.5 * np.arctan2(2.0 * j12, j11 - j22)          # [-pi/2, pi/2)
    theta = np.mod(theta, np.pi)
    trace = j11 + j22
    aniso = np.sqrt(np.maximum((j11 - j22) ** 2 + 4.0 * j12 * j12, 0.0))
    coherence = aniso / np.maximum(trace, 1e-8)
    return theta.astype(np.float32), coherence.astype(np.float32), trace


def estimate_pitch(ink):
    """Typical hatch half-pitch, measured only where there is linework.

    Distance-to-nearest-ink peaks at d/2 inside a hatch of pitch d, but taken
    over a whole sheet the statistic is swamped by blank paper, which has huge DT
    values and no pattern in it at all. Restricting to locally inky neighbourhoods
    is what makes this an estimate of hatch spacing rather than of page
    emptiness: on image 8 it moves the estimate from 17.9px to 4.8px, and a
    closing radius built on the former merges the whole drawing into one blob.
    """
    dt = cv2.distanceTransform((1 - ink).astype(np.uint8), cv2.DIST_L2, 3)
    dens = cv2.boxFilter(ink.astype(np.float32), -1, (31, 31))
    vals = dt[(dt > 0) & (dt < 40) & (dens > 0.05)]
    half = float(np.percentile(vals, 60)) if vals.size > 100 else 3.0
    return dt, float(np.clip(half, 1.0, 12.0))


def build_cells(ink, half_pitch, min_area, use_barriers=True):
    """Partition the drawing into filled candidate regions.

    Closing with a radius above the hatch pitch turns a stroke field into the
    solid area the pattern occupies. Long isolated segments (region outlines,
    walls) are cut out first so that two differently-hatched regions sharing a
    boundary do not merge into one cell. Hatch strokes themselves are excluded
    from the barrier set by an isolation test: a stroke belonging to a periodic
    family has a parallel neighbour within about one pitch, an outline does not.

    The result is then cut by a grid of ~4 pitches. Under-segmentation is the
    fatal error here -- one cell spanning two materials can never be scored
    correctly, and the whole method collapses to a constant map -- whereas
    over-segmentation costs only a little boundary blockiness, since the
    descriptor is pooled over a wider context than the cell it labels.
    """
    r = int(max(3, round(1.5 * half_pitch))) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    filled = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)

    barrier = np.zeros_like(ink)
    if use_barriers:
        lsd = cv2.createLineSegmentDetector()
        segs = lsd.detect((1 - ink) * 255)[0]
        if segs is not None:
            dt_paper = cv2.distanceTransform((1 - ink).astype(np.uint8),
                                             cv2.DIST_L2, 3)
            h, w = ink.shape
            min_len = max(30.0, 6.0 * half_pitch)
            for s in segs.reshape(-1, 4):
                x1, y1, x2, y2 = s
                dx, dy = x2 - x1, y2 - y1
                L = float(np.hypot(dx, dy))
                if L < min_len:
                    continue
                # Sample the paper clearance on both sides at the midpoint.
                nx, ny = -dy / L, dx / L
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                off = max(2.0, half_pitch)
                clear = []
                for sgn in (1, -1):
                    px = int(round(mx + sgn * off * nx))
                    py = int(round(my + sgn * off * ny))
                    if 0 <= px < w and 0 <= py < h:
                        clear.append(dt_paper[py, px])
                # Isolated on both sides -> an outline, not one stroke of a hatch.
                if len(clear) == 2 and min(clear) > 0.9 * half_pitch:
                    cv2.line(barrier, (int(x1), int(y1)), (int(x2), int(y2)), 1,
                             thickness=max(1, int(round(half_pitch))))

    region = (filled > 0) & (barrier == 0)
    n, cc = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(cc, np.int32), 0, filled

    h, w = cc.shape
    g = int(max(12, round(4.0 * half_pitch)))
    block = (np.arange(h)[:, None] // g) * (w // g + 1) + (np.arange(w)[None, :] // g)
    combo = cc.astype(np.int64) * (int(block.max()) + 1) + block
    combo[cc == 0] = -1

    uniq, inv = np.unique(combo, return_inverse=True)
    labels = inv.reshape(cc.shape).astype(np.int32) + 1
    labels[cc == 0] = 0

    areas = np.bincount(labels.ravel(), minlength=labels.max() + 1)
    small = np.where(areas < min_area)[0]
    if small.size:
        labels[np.isin(labels, small[small > 0])] = 0
    keep = np.unique(labels)
    keep = keep[keep > 0]
    remap = np.zeros(labels.max() + 1, np.int32)
    remap[keep] = np.arange(1, keep.size + 1)
    return remap[labels].astype(np.int32), int(keep.size), filled


# ------------------------------------------------------------------ descriptors

def dense_histograms(gray, ink, dt, pool):
    """Orientation and spacing histograms per pixel, pooled over a context window.

    Descriptor support (this window) is deliberately decoupled from decision
    granularity (a cell). The window has to span several pitches before an
    orientation or spacing histogram is stable, while a cell has to stay small
    enough not to straddle a material boundary. Tying the two together forces a
    choice between unstable descriptors and cells that span two patterns.
    """
    theta, coh, energy = structure_tensor(gray)
    w = (np.sqrt(np.maximum(energy, 0.0)) * coh).astype(np.float32)
    b = theta / np.pi * N_ORI
    lo = np.floor(b).astype(np.int32) % N_ORI
    frac = (b - np.floor(b)).astype(np.float32)
    hi = (lo + 1) % N_ORI

    k = int(pool) | 1
    h, wd = gray.shape
    ori = np.empty((h, wd, N_ORI), np.float32)
    for i in range(N_ORI):
        acc = w * ((lo == i) * (1.0 - frac) + (hi == i) * frac)
        ori[:, :, i] = cv2.boxFilter(acc, cv2.CV_32F, (k, k), normalize=False)

    n_dt = len(DT_EDGES) - 1
    dt_bin = np.clip(np.digitize(dt, DT_EDGES) - 1, 0, n_dt - 1)
    paper = (ink < 0.5)
    spa = np.empty((h, wd, n_dt), np.float32)
    for i in range(n_dt):
        acc = (paper & (dt_bin == i)).astype(np.float32)
        spa[:, :, i] = cv2.boxFilter(acc, cv2.CV_32F, (k, k), normalize=False)

    # Stroke weight. Distance-to-edge inside the ink is half the local line
    # width, so its histogram is a line-weight spectrum. In CAD this is a
    # deliberate drafting choice -- heavy poche versus light hatch -- and it
    # separates materials that share an orientation and a spacing.
    ink_dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    n_w = len(WID_EDGES) - 1
    w_bin = np.clip(np.digitize(ink_dt, WID_EDGES) - 1, 0, n_w - 1)
    inkm = ink > 0
    wid = np.empty((h, wd, n_w), np.float32)
    for i in range(n_w):
        acc = (inkm & (w_bin == i)).astype(np.float32)
        wid[:, :, i] = cv2.boxFilter(acc, cv2.CV_32F, (k, k), normalize=False)

    dens = cv2.boxFilter(ink.astype(np.float32), cv2.CV_32F, (k, k))
    # boxFilter accumulates running sums, so exact zeros can come back as tiny
    # negatives; Bhattacharyya takes a square root and would produce NaN.
    for v in (ori, spa, wid):
        np.maximum(v, 0.0, out=v)
    ori /= np.maximum(ori.sum(2, keepdims=True), 1e-9)
    spa /= np.maximum(spa.sum(2, keepdims=True), 1e-9)
    wid /= np.maximum(wid.sum(2, keepdims=True), 1e-9)
    return ori, spa, wid, dens


def region_descriptor(ori, spa, wid, dens, labels, n_labels):
    """Mean pooled descriptor per label: orientation | spacing | weight | density."""
    flat = labels.ravel()
    sel = flat > 0
    lab = flat[sel] - 1
    cnt = np.maximum(np.bincount(lab, minlength=n_labels).astype(np.float64), 1)

    def agg(vol):
        m = np.stack([np.bincount(lab, weights=vol[:, :, i].ravel()[sel],
                                  minlength=n_labels)
                      for i in range(vol.shape[2])], 1)
        return m / cnt[:, None]

    O, S, W = agg(ori), agg(spa), agg(wid)
    D = np.bincount(lab, weights=dens.ravel()[sel], minlength=n_labels) / cnt

    O = circular_smooth(O)
    O /= np.maximum(O.sum(1, keepdims=True), 1e-9)
    for M in (S, W):
        M[:] = smooth1d(M)
    S /= np.maximum(S.sum(1, keepdims=True), 1e-9)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    return O, S, W, D


def circular_smooth(h, passes=1):
    for _ in range(passes):
        h = 0.25 * np.roll(h, 1, 1) + 0.5 * h + 0.25 * np.roll(h, -1, 1)
    return h


def smooth1d(h, passes=1):
    for _ in range(passes):
        p = np.pad(h, ((0, 0), (1, 1)), mode="edge")
        h = 0.25 * p[:, :-2] + 0.5 * h + 0.25 * p[:, 2:]
    return h


def similarity(ori, spa, wid, den, r_ori, r_spa, r_wid, r_den,
               w_ori, w_spa, w_wid, w_den):
    """Per-component comparison, each on its natural scale.

    Orientation and spacing are distributions -> Bhattacharyya coefficient, which
    is bounded in [0,1] and tolerant of small shifts after smoothing. Density is a
    scalar -> exponential falloff.
    """
    s_ori = np.sqrt(np.maximum(ori * r_ori[None, :], 0)).sum(1)
    s_spa = np.sqrt(np.maximum(spa * r_spa[None, :], 0)).sum(1)
    s_wid = np.sqrt(np.maximum(wid * r_wid[None, :], 0)).sum(1)
    s_den = np.exp(-np.abs(den - r_den) / 0.15)
    total = w_ori + w_spa + w_wid + w_den
    return (w_ori * s_ori + w_spa * s_spa + w_wid * s_wid + w_den * s_den) / total


# -------------------------------------------------------------------- evaluate

def auc(inside, outside, n=4000, rng=None):
    rng = rng or np.random
    a = rng.choice(inside, min(n, inside.size), replace=False)
    b = rng.choice(outside, min(n, outside.size), replace=False)
    return float((a[:, None] > b[None, :]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indices", default=VALIDATION)
    ap.add_argument("--input-size", type=int, default=1280, help="0 = native")
    ap.add_argument("--eval-size", type=int, default=1280)
    ap.add_argument("--w-ori", type=float, default=1.0)
    ap.add_argument("--w-spa", type=float, default=1.0)
    ap.add_argument("--w-wid", type=float, default=0.0,
                    help="Stroke weight. Measured negative (0.877->0.852 at equal weight) -- CAD strokes are 1-2px, so at scan quality this is mostly noise.")
    ap.add_argument("--w-den", type=float, default=0.25)
    ap.add_argument("--min-area", type=int, default=64)
    ap.add_argument("--pool-pitches", type=float, default=6.0,
                    help="Descriptor context window, in half-pitches.")
    ap.add_argument("--no-barriers", action="store_true")
    ap.add_argument("--clusters", type=int, default=0,
                    help="Material families to segment per image (0 = off). The "
                         "task is image-local, so a drawing holds a handful of "
                         "mutually exclusive materials; clustering the pooled "
                         "descriptor and scoring per family suppresses the "
                         "false positives that dominate the error budget.")
    ap.add_argument("--contrast", choices=["none","hard","margin"], default="none",
                    help="none: absolute similarity. hard: one score per cluster. "
                         "margin: reference similarity minus the best rival family.")
    ap.add_argument("--contrast-w", type=float, default=1.0)
    ap.add_argument("--ref-parts", type=int, default=1,
                    help="Split the reference box into NxN parts and keep the "
                         "best-matching one (mixture instead of mean).")
    ap.add_argument("--feedback", type=int, default=0,
                    help="Relevance-feedback iterations.")
    ap.add_argument("--feedback-frac", type=float, default=0.10)
    ap.add_argument("--feedback-alpha", type=float, default=0.5)
    ap.add_argument("--min-ink", type=float, default=0.0,
                    help="Suppress the score where pooled ink density is below "
                         "this. 18%% of false positives sit on blank paper, "
                         "which carries no pattern by definition.")
    ap.add_argument("--mode", choices=["cell", "pixel"], default="cell",
                    help="cell: one decision per region (needs region recovery to "
                         "work). pixel: score every pixel, directly comparable to "
                         "the Gabor bank.")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    key = args.indices.strip().lower()
    indices = {"validation": VALIDATION, "val": VALIDATION, "hf14": HF14}.get(key, args.indices)
    idxs = [int(v) for v in indices.split(",") if v.strip()]

    ds = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                              config="real-world-test", split="test")

    rows = []
    for image_idx in idxs:
        rec = ds[image_idx]
        im0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        h0, w0 = im0.shape[:2]
        masks0 = [render_instance_mask(a["segmentation"], h0, w0).astype(bool)
                  for a in anns]
        cats = [a.get("category_name", "pattern") for a in anns]

        native = max(h0, w0)
        long_side = native if args.input_size == 0 else args.input_size
        s = long_side / native
        nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
        rgb = cv2.resize(im0, (nw, nh), interpolation=cv2.INTER_AREA)

        t0 = time.time()
        gray = flat_field(paper_gray(rgb))
        ink = sauvola(gray)
        dt, half_pitch = estimate_pitch(ink)
        labels, n_cells, _ = build_cells(ink, half_pitch, args.min_area,
                                         use_barriers=not args.no_barriers)
        if n_cells == 0:
            print(f"  image {image_idx:3d}: no cells found, skipped")
            continue
        pool = max(15, int(round(args.pool_pitches * half_pitch)))
        ori_v, spa_v, wid_v, dens_v = dense_histograms(gray, ink, dt, pool)
        c_ori, c_spa, c_wid, c_den = region_descriptor(ori_v, spa_v, wid_v, dens_v,
                                                       labels, n_cells)
        if args.mode == "pixel":
            sq_ori, sq_spa, sq_wid = (np.sqrt(ori_v), np.sqrt(spa_v), np.sqrt(wid_v))

        km_lab = km_ori = km_spa = km_den = None
        if args.clusters > 0 and args.mode == "pixel":
            # In sqrt-space a dot product IS the Bhattacharyya coefficient, so
            # Euclidean k-means here is clustering by Hellinger distance -- the
            # same geometry the matcher scores in.
            feat = np.concatenate(
                [sq_ori.reshape(-1, N_ORI),
                 sq_spa.reshape(-1, spa_v.shape[2]),
                 dens_v.reshape(-1, 1)], 1).astype(np.float32)
            step = max(1, feat.shape[0] // 40000)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
            _, _, cent = cv2.kmeans(feat[::step], args.clusters, None, crit, 3,
                                    cv2.KMEANS_PP_CENTERS)
            # Assign every pixel to its nearest centre: ||x-c||^2 ranked by
            # (x.c - |c|^2/2), one GEMM instead of a K-way distance tensor.
            score = feat @ cent.T - 0.5 * (cent * cent).sum(1)[None, :]
            km_lab = np.argmax(score, 1).reshape(gray.shape).astype(np.int32) + 1
            km_ori, km_spa, km_wid, km_den = region_descriptor(
                ori_v, spa_v, wid_v, dens_v, km_lab, args.clusters)
        dt_ms = (time.time() - t0) * 1000

        # Eval grid: identical to texture_resolution_sweep.
        es = args.eval_size / native
        eh, ew = max(1, round(h0 * es)), max(1, round(w0 * es))
        rng = np.random.default_rng(0)

        for ref_idx, (m0, cat) in enumerate(zip(masks0, cats)):
            r = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
            x, y, bw, bh = sample_reference_box(m0, 128, 512, rng=r)
            # Box in working (input) resolution, for the reference descriptor.
            bx, by = max(0, round(x * s)), max(0, round(y * s))
            bw_, bh_ = max(4, round(bw * s)), max(4, round(bh * s))
            bx, by = min(bx, nw - 2), min(by, nh - 2)
            bw_, bh_ = min(bw_, nw - bx), min(bh_, nh - by)
            box = np.zeros((nh, nw), np.int32)
            box[by:by + bh_, bx:bx + bw_] = 1
            if box.sum() < 16:
                continue
            # Reference model. A single mean histogram over the box is a thin
            # model of a material that varies across a region, and a box
            # straddling two textures yields a blend that matches neither
            # strongly while weakly matching many things -- exactly the
            # false-positive signature the error budget points at. Splitting the
            # box into parts and keeping the best-matching part models the
            # reference as a mixture instead of a mean.
            P = max(1, args.ref_parts)
            b_ori, b_spa, b_wid, b_den = region_descriptor(ori_v, spa_v, wid_v,
                                                           dens_v, box, 1)
            if P > 1:
                gy = np.minimum(((np.arange(nh) - by) * P) // max(bh_, 1), P - 1)
                gx = np.minimum(((np.arange(nw) - bx) * P) // max(bw_, 1), P - 1)
                parts = np.clip(gy, 0, P - 1)[:, None] * P + np.clip(gx, 0, P - 1)[None, :]
                parts = np.where(box > 0, parts.astype(np.int32) + 1, 0)
                n_parts = P * P
                r_ori, r_spa, r_wid, r_den = region_descriptor(
                    ori_v, spa_v, wid_v, dens_v, parts, n_parts)
            else:
                n_parts = 1
                r_ori, r_spa, r_wid, r_den = b_ori, b_spa, b_wid, b_den

            if args.mode == "cell":
                score_cells = similarity(c_ori, c_spa, c_wid, c_den,
                                         r_ori[0], r_spa[0], r_wid[0], r_den[0],
                                         args.w_ori, args.w_spa, args.w_wid,
                                         args.w_den)
                lut = np.concatenate([[0.0], score_cells]).astype(np.float32)
                sim = lut[labels]
            elif km_lab is not None and args.contrast == "hard":
                sc = similarity(km_ori, km_spa, km_wid, km_den,
                                r_ori[0], r_spa[0], r_wid[0], r_den[0],
                                args.w_ori, args.w_spa, args.w_wid, args.w_den)
                sim = np.concatenate([[0.0], sc]).astype(np.float32)[km_lab]
            else:
                tot = args.w_ori + args.w_spa + args.w_wid + args.w_den

                def score_map(o, s, w, d):
                    """Bhattacharyya to one descriptor, evaluated at every pixel."""
                    return (args.w_ori * (sq_ori @ np.sqrt(o).astype(np.float32))
                            + args.w_spa * (sq_spa @ np.sqrt(s).astype(np.float32))
                            + args.w_wid * (sq_wid @ np.sqrt(w).astype(np.float32))
                            + args.w_den * np.exp(-np.abs(dens_v - d) / 0.15)) / tot

                def full_score(o, s, w, d):
                    m = score_map(o, s, w, d)
                    if args.contrast == "margin" and km_lab is not None:
                        # A pixel should count only if it looks MORE like the
                        # reference than like any other material in this drawing.
                        # Absolute similarity lets generically-hatched areas
                        # score high everywhere, which is what the error budget
                        # says dominates: 92% of false positives are far from any
                        # boundary. Subtracting the best rival family turns the
                        # score into a margin, making the image-local
                        # exclusivity of the task explicit.
                        r2c = similarity(km_ori, km_spa, km_wid, km_den,
                                         o, s, w, d, args.w_ori, args.w_spa,
                                         args.w_wid, args.w_den)
                        k_star = int(np.argmax(r2c))
                        alt = None
                        for k in range(args.clusters):
                            if k == k_star:
                                continue
                            a = score_map(km_ori[k], km_spa[k], km_wid[k], km_den[k])
                            alt = a if alt is None else np.maximum(alt, a)
                        if alt is not None:
                            m = np.clip(m - args.contrast_w * alt + 0.5, 0, 1)
                    return m

                sim = None
                for p in range(n_parts):
                    m = full_score(r_ori[p], r_spa[p], r_wid[p], r_den[p])
                    sim = m if sim is None else np.maximum(sim, m)

                # Relevance feedback. One box is a small sample of a material;
                # the pixels the first pass is most confident about are a much
                # larger one. Blending back toward the user's own box bounds the
                # drift if that first pass was wrong.
                for _ in range(args.feedback):
                    thr = float(np.quantile(sim, 1.0 - args.feedback_frac))
                    conf = (sim >= thr).astype(np.int32)
                    if int(conf.sum()) < 64:
                        break
                    f_o, f_s, f_w, f_d = region_descriptor(ori_v, spa_v, wid_v,
                                                           dens_v, conf, 1)
                    a = args.feedback_alpha
                    def blend(u, v):
                        z = (1.0 - a) * u + a * v
                        return z / max(float(z.sum()), 1e-9)
                    sim = full_score(blend(b_ori[0], f_o[0]),
                                     blend(b_spa[0], f_s[0]),
                                     blend(b_wid[0], f_w[0]),
                                     (1.0 - a) * b_den[0] + a * f_d[0])

                if args.min_ink > 0:
                    sim = np.where(dens_v < args.min_ink, 0.0, sim)
            sim = cv2.resize(sim, (ew, eh), interpolation=cv2.INTER_NEAREST)

            tgt = np.logical_or.reduce([m for m, c in zip(masks0, cats) if c == cat])
            tgt = cv2.resize(tgt.astype(np.uint8), (ew, eh),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
            if tgt.sum() < 10 or (~tgt).sum() < 10:
                continue
            a = auc(sim[tgt], sim[~tgt], rng=rng)
            best = max((((sim > t) & tgt).sum() / max(((sim > t) | tgt).sum(), 1))
                       for t in np.arange(0.0, 1.0, 0.02))
            rows.append(dict(image=image_idx, ref=ref_idx, native=native,
                             input_size=long_side, cells=n_cells,
                             half_pitch=round(half_pitch, 2),
                             auc=a, iou=float(best)))
        print(f"  image {image_idx:3d} native={native:5d} cells={n_cells:4d} "
              f"half_pitch={half_pitch:4.1f} {dt_ms:6.0f}ms", flush=True)

    A = np.array([r["auc"] for r in rows])
    B = np.array([r["iou"] for r in rows])
    print(f"\n{len(idxs)} images, {len(rows)} selections, "
          f"input={'native' if args.input_size == 0 else args.input_size}, "
          f"eval grid={args.eval_size}px")
    print(f"  hatch matcher  AUC  mean={A.mean():.3f} median={np.median(A):.3f}")
    print(f"  hatch matcher  IoU  mean={B.mean():.3f} median={np.median(B):.3f}")
    print(f"  selections AUC>0.75: {(A > 0.75).mean()*100:.0f}%   "
          f"AUC>0.90: {(A > 0.90).mean()*100:.0f}%")
    print("\nbaselines on the same split/grid: Gabor bank AUC 0.735 / IoU 0.307 "
          "(native, multi-scale); naive NCC AUC 0.433.")
    print("ceiling for perfect region identification: IoU 0.9485.")

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
