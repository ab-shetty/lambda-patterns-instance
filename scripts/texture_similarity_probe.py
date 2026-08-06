#!/usr/bin/env python3
"""Does an engineered texture-similarity map carry the region-identification signal?

Region identification, not boundary placement, is what limits the model: boundary
refinement is worth ~0 on validation while per-image IoU ranges 0.42-0.98.

Naive normalised cross-correlation of the reference patch scored AUC 0.433
(below chance) because CAD hatch is periodic: NCC fires only where the phase
aligns, giving a sparse lattice instead of a filled region, and it responds to
any dense linework regardless of pattern identity.

A hatch pattern is defined by line ORIENTATION and SPACING, so the natural
descriptor is local Gabor energy: phase-invariant, and directly encoding
"45-degree lines about 8px apart". This probe builds a per-pixel descriptor,
pools it locally, and cosine-matches it against the reference patch's descriptor,
then measures how well the resulting map separates target from non-target.

AUC well above 0.5 means the signal exists and is worth feeding to the network as
an extra input channel; near 0.5 means it does not.
"""

import argparse
import io
import json
import random

import cv2
import numpy as np
from PIL import Image

from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask, sample_reference_box

VALIDATION = "4,5,6,8,9,10,13,15,17,19,20,21,22,26"


def gabor_bank(n_theta=6, wavelengths=(4, 8, 16, 32), ksize=31):
    bank = []
    for w in wavelengths:
        for t in range(n_theta):
            theta = np.pi * t / n_theta
            k = cv2.getGaborKernel((ksize, ksize), sigma=0.56 * w, theta=theta,
                                   lambd=w, gamma=0.5, psi=0)
            k -= k.mean()
            bank.append(k)
    return bank


def texture_features(gray, bank, pool):
    """Per-pixel L2-normalised local Gabor-energy descriptor."""
    feats = []
    for k in bank:
        r = cv2.filter2D(gray, cv2.CV_32F, k)
        feats.append(cv2.blur(np.abs(r), (pool, pool)))
    f = np.stack(feats, -1)
    n = np.linalg.norm(f, axis=-1, keepdims=True)
    return f / np.maximum(n, 1e-6)


def auc(inside, outside, n=4000, rng=None):
    rng = rng or np.random
    a = rng.choice(inside, min(n, inside.size), replace=False)
    b = rng.choice(outside, min(n, outside.size), replace=False)
    return float((a[:, None] > b[None, :]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indices", default=VALIDATION)
    ap.add_argument("--image-max-size", type=int, default=1280)
    ap.add_argument("--pool", type=int, default=33, help="local pooling window")
    args = ap.parse_args()

    ds = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                              config="real-world-test", split="test")
    idxs = [int(v) for v in args.indices.split(",") if v.strip()]
    bank = gabor_bank()
    rng = np.random.default_rng(0)

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

        s = args.image_max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
        gray = cv2.cvtColor(cv2.resize(im0, (nw, nh)), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        feat = texture_features(gray, bank, args.pool)

        for ref_idx, (m0, cat) in enumerate(zip(masks0, cats)):
            r = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
            x, y, bw, bh = sample_reference_box(m0, 128, 512, rng=r)
            rx, ry = max(0, round(x * s)), max(0, round(y * s))
            rw, rh = max(4, round(bw * s)), max(4, round(bh * s))
            rx, ry = min(rx, nw - 2), min(ry, nh - 2)
            rw, rh = min(rw, nw - rx), min(rh, nh - ry)
            patch = feat[ry:ry + rh, rx:rx + rw].reshape(-1, feat.shape[-1])
            if patch.shape[0] < 4:
                continue
            desc = patch.mean(0)
            desc /= max(np.linalg.norm(desc), 1e-6)
            sim = feat @ desc                                    # cosine map

            tgt = np.logical_or.reduce([m for m, c in zip(masks0, cats) if c == cat])
            tgt = cv2.resize(tgt.astype(np.uint8), (nw, nh),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
            if tgt.sum() < 10 or (~tgt).sum() < 10:
                continue
            a = auc(sim[tgt], sim[~tgt], rng=rng)
            best = max((((sim > t) & tgt).sum() / max(((sim > t) | tgt).sum(), 1))
                       for t in np.arange(0.5, 1.0, 0.02))
            rows.append((a, best, image_idx))

    A = np.array([r[0] for r in rows])
    B = np.array([r[1] for r in rows])
    print(f"n={len(rows)} selections, pool={args.pool}")
    print(f"Gabor-texture similarity AUC:      mean={A.mean():.3f} median={np.median(A):.3f}")
    print(f"  (naive NCC on HF14 was 0.433 -- below chance)")
    print(f"Best-threshold IoU, texture alone: mean={B.mean():.3f} median={np.median(B):.3f}")
    print(f"  selections with AUC > 0.75: {(A > 0.75).mean()*100:.0f}%")
    print(f"  selections with AUC > 0.90: {(A > 0.90).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
