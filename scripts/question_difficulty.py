#!/usr/bin/env python3
"""Is the synthetic -> HF14 gap domain transfer, or harder questions?

Every source measured so far (v6d, v7, Gemini; 148 to 100k plans; RefUNet and
swin_t) scores HF14 at ~0.85 of its own unseen-plan IoU. A gap that ignores the
training source may simply be HF14 asking harder questions. This scores unseen
synthetic plans with EXACTLY the HF14 protocol (`evaluate_model`: every
labelled instance is a question, same box sampler, resize and threshold), gives
every question the same source-agnostic difficulty features, and then asks:

  1. within a difficulty bucket, do HF14 and synthetic questions score alike?
  2. regressing IoU on the features plus an is_real flag, what is left for
     is_real? That coefficient is the domain gap net of question difficulty.

Features (native resolution, computed identically for both sources):
  n_fams       distinct pattern families on the sheet
  n_regions    instances in the target family
  target_frac  target union area / sheet area
  lookalike    max cosine similarity between the target family's texture
               descriptor and any OTHER family's on the same sheet (0 if
               single-family); descriptor = grey-level + gradient-orientation
               histograms over the family's union

    PYTHONPATH=. python3 scripts/question_difficulty.py \\
        --checkpoint data/runs/ck_tr_v6d_res1024_seed7/epoch_8.pth \\
        --pool data/synthetic/v6d_fresh600 --n-images 150 --image-max-size 1024 \\
        --out data/evaluations/qd_v6d_1024.json
"""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from refmask2former import load_parquet_records                      # noqa: E402
from refmask2former.dataset import render_instance_mask             # noqa: E402
from scripts.evaluate_refunet_selection import HOLDOUT, evaluate_model, load_refunet  # noqa: E402


def local_records(pool, n):
    anns = sorted(os.listdir(f"{pool}/annotations"))[:n]
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(f"{pool}/images")}
    out = []
    for a in anns:
        A = json.load(open(f"{pool}/annotations/{a}"))
        stem = os.path.splitext(a)[0]
        with open(f"{pool}/images/{imgs[stem]}", "rb") as f:
            out.append({"image": f.read(), "annotations": A["annotations"]})
    return out


def descriptor(gray, gx, gy, mask):
    if mask.sum() < 50:
        return None
    g = gray[mask]
    h1 = np.histogram(g, bins=16, range=(0, 256))[0].astype(np.float64)
    mag = np.hypot(gx[mask], gy[mask])
    ang = np.mod(np.arctan2(gy[mask], gx[mask]), np.pi)
    h2 = np.histogram(ang, bins=12, range=(0, np.pi), weights=mag)[0]
    h3 = np.histogram(mag, bins=8, range=(0, 400))[0].astype(np.float64)
    v = np.concatenate([h1 / max(h1.sum(), 1), h2 / max(h2.sum(), 1e-9), h3 / max(h3.sum(), 1)])
    return v / max(np.linalg.norm(v), 1e-9)


def features(rec, max_side=1024):
    """Per-family features for one sheet, keyed by category."""
    img = cv2.imdecode(np.frombuffer(rec["image"], np.uint8), cv2.IMREAD_COLOR)
    anns = rec["annotations"]
    if isinstance(anns, str):
        anns = json.loads(anns)
    h0, w0 = img.shape[:2]
    cats = [a.get("category_name", "pattern") for a in anns]
    masks = [render_instance_mask(a["segmentation"], h0, w0).astype(bool) for a in anns]
    # descriptors at a common working size so line spacing is comparable
    s = max_side / max(h0, w0)
    small = cv2.resize(img, (max(1, round(w0 * s)), max(1, round(h0 * s))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    fam = {}
    for c in sorted(set(cats)):
        u = np.logical_or.reduce([m for m, cc in zip(masks, cats) if cc == c])
        us = cv2.resize(u.astype(np.uint8), (small.shape[1], small.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        fam[c] = {"n_regions": sum(cc == c for cc in cats),
                  "target_frac": float(u.sum()) / (h0 * w0),
                  "desc": descriptor(gray, gx, gy, us)}
    for c, f in fam.items():
        sims = [float(f["desc"] @ o["desc"]) for oc, o in fam.items()
                if oc != c and f["desc"] is not None and o["desc"] is not None]
        f["lookalike"] = max(sims) if sims else 0.0
    return fam, len(fam)


def score(model, recs, indices, size, thresh, is_real, src):
    rows = evaluate_model(model, recs, indices, image_max_size=size, ref_size=224,
                          mask_thresh=thresh, device=torch.device("cuda"))
    feats = {i: features(recs[i]) for i in set(r["image_index"] for r in rows)}
    out = []
    for r in rows:
        fam, nf = feats[r["image_index"]]
        f = fam[r["category"]]
        out.append({"src": src, "is_real": is_real, "image": r["image_index"], "iou": r["iou"],
                    "n_fams": nf, "n_regions": f["n_regions"], "target_frac": f["target_frac"],
                    "lookalike": f["lookalike"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pool", required=True, help="unseen synthetic/generated plans (local-data)")
    ap.add_argument("--n-images", type=int, default=150)
    ap.add_argument("--image-max-size", type=int, default=1024)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, _ = load_refunet(a.checkpoint, torch.device("cuda"))
    syn = local_records(a.pool, a.n_images)
    rows = score(model, syn, list(range(len(syn))), a.image_max_size, a.mask_thresh, 0, "synthetic")
    hf = load_parquet_records("abshetty/floz-synth-v5", cache_dir=a.cache_dir,
                              config="real-world-test", split="test")
    idx = [int(i) for i in HOLDOUT.split(",")]
    rows += score(model, hf, idx, a.image_max_size, a.mask_thresh, 1, "hf14")
    json.dump({"checkpoint": a.checkpoint, "pool": a.pool, "rows": rows}, open(a.out, "w"))
    print(f"wrote {a.out}: {sum(r['is_real']==0 for r in rows)} synthetic + "
          f"{sum(r['is_real'] for r in rows)} HF14 questions")


if __name__ == "__main__":
    main()
