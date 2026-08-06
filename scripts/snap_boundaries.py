#!/usr/bin/env python3
"""Post-hoc boundary refinement for RefUNet predictions.

The ceiling analysis says a perfect model that misplaces boundaries by 3px still
only reaches ~0.9485 mean union IoU, so boundary precision is where a high target
is won or lost. Predictions are made at `--image-max-size` and upsampled with
nearest-neighbour to native resolution, which on a 3506px plan turns every
predicted pixel into ~2.7 native pixels of quantisation before the model errs at
all.

Variants (all post-hoc, no retraining):

  raw       threshold the upsampled probability map (current behaviour)
  bilinear  upsample probabilities bilinearly instead of the mask by nearest
  guided    edge-aware guided filter (He et al.) of the probability map with the
            grayscale plan as guide, so the mask boundary snaps to drawn linework
  morph     morphological close-then-open to remove speckle and fill pinholes
  cellfill  flood regions bounded by ink: connected components of the non-ink
            area are accepted whole when their mean probability clears a cutoff

Decide variants on the validation indices, never on HF14.

Usage:
    PYTHONPATH=. python3 scripts/snap_boundaries.py \
        --checkpoint data/runs/ck_fix_mix3652_seed31/epoch_7.pth \
        --indices 4,5,6,8,9,10,13,15,17,19,20,21,22,26
"""

import argparse
import io
import json
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

from refmask2former import load_parquet_records
from refmask2former.dataset import _normalize_chw, render_instance_mask, sample_reference_box
from scripts.evaluate_refunet_selection import load_refunet

VALIDATION = "4,5,6,8,9,10,13,15,17,19,20,21,22,26"


def guided_filter(guide, src, radius, eps):
    """He et al. guided filter; edge-aware smoothing of `src` guided by `guide`."""
    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    k = (radius, radius)
    mean_g = cv2.blur(g, k)
    mean_p = cv2.blur(p, k)
    cov = cv2.blur(g * p, k) - mean_g * mean_p
    var = cv2.blur(g * g, k) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_p - a * mean_g
    return cv2.blur(a, k) * g + cv2.blur(b, k)


def cell_fill(prob, gray, cutoff, ink_thresh=0.6, min_cell=64):
    """Accept whole ink-bounded cells whose mean probability clears `cutoff`."""
    ink = (gray < ink_thresh).astype(np.uint8)
    free = (1 - ink).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    out = np.zeros(prob.shape, bool)
    if n <= 1:
        return out
    sums = np.bincount(labels.ravel(), weights=prob.ravel(), minlength=n)
    counts = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    means = sums / np.maximum(counts, 1)
    keep = np.where((means >= cutoff) & (counts >= min_cell))[0]
    keep = keep[keep != 0]
    if keep.size:
        out = np.isin(labels, keep)
    # ink inside an accepted cell belongs to the region too: close the hairlines
    if out.any():
        r = max(3, int(round(min(prob.shape) / 400)) | 1)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((r, r), np.uint8)).astype(bool)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--indices", default=VALIDATION)
    ap.add_argument("--image-max-size", type=int, default=1280)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_refunet(args.checkpoint, device)
    ds = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                              config="real-world-test", split="test")
    indices = [int(v) for v in args.indices.split(",") if v.strip()]

    scores = defaultdict(list)
    per_image = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for image_idx in indices:
            rec = ds[image_idx]
            image0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
            anns = rec["annotations"]
            if isinstance(anns, str):
                anns = json.loads(anns)
            h0, w0 = image0.shape[:2]
            masks0 = [render_instance_mask(a["segmentation"], h0, w0).astype(bool)
                      for a in anns]
            cats = [a.get("category_name", "pattern") for a in anns]
            gray = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

            s = args.image_max_size / max(h0, w0)
            nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
            small = cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)
            image_t = _normalize_chw(small).unsqueeze(0).to(device)

            for ref_idx, (m0, cat) in enumerate(zip(masks0, cats)):
                rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
                x, y, bw, bh = sample_reference_box(m0, 128, 512, rng=rng)
                crop = image0[y:y + bh, x:x + bw]
                if not crop.size:
                    continue
                crop = cv2.resize(crop, (args.ref_size, args.ref_size),
                                  interpolation=cv2.INTER_LINEAR)
                ref = _normalize_chw(crop).unsqueeze(0).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    prob_small = model(image_t, ref).sigmoid()[0, 0].float().cpu().numpy()

                target = np.logical_or.reduce(
                    [m for m, c in zip(masks0, cats) if c == cat])

                def iou(pred):
                    return (pred & target).sum() / max((pred | target).sum(), 1)

                # raw: nearest-neighbour mask upsample (current evaluator)
                m = cv2.resize((prob_small > args.mask_thresh).astype(np.uint8),
                               (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
                scores["raw"].append(iou(m)); per_image[image_idx]["raw"].append(iou(m))

                # bilinear: upsample probabilities, then threshold
                pb = cv2.resize(prob_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
                mb = pb > args.mask_thresh
                scores["bilinear"].append(iou(mb))
                per_image[image_idx]["bilinear"].append(iou(mb))

                # guided: edge-aware snap to linework
                r = max(4, int(round(max(h0, w0) / 250)))
                for eps in (1e-4, 1e-3):
                    gf = guided_filter(gray, pb, r, eps)
                    key = f"guided_r{r}_e{eps:g}".replace(f"_r{r}", "")
                    mg = gf > args.mask_thresh
                    scores[f"guided{key[6:]}"].append(iou(mg))
                    per_image[image_idx][f"guided{key[6:]}"].append(iou(mg))

                # morph: clean speckle, fill pinholes
                kk = max(3, int(round(max(h0, w0) / 300)) | 1)
                mm = cv2.morphologyEx(mb.astype(np.uint8), cv2.MORPH_CLOSE,
                                      np.ones((kk, kk), np.uint8))
                mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN,
                                      np.ones((kk, kk), np.uint8)).astype(bool)
                scores["morph"].append(iou(mm)); per_image[image_idx]["morph"].append(iou(mm))

                # cellfill: accept whole ink-bounded cells
                for cutoff in (0.35, 0.5):
                    mc = cell_fill(pb, gray, cutoff)
                    scores[f"cellfill{cutoff}"].append(iou(mc))
                    per_image[image_idx][f"cellfill{cutoff}"].append(iou(mc))

    print(f"{len(scores['raw'])} selections over {len(indices)} images\n")
    print(f"{'variant':22s} {'mean IoU':>9} {'median':>8} {'vs raw':>8}")
    base = np.mean(scores["raw"])
    for k in sorted(scores, key=lambda k: -np.mean(scores[k])):
        v = np.array(scores[k])
        print(f"{k:22s} {v.mean():9.4f} {np.median(v):8.4f} {v.mean()-base:+8.4f}")

    best = max(scores, key=lambda k: np.mean(scores[k]))
    print(f"\nper-image, raw vs best ({best}):")
    for i in sorted(per_image):
        a = np.mean(per_image[i]["raw"]); b = np.mean(per_image[i][best])
        print(f"  img {i:2d}: raw={a:.3f} {best}={b:.3f} ({b-a:+.3f})")


if __name__ == "__main__":
    main()
