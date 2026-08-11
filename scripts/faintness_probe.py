#!/usr/bin/env python3
"""Is faintness what breaks the worst real plans, and does contrast fix it?

Reports per-image contrast statistics next to per-image IoU, then optionally
re-evaluates with a contrast normalisation applied at inference:

    none     as-is
    stretch  per-image percentile stretch, p2..p98 -> 0..255
    clahe    CLAHE on the luminance channel

Contrast normalisation is an INFERENCE KNOB. `startup.md` forbids sweeping those
against HF14, so this defaults to the validation split; confirm on HF14 once,
after deciding.

    PYTHONPATH=. python3 scripts/faintness_probe.py --checkpoint <ck> --mode none
    PYTHONPATH=. python3 scripts/faintness_probe.py --checkpoint <ck> --mode stretch
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
from scripts.evaluate_refunet_selection import HOLDOUT, load_refunet

VALIDATION = "4,5,6,8,9,10,13,15,17,19,20,21,22,26"


def contrast_stats(img):
    """Faintness descriptors for a plan: how dark is the ink, how much spread."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    p2, p50, p98 = np.percentile(g, [2, 50, 98])
    ink = g < (p50 - 12)                       # pixels meaningfully darker than paper
    return {
        "p2": float(p2), "median": float(p50), "p98": float(p98),
        "range": float(p98 - p2),              # low = washed out
        "std": float(g.std()),
        "ink_frac": float(ink.mean()),
        "ink_depth": float(p50 - g[ink].mean()) if ink.any() else 0.0,
    }


def enhance(img, mode):
    if mode == "none":
        return img
    if mode == "stretch":
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lo, hi = np.percentile(g, [2, 98])
        if hi - lo < 1:
            return img
        out = (img.astype(np.float32) - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)
    if mode == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    raise ValueError(mode)


@torch.no_grad()
def run(model, records, indices, mode, size, ref_size, thresh, device):
    per_image = defaultdict(list)
    stats = {}
    for image_idx in indices:
        rec = records[image_idx]
        img0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = rec["annotations"]
        anns = json.loads(anns) if isinstance(anns, str) else anns
        h0, w0 = img0.shape[:2]
        stats[image_idx] = contrast_stats(img0)

        img_e = enhance(img0, mode)
        masks = [render_instance_mask(a["segmentation"], h0, w0).astype(bool) for a in anns]
        cats = [a.get("category_name", "pattern") for a in anns]
        s = size / max(h0, w0)
        nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
        tensor = _normalize_chw(cv2.resize(img_e, (nw, nh),
                                           interpolation=cv2.INTER_LINEAR)
                                ).unsqueeze(0).to(device)
        for ref_idx, (rm, cat) in enumerate(zip(masks, cats)):
            rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
            x, y, w, h = sample_reference_box(rm, 128, 512, rng=rng)
            crop = img_e[y:y + h, x:x + w]
            if not crop.size:
                continue
            ref = _normalize_chw(cv2.resize(crop, (ref_size, ref_size),
                                            interpolation=cv2.INTER_LINEAR)
                                 ).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prob = model(tensor, ref).sigmoid()[0, 0]
            pred = cv2.resize((prob > thresh).cpu().numpy().astype(np.uint8),
                              (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
            tgt = np.logical_or.reduce([m for m, c in zip(masks, cats) if c == cat])
            per_image[image_idx].append(int((pred & tgt).sum()) / max(1, int((pred | tgt).sum())))
    return per_image, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", nargs="+", default=["none", "stretch", "clahe"])
    ap.add_argument("--indices", default=VALIDATION)
    ap.add_argument("--hf14", action="store_true", help="use the acceptance set instead")
    ap.add_argument("--image-max-size", type=int, default=1280)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--cache-dir", default="./data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_parquet_records(cache_dir=args.cache_dir,
                                   config="real-world-test", split="test")
    idxs = [int(v) for v in (HOLDOUT if args.hf14 else args.indices).split(",") if v.strip()]
    model, _ = load_refunet(args.checkpoint, device)

    results = {}
    for mode in args.mode:
        per_image, stats = run(model, records, idxs, mode, args.image_max_size,
                               args.ref_size, args.mask_thresh, device)
        results[mode] = per_image
        allv = [v for vs in per_image.values() for v in vs]
        print(f"mode={mode:<8} mean IoU {np.mean(allv):.4f}  ({len(allv)} selections)")

    base = results[args.mode[0]]
    print(f"\n{'img':>4}{'sel':>5}{'range':>8}{'std':>7}{'ink_frac':>10}{'ink_depth':>11}"
          + "".join(f"{m:>10}" for m in args.mode))
    order = sorted(base, key=lambda i: np.mean(base[i]))
    for i in order:
        st = stats[i]
        row = (f"{i:>4}{len(base[i]):>5}{st['range']:>8.0f}{st['std']:>7.1f}"
               f"{st['ink_frac']:>10.3f}{st['ink_depth']:>11.1f}")
        row += "".join(f"{np.mean(results[m][i]):>10.3f}" for m in args.mode)
        print(row)

    ious = np.array([np.mean(base[i]) for i in order])
    for key in ("range", "std", "ink_frac", "ink_depth"):
        v = np.array([stats[i][key] for i in order])
        print(f"corr(IoU, {key:<10}) = {np.corrcoef(ious, v)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
