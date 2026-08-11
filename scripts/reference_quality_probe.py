#!/usr/bin/env python3
"""Is the evaluation asking questions a user would never ask?

`sample_reference_box` guarantees the rectangle lies INSIDE the pattern mask, but
not that it shows the pattern. On HF14 image 7 it lands squarely on the words
"UNDER FLOOR OF THE EXISTING GARAGE" printed over a large stippled floor: the
crop is a picture of text, the model hunts for text, and the selection scores
0.12. A user choosing by hand would obviously pick one of the vast clean areas.

This measures how REPRESENTATIVE each reference crop is of the material it is
supposed to exemplify, without needing a text detector: embed the crop and many
random patches from the same instance with an ImageNet ResNet50, and report the
crop's cosine similarity to the region's own centroid. A crop dominated by text,
a dimension line or a fixture is an outlier against its region; a clean patch of
hatch is not.

    PYTHONPATH=. python3 scripts/reference_quality_probe.py --checkpoint <ck>
"""
import argparse
import io
import json
import random

import cv2
import numpy as np
import torch
from PIL import Image

from refmask2former import load_parquet_records
from refmask2former.dataset import (IMAGENET_MEAN, IMAGENET_STD, _normalize_chw,
                                    render_instance_mask, sample_reference_box)
from scripts.evaluate_refunet_selection import HOLDOUT, load_refunet

VALIDATION = "4,5,6,8,9,10,13,15,17,19,20,21,22,26"
PATCH = 96


def _encoder(device):
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    return m.eval().to(device)


def _embed(enc, crops, device):
    x = np.stack([cv2.resize(c, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
                  for c in crops]).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        return torch.nn.functional.normalize(enc(t), dim=1).cpu().numpy()


def _region_patches(img, mask, rng, n=24):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return []
    H, W = mask.shape
    out = []
    for _ in range(400):
        i = rng.integers(len(xs))
        x = int(np.clip(xs[i] - PATCH // 2, 0, max(0, W - PATCH)))
        y = int(np.clip(ys[i] - PATCH // 2, 0, max(0, H - PATCH)))
        if mask[y:y + PATCH, x:x + PATCH].mean() > 0.95:
            out.append(img[y:y + PATCH, x:x + PATCH])
            if len(out) >= n:
                break
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--indices", default=None, help="default: HF14 and validation")
    ap.add_argument("--image-max-size", type=int, default=1280)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--cache-dir", default="./data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_parquet_records(cache_dir=args.cache_dir,
                                   config="real-world-test", split="test")
    model, _ = load_refunet(args.checkpoint, device)
    enc = _encoder(device)
    rng = np.random.default_rng(0)

    sets = ({"HF14": [int(v) for v in HOLDOUT.split(",")],
             "val": [int(v) for v in VALIDATION.split(",")]}
            if args.indices is None else
            {"custom": [int(v) for v in args.indices.split(",")]})

    for name, idxs in sets.items():
        rows = []
        for image_idx in idxs:
            rec = records[image_idx]
            img0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
            anns = rec["annotations"]
            anns = json.loads(anns) if isinstance(anns, str) else anns
            h0, w0 = img0.shape[:2]
            masks = [render_instance_mask(a["segmentation"], h0, w0).astype(bool) for a in anns]
            cats = [a.get("category_name", "pattern") for a in anns]
            s = args.image_max_size / max(h0, w0)
            nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
            tensor = _normalize_chw(cv2.resize(img0, (nw, nh),
                                               interpolation=cv2.INTER_LINEAR)
                                    ).unsqueeze(0).to(device)
            for ref_idx, (rm, cat) in enumerate(zip(masks, cats)):
                r = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
                x, y, w, h = sample_reference_box(rm, 128, 512, rng=r)
                crop = img0[y:y + h, x:x + w]
                pool = _region_patches(img0, rm, rng)
                if not crop.size or len(pool) < 4:
                    continue
                emb = _embed(enc, [crop] + pool, device)
                centroid = emb[1:].mean(0)
                centroid /= np.linalg.norm(centroid) + 1e-9
                repr_sim = float(emb[0] @ centroid)
                peer = float(np.mean(emb[1:] @ centroid))     # what a clean patch scores

                ref = _normalize_chw(cv2.resize(crop, (args.ref_size, args.ref_size),
                                                interpolation=cv2.INTER_LINEAR)
                                     ).unsqueeze(0).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prob = model(tensor, ref).sigmoid()[0, 0]
                pred = cv2.resize((prob > args.mask_thresh).cpu().numpy().astype(np.uint8),
                                  (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
                tgt = np.logical_or.reduce([m for m, c in zip(masks, cats) if c == cat])
                iou = int((pred & tgt).sum()) / max(1, int((pred | tgt).sum()))
                rows.append((image_idx, ref_idx, repr_sim, peer, repr_sim - peer, iou))

        rs = np.array([r[2] for r in rows]); gap = np.array([r[4] for r in rows])
        io_ = np.array([r[5] for r in rows])
        print(f"\n=== {name}: {len(rows)} selections ===")
        print(f"  corr(IoU, crop representativeness) = {np.corrcoef(io_, rs)[0, 1]:+.3f}")
        print(f"  corr(IoU, crop-vs-peer gap)        = {np.corrcoef(io_, gap)[0, 1]:+.3f}")
        odd = sorted(rows, key=lambda r: r[4])[:6]
        print(f"  {'img':>4}{'ref':>5}{'repr':>8}{'peer':>7}{'gap':>8}{'IoU':>8}   least representative crops")
        for im, rf, r_, p_, g_, i_ in odd:
            print(f"  {im:>4}{rf:>5}{r_:>8.3f}{p_:>7.3f}{g_:>8.3f}{i_:>8.3f}")
        q = np.quantile(gap, 0.25)
        lo, hi = io_[gap <= q], io_[gap > q]
        print(f"  mean IoU | least-representative quartile {lo.mean():.3f}  vs rest {hi.mean():.3f}")


if __name__ == "__main__":
    main()
