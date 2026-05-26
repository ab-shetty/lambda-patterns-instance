#!/usr/bin/env python3
"""Overlay model detections vs GT on real-world plans to see WHAT it misses.

Per image: left = raw, middle = GT instances (green outlines), right = the
model's class-agnostic detections (filled, prob>score) with GT outlines on top.
A green outline with no fill underneath it = a missed instance.
"""
import argparse, io, json, random
from functools import partial
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import load_model
from refmask2former.dataset import (render_instance_mask, load_parquet_records,
                                     _normalize_chw, IMAGENET_MEAN, IMAGENET_STD)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-max-size", type=int, default=2048)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--indices", default="", help="comma-separated record indices")
    p.add_argument("--out", default="/tmp/overlay.png")
    return p.parse_args()


def load(rec):
    image = np.array(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
    anns = rec["annotations"]
    if isinstance(anns, str):
        anns = json.loads(anns)
    return image, anns


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch {ck.get('epoch','?')})")

    recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                config="real-world-test", split="test")
    # pick the images with the most instances (misses are most visible there)
    counts = [(i, len(load(recs[i])[1])) for i in range(len(recs))]
    counts.sort(key=lambda t: -t[1])
    if args.indices:
        pick = [int(x) for x in args.indices.split(",")]
    else:
        pick = [i for i, _ in counts[:args.n]]

    for row, idx in enumerate(pick):
        fig, axes = plt.subplots(1, 3, figsize=(3 * 9, 6))
        axes = axes[None]
        row = 0
        image, anns = load(recs[idx])
        h0, w0 = image.shape[:2]
        scale = args.image_max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
        img_r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # reference: crop from first instance (just to drive the head; detection
        # is reference-independent so any ref works for the class-agnostic view)
        m0 = render_instance_mask(anns[0]["segmentation"], h0, w0)
        ys, xs = np.where(m0 > 0)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        ref = cv2.resize(image[y0:y1+1, x0:x1+1], (224, 224))

        with torch.no_grad():
            im_t = _normalize_chw(img_r)[None].to(device)
            pm_t = torch.ones(1, nh, nw, dtype=torch.bool, device=device)
            ref_t = _normalize_chw(ref)[None].to(device)
            out = model(im_t, pm_t, ref_t)
            probs = out["pred_logits"].softmax(-1)[..., 1][0]
            masks = F.interpolate(out["pred_masks"], size=(nh, nw),
                                  mode="bilinear", align_corners=False).sigmoid()[0]
            keep = probs > args.score_thresh
            det = (masks[keep] > 0.5).cpu().numpy()

        # GT masks at resized scale
        gt = [cv2.resize(render_instance_mask(a["segmentation"], h0, w0), (nw, nh),
                         interpolation=cv2.INTER_NEAREST) for a in anns]

        base = img_r.copy()
        # middle: GT outlines
        midimg = base.copy()
        for m in gt:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(midimg, cnts, -1, (0, 200, 0), 3)
        # right: detections filled + GT outlines
        rt = base.copy()
        rng = random.Random(0)
        for m in det:
            color = np.array([rng.randint(60, 255) for _ in range(3)], np.uint8)
            rt[m] = (0.45 * rt[m] + 0.55 * color).astype(np.uint8)
        for m in gt:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rt, cnts, -1, (0, 200, 0), 3)

        axes[row, 0].imshow(base); axes[row, 0].set_title(f"img {idx} ({w0}x{h0})", fontsize=8)
        axes[row, 1].imshow(midimg); axes[row, 1].set_title(f"GT: {len(gt)} instances", fontsize=8)
        axes[row, 2].imshow(rt); axes[row, 2].set_title(f"detections: {int(keep.sum())} (fill) vs GT (green)", fontsize=8)
        for c in range(3):
            axes[row, c].axis("off")
        plt.tight_layout()
        fn = args.out.replace(".png", f"_{idx}.png")
        plt.savefig(fn, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("saved", fn)


if __name__ == "__main__":
    main()
