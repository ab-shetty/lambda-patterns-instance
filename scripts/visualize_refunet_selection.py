#!/usr/bin/env python3
"""Render the required heldout visual audit for a RefUNet checkpoint.

`PROJECT_UNDERSTANDING.md` requires that every evaluated reference selection be
shown as four views: the input with the user rectangle, the expected union of
matching image-local regions, the model's reference-conditioned predicted union,
and an error view separating overlap, excess selection, and missed pixels.

The reference boxes, resize, threshold, and union logic are taken from
`evaluate_refunet_selection.evaluate_model` and the per-selection RNG seed is
reproduced exactly, so the IoU printed on each panel is the same number that
feeds the reported mean. Files are named by IoU so the worst cases sort first.

Usage:
    PYTHONPATH=. python scripts/visualize_refunet_selection.py \
        --checkpoint data/runs/ck_refunet_mix3652_s31/epoch_5.pth \
        --mask-thresh 0.35 --out data/visualizations/mix3652_e5
"""

import argparse
import csv
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from refmask2former import load_parquet_records
from refmask2former.dataset import _normalize_chw, render_instance_mask, sample_reference_box
from scripts.evaluate_refunet_selection import HOLDOUT, load_refunet

PANEL_W = 760
OVERLAP = (34, 197, 94)    # green  - correctly selected
EXCESS = (239, 68, 68)     # red    - selected but should not be
MISSED = (59, 130, 246)    # blue   - should be selected but was not
BOX = (234, 88, 12)        # orange - the user's reference rectangle


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _tint(image, mask, color, alpha=0.55):
    """Blend `color` into `image` where `mask` is set."""
    out = image.astype(np.float32).copy()
    layer = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * layer
    return out.astype(np.uint8)


def _dim(image, factor=0.45):
    white = np.full_like(image, 255, dtype=np.float32)
    return ((1 - factor) * white + factor * image.astype(np.float32)).astype(np.uint8)


def _fit(image, width):
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(image, (width, max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def render_selection(image0, box, target, prediction, iou, meta, out_path):
    x, y, w, h = box
    panels = []

    # 1. input + the user's reference rectangle
    a = image0.copy()
    thickness = max(2, round(max(image0.shape[:2]) / 300))
    cv2.rectangle(a, (x, y), (x + w, y + h), BOX, thickness)
    panels.append(("1. user reference selection", a))

    # 2. expected union of same-pattern regions
    panels.append(("2. expected union (ground truth)",
                   _tint(_dim(image0), target, OVERLAP)))

    # 3. what the model selected
    panels.append(("3. model predicted union",
                   _tint(_dim(image0), prediction, MISSED)))

    # 4. error view
    err = _dim(image0)
    err = _tint(err, prediction & target, OVERLAP)
    err = _tint(err, prediction & ~target, EXCESS)
    err = _tint(err, ~prediction & target, MISSED)
    panels.append(("4. overlap / excess / missed", err))

    fitted = [(label, _fit(img, PANEL_W)) for label, img in panels]
    cell_h = max(img.shape[0] for _, img in fitted)
    pad, label_h, title_h = 18, 30, 44
    W = pad * (len(fitted) + 1) + PANEL_W * len(fitted)
    H = pad * 2 + title_h + label_h + cell_h

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), meta, fill=(17, 17, 17), font=_font(22))

    for i, (label, img) in enumerate(fitted):
        x0 = pad + i * (PANEL_W + pad)
        draw.text((x0, pad + title_h - 4), label, fill=(90, 90, 90), font=_font(16))
        canvas.paste(Image.fromarray(img), (x0, pad + title_h + label_h))
        draw.rectangle([x0, pad + title_h + label_h,
                        x0 + PANEL_W, pad + title_h + label_h + img.shape[0]],
                       outline=(200, 200, 200), width=1)

    legend = ("green = correctly selected    red = excess selection    "
              "blue = missed    orange = user rectangle")
    draw.text((pad, H - pad - 2), legend, fill=(120, 120, 120), font=_font(15),
              anchor="ls")
    canvas.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--indices", default=HOLDOUT)
    parser.add_argument("--image-max-size", type=int, default=1280)
    parser.add_argument("--ref-size", type=int, default=224)
    parser.add_argument("--mask-thresh", type=float, default=0.35)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_refunet(args.checkpoint, device)
    records = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                   config="real-world-test", split="test")
    indices = [int(v) for v in args.indices.split(",") if v.strip()]

    rows = []
    with torch.inference_mode():
        for image_idx in indices:
            rec = records[image_idx]
            image0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
            anns = rec["annotations"]
            if isinstance(anns, str):
                anns = json.loads(anns)
            h0, w0 = image0.shape[:2]
            masks0 = [render_instance_mask(a["segmentation"], h0, w0).astype(bool)
                      for a in anns]
            categories = [a.get("category_name", "pattern") for a in anns]
            scale = args.image_max_size / max(h0, w0)
            nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
            image = cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)
            image_tensor = _normalize_chw(image).unsqueeze(0).to(device)

            for ref_idx, (ref_mask, category) in enumerate(zip(masks0, categories)):
                rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
                x, y, w, h = sample_reference_box(ref_mask, 128, 512, rng=rng)
                crop = image0[y:y + h, x:x + w]
                if not crop.size:
                    continue
                crop = cv2.resize(crop, (args.ref_size, args.ref_size),
                                  interpolation=cv2.INTER_LINEAR)
                reference = _normalize_chw(crop).unsqueeze(0).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    probability = model(image_tensor, reference).sigmoid()[0, 0]
                pred_small = probability > args.mask_thresh
                prediction = cv2.resize(pred_small.cpu().numpy().astype(np.uint8),
                                        (w0, h0),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                target = np.logical_or.reduce(
                    [m for m, c in zip(masks0, categories) if c == category])
                inter = int((prediction & target).sum())
                union = int((prediction | target).sum())
                iou = inter / max(union, 1)

                name = f"iou{iou:.3f}_img{image_idx:02d}_ref{ref_idx:02d}.png"
                meta = (f"image {image_idx}  |  reference instance {ref_idx}  "
                        f"|  pattern {category}  |  IoU {iou:.4f}")
                render_selection(image0, (x, y, w, h), target, prediction,
                                 iou, meta, out_dir / name)
                rows.append({"file": name, "image_index": image_idx,
                             "reference_instance": ref_idx, "category": category,
                             "iou": round(iou, 6),
                             "prediction_pixels": int(prediction.sum()),
                             "target_pixels": int(target.sum()),
                             "excess_pixels": int((prediction & ~target).sum()),
                             "missed_pixels": int((~prediction & target).sum())})

    rows.sort(key=lambda r: r["iou"])
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ious = np.array([r["iou"] for r in rows])
    summary = (
        f"checkpoint: {args.checkpoint}\n"
        f"epoch: {checkpoint.get('actual_epoch', checkpoint.get('epoch'))}\n"
        f"mask_thresh: {args.mask_thresh}   image_max_size: {args.image_max_size}\n"
        f"images: {len(indices)}   reference selections: {len(rows)}\n"
        f"mean IoU: {ious.mean():.10f}\n"
        f"median: {np.median(ious):.4f}   min: {ious.min():.4f}   max: {ious.max():.4f}\n"
        f"selections below 0.25: {int((ious < 0.25).sum())}\n"
        f"selections above 0.75: {int((ious > 0.75).sum())}\n")
    (out_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"wrote {len(rows)} panels to {out_dir}")


if __name__ == "__main__":
    main()
