#!/usr/bin/env python3
"""Evaluate the actual Floz interaction: rectangle -> matching region union.

Every labelled instance in each heldout image is used once as a deterministic
reference rectangle.  The target is the union of all annotations sharing that
instance's category.  The primary metric is binary union IoU between the model's
selected pixels and that target, so misses and excess/wrong-pattern pixels are
both penalized.
"""

import argparse
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from evaluate import load_model
from refmask2former import load_parquet_records
from refmask2former.dataset import (
    _normalize_chw,
    render_instance_mask,
    sample_reference_box,
)


HOLDOUT = "12,16,27,7,11,25,23,1,18,2,0,3,14,24"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--indices", default=HOLDOUT)
    p.add_argument("--image-max-size", type=int, default=1024)
    p.add_argument("--ref-size", type=int, default=224)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--match-thresh", type=float, default=0.0)
    return p.parse_args()


def title_panel(image, title):
    band = 48
    out = Image.new("RGB", (image.shape[1], image.shape[0] + band), "white")
    out.paste(Image.fromarray(image), (0, band))
    ImageDraw.Draw(out).text((12, 15), title, fill="black")
    return np.asarray(out)


def tint(image, mask, color, alpha=0.42):
    out = image.copy()
    color = np.asarray(color, dtype=np.uint8)
    out[mask] = ((1 - alpha) * out[mask] + alpha * color).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, tuple(int(x) for x in color), 3)
    return out


def audit_view(image, target, prediction):
    out = image.copy()
    tp = target & prediction
    fp = ~target & prediction
    fn = target & ~prediction
    for mask, color in ((tp, (40, 190, 80)), (fp, (230, 50, 50)),
                        (fn, (50, 100, 230))):
        out[mask] = (0.45 * out[mask] + 0.55 * np.asarray(color)).astype(np.uint8)
    return out


def main():
    args = parse_args()
    device = torch.device("cuda")
    model, checkpoint = load_model(args.checkpoint, device)
    records = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                   config="real-world-test", split="test")
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    out_root = Path(args.out) if args.out else None
    if out_root:
        (out_root / "comparisons").mkdir(parents=True, exist_ok=True)
        (out_root / "reference_selections").mkdir(parents=True, exist_ok=True)
        (out_root / "target_regions").mkdir(parents=True, exist_ok=True)
        (out_root / "predictions").mkdir(parents=True, exist_ok=True)

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
            masks = [cv2.resize(m.astype(np.uint8), (nw, nh),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                     for m in masks0]
            image_tensor = _normalize_chw(image).unsqueeze(0).to(device)
            pixel_mask = torch.ones((1, nh, nw), dtype=torch.bool, device=device)

            for ref_idx, (ref_mask0, category) in enumerate(zip(masks0, categories)):
                rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
                x, y, w, h = sample_reference_box(ref_mask0, 128, 512, rng=rng)
                ref_crop = image0[y:y + h, x:x + w]
                if not ref_crop.size:
                    continue
                ref_crop = cv2.resize(ref_crop, (args.ref_size, args.ref_size),
                                      interpolation=cv2.INTER_LINEAR)
                reference = _normalize_chw(ref_crop).unsqueeze(0).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(image_tensor, pixel_mask, reference)
                scores = outputs["pred_logits"].softmax(-1)[0, :, 1]
                similarities = torch.einsum(
                    "qd,d->q", outputs["pred_ref"][0], outputs["reference_emb"][0])
                pred_masks = (F.interpolate(outputs["pred_masks"], size=(nh, nw),
                                             mode="bilinear", align_corners=False)
                              .sigmoid()[0] > 0.5)
                keep = (scores > args.score_thresh) & (similarities > args.match_thresh)
                prediction = (pred_masks[keep].any(0).cpu().numpy() if keep.any()
                              else np.zeros((nh, nw), dtype=bool))
                target = np.logical_or.reduce(
                    [m for m, c in zip(masks, categories) if c == category])
                intersection = int((prediction & target).sum())
                union = int((prediction | target).sum())
                iou = intersection / max(union, 1)
                row = {
                    "image_index": image_idx, "reference_instance": ref_idx,
                    "category": category,
                    "reference_box_native": [int(x), int(y), int(w), int(h)],
                    "iou": iou, "prediction_pixels": int(prediction.sum()),
                    "target_pixels": int(target.sum()), "kept_queries": int(keep.sum()),
                }
                rows.append(row)

                if out_root:
                    stem = f"hf_{image_idx:02d}_ref_{ref_idx:02d}_{category}"
                    selected = image.copy()
                    p0 = (round(x * scale), round(y * scale))
                    p1 = (round((x + w) * scale), round((y + h) * scale))
                    cv2.rectangle(selected, p0, p1, (245, 180, 30), 5)
                    target_view = tint(image, target, (40, 190, 80))
                    pred_view = tint(image, prediction, (190, 50, 210))
                    audit = audit_view(image, target, prediction)
                    selected_p = title_panel(selected, f"User rectangle: {category}")
                    target_p = title_panel(target_view, "Expected matching regions")
                    pred_p = title_panel(pred_view, f"Predicted selection ({int(keep.sum())} queries)")
                    audit_p = title_panel(audit, f"IoU={iou:.3f}; green=overlap red=extra blue=miss")
                    Image.fromarray(selected_p).save(
                        out_root / "reference_selections" / f"{stem}.png")
                    Image.fromarray(target_p).save(
                        out_root / "target_regions" / f"{stem}.png")
                    Image.fromarray(pred_p).save(
                        out_root / "predictions" / f"{stem}.png")
                    Image.fromarray(np.concatenate(
                        [selected_p, target_p, pred_p, audit_p], axis=1)).save(
                        out_root / "comparisons" / f"{stem}.png")

    mean_iou = sum(r["iou"] for r in rows) / max(len(rows), 1)
    by_image = {}
    for row in rows:
        by_image.setdefault(str(row["image_index"]), []).append(row["iou"])
    result = {
        "metric": "reference-conditioned union IoU",
        "definition": "IoU(union predicted pixels, union GT pixels with reference category)",
        "checkpoint": args.checkpoint, "checkpoint_epoch": checkpoint.get("epoch"),
        "image_max_size": args.image_max_size,
        "score_thresh": args.score_thresh, "match_thresh": args.match_thresh,
        "n_images": len(indices), "n_reference_selections": len(rows),
        "mean_iou": mean_iou,
        "image_mean_iou": {k: sum(v) / len(v) for k, v in by_image.items()},
        "selections": rows,
    }
    print(json.dumps(result, indent=2))
    metrics_path = (Path(args.metrics_out) if args.metrics_out else
                    (out_root / "metrics.json" if out_root else None))
    if metrics_path:
        metrics_path.write_text(json.dumps(result, indent=2))
    if out_root:
        (out_root / "README.md").write_text(
            "# Reference-conditioned heldout evaluation\n\n"
            "Each file represents one possible user-selected reference rectangle.\n\n"
            "- `reference_selections/`: the input rectangle in yellow.\n"
            "- `target_regions/`: all GT regions matching its pattern label.\n"
            "- `predictions/`: only regions selected from that reference.\n"
            "- `comparisons/`: input, target, prediction, and pixel audit.\n"
            "- `metrics.json`: reference-conditioned union-IoU results.\n")


if __name__ == "__main__":
    main()
