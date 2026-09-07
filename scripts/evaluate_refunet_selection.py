#!/usr/bin/env python3
"""Evaluate a RefUNet checkpoint on the fixed reference-selection task."""

import argparse
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from refmask2former import load_parquet_records
from refmask2former.dataset import (_normalize_chw, render_instance_mask,
                                    sample_reference_box, scale_matched_reference)
from refmask2former.ref_unet import RefUNet

HOLDOUT = "12,16,27,7,11,25,23,1,18,2,0,3,14,24"


def _dihedral(t, hflip, k):
    """Apply hflip then k quarter-turns to the last two dims."""
    if hflip:
        t = t.flip(-1)
    if k:
        t = torch.rot90(t, k, dims=(-2, -1))
    return t


def _dihedral_inv(t, hflip, k):
    if k:
        t = torch.rot90(t, -k, dims=(-2, -1))
    if hflip:
        t = t.flip(-1)
    return t


def tta_transforms(n):
    """1: identity; 2: +hflip; 4: hflip x {0, 180}; 8: the full dihedral group.
    Training applies the same group online (dataset.py), so every member is a
    view the model was trained on."""
    if n <= 1:
        return [(False, 0)]
    if n == 2:
        return [(False, 0), (True, 0)]
    if n == 4:
        return [(False, 0), (True, 0), (False, 2), (True, 2)]
    return [(hf, k) for hf in (False, True) for k in range(4)]


def load_refunet(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    # `anchor` must be rebuilt from the saved args: it widens the stem to 4
    # channels, so an anchored checkpoint cannot load into a default model.
    # Training's per-epoch diagnostic passes the live model and never hit this.
    model = RefUNet(width=int(args.get("width", 128)), pretrained=False,
                    corr_grid=int(args.get("corr_grid", 0) or 0),
                    anchor=bool(args.get("anchor", False)),
                    anchor_ref_plane=float(args.get("anchor_ref_plane", 1.0)),
                    self_support=float(args.get("self_support", 0.0) or 0.0),
                    self_support_thresh=float(args.get("self_support_thresh", 0.7)),
                    dynamic_filter=bool(args.get("dynamic_filter", False))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def evaluate_model(model, records, indices, image_max_size=1280, ref_size=224,
                   mask_thresh=0.5, device=None, scale_matched_ref=False, tta=1):
    """`tta` averages sigmoid maps over dihedral views of image AND reference
    (the reference crop is a piece of the same image, so it turns with it)."""
    device = device or next(model.parameters()).device
    transforms = tta_transforms(tta)
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
            scale = image_max_size / max(h0, w0)
            nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
            image = cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)
            image_tensor = _normalize_chw(image).unsqueeze(0).to(device)
            for ref_idx, (ref_mask, category) in enumerate(zip(masks0, categories)):
                rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
                x, y, w, h = sample_reference_box(ref_mask, 128, 512, rng=rng)
                crop = image0[y:y + h, x:x + w]
                if not crop.size:
                    continue
                if scale_matched_ref:
                    crop = scale_matched_reference(crop, scale, ref_size)
                else:
                    crop = cv2.resize(crop, (ref_size, ref_size),
                                      interpolation=cv2.INTER_LINEAR)
                reference = _normalize_chw(crop).unsqueeze(0).to(device)
                # The anchor plane must be built the same way here as in the
                # dataset, or an anchored model is evaluated without the input it
                # was trained on. Built at native size, then resized with the
                # image so it stays pixel-aligned.
                ref_box = None
                if getattr(model, "anchor", False):
                    box_native = np.zeros((h0, w0), np.uint8)
                    box_native[y:y + h, x:x + w] = 1
                    ref_box = torch.from_numpy(
                        cv2.resize(box_native, (nw, nh),
                                   interpolation=cv2.INTER_NEAREST)
                    ).float()[None, None].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    acc = None
                    for hf, k in transforms:
                        img_t = _dihedral(image_tensor, hf, k)
                        ref_t = _dihedral(reference, hf, k)
                        box_t = _dihedral(ref_box, hf, k) if ref_box is not None else None
                        prob_t = model(img_t, ref_t, ref_box=box_t).sigmoid()
                        prob = _dihedral_inv(prob_t.float(), hf, k)[0, 0]
                        acc = prob if acc is None else acc + prob
                    probability = acc / len(transforms)
                pred_small = probability > mask_thresh
                prediction = cv2.resize(pred_small.cpu().numpy().astype(np.uint8),
                                        (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
                target = np.logical_or.reduce(
                    [m for m, c in zip(masks0, categories) if c == category])
                intersection = int((prediction & target).sum())
                union = int((prediction | target).sum())
                rows.append({"image_index": image_idx,
                             "reference_instance": ref_idx,
                             "category": category,
                             "reference_box_native": [x, y, w, h],
                             "iou": intersection / max(union, 1),
                             "prediction_pixels": int(prediction.sum()),
                             "target_pixels": int(target.sum())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--indices", default=HOLDOUT)
    parser.add_argument("--image-max-size", type=int, default=1280)
    parser.add_argument("--ref-size", type=int, default=224)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--tta", type=int, default=1, choices=(1, 2, 4, 8),
                        help="dihedral test-time augmentation views to average")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_refunet(args.checkpoint, device)
    records = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                   config="real-world-test", split="test")
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    ckpt_args = checkpoint.get("args", {}) or {}
    rows = evaluate_model(model, records, indices, args.image_max_size,
                          args.ref_size, args.mask_thresh, device,
                          scale_matched_ref=bool(ckpt_args.get("scale_matched_ref")),
                          tta=args.tta)
    mean_iou = float(np.mean([row["iou"] for row in rows]))
    metrics = {"metric": "reference-conditioned union IoU",
               "checkpoint": args.checkpoint,
               "checkpoint_epoch": checkpoint.get("actual_epoch",
                                                    checkpoint.get("epoch")),
               "image_max_size": args.image_max_size,
               "mask_thresh": args.mask_thresh, "tta": args.tta,
               "n_images": len(indices), "n_reference_selections": len(rows),
               "mean_iou": mean_iou, "selections": rows}
    path = Path(args.metrics_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        metrics, indent=2,
        default=lambda value: (int(value) if isinstance(value, np.integer)
                               else float(value))) + "\n")
    print(json.dumps({k: metrics[k] for k in
                      ("checkpoint_epoch", "n_reference_selections", "mean_iou")},
                     indent=2))


if __name__ == "__main__":
    main()
