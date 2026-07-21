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
    p.add_argument("--mask-thresh", type=float, default=0.5)
    p.add_argument("--match-thresh", type=float, default=0.0)
    p.add_argument("--match-margin", type=float, default=None,
                   help="Optional relative matching rule: among object queries, "
                        "keep similarities within this margin of the best query. "
                        "Avoids absolute cosine calibration drift per reference.")
    p.add_argument("--match-cluster", action="store_true",
                   help="Split object-query similarities into two 1-D clusters and "
                        "keep the higher-similarity cluster. This is label-free and "
                        "adapts to each reference's cosine scale.")
    p.add_argument("--match-largest-gap", action="store_true",
                   help="Keep the leading similarity-ranked queries above the "
                        "largest adjacent score gap (label-free per reference).")
    p.add_argument("--match-top-k", type=int, default=None,
                   help="Keep the globally calibrated top K object queries by "
                        "reference similarity.")
    p.add_argument("--match-roi-imagenet", action="store_true",
                   help="Compare the reference with isolated predicted-instance "
                        "crops using the checkpoint's pretrained reference backbone.")
    p.add_argument("--match-roi-fixed-patch", action="store_true",
                   help="With --match-roi-imagenet, crop each instance at the "
                        "reference rectangle's native scale instead of its bbox.")
    p.add_argument("--match-roi-blend", type=float, default=None,
                   help="Rank-fuse ROI similarity with learned query similarity; "
                        "value is ROI rank weight in [0,1].")
    p.add_argument("--oracle-overlap-thresh", type=float, default=None,
                   help="Diagnostic only: select queries by fraction of their mask "
                        "inside GT target. Never use this to report product scores.")
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
                if "pred_match_logits" in outputs:
                    similarities = outputs["pred_match_logits"][0]
                pred_masks = (F.interpolate(outputs["pred_masks"], size=(nh, nw),
                                             mode="bilinear", align_corners=False)
                              .sigmoid()[0] > args.mask_thresh)
                objects = scores > args.score_thresh
                if args.match_roi_imagenet and objects.any():
                    roi_indices = objects.nonzero(as_tuple=False).flatten()
                    white = image_tensor.new_tensor(
                        [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224,
                         (1.0 - 0.406) / 0.225])[:, None, None]
                    crops = []
                    for query_idx in roi_indices.tolist():
                        instance_mask = pred_masks[query_idx]
                        points = instance_mask.nonzero(as_tuple=False)
                        if points.numel() == 0:
                            crops.append(white.expand(3, args.ref_size,
                                                      args.ref_size).clone())
                            continue
                        if args.match_roi_fixed_patch:
                            mask_np = instance_mask.cpu().numpy().astype(np.uint8)
                            distance = cv2.distanceTransform(mask_np,
                                                             cv2.DIST_L2, 5)
                            cy, cx = np.unravel_index(distance.argmax(),
                                                     distance.shape)
                            crop_w = max(8, round(w * scale))
                            crop_h = max(8, round(h * scale))
                            x0 = max(0, min(nw - crop_w, cx - crop_w // 2))
                            y0 = max(0, min(nh - crop_h, cy - crop_h // 2))
                            x1, y1 = min(nw, x0 + crop_w), min(nh, y0 + crop_h)
                        else:
                            y0, x0 = points.min(0).values.tolist()
                            y1, x1 = (points.max(0).values + 1).tolist()
                        crop = image_tensor[0, :, y0:y1, x0:x1]
                        crop_mask = instance_mask[y0:y1, x0:x1]
                        crop = torch.where(crop_mask[None], crop,
                                           white.expand_as(crop))
                        crop = F.interpolate(crop[None],
                                             size=(args.ref_size, args.ref_size),
                                             mode="bilinear",
                                             align_corners=False)[0]
                        crops.append(crop)
                    roi_batch = torch.stack(crops)
                    roi_features = model.reference_encoder.backbone(
                        roi_batch).flatten(1)
                    ref_features = model.reference_encoder.backbone(
                        reference).flatten(1)
                    roi_features = F.normalize(roi_features.float(), dim=-1)
                    ref_features = F.normalize(ref_features.float(), dim=-1)
                    roi_sims = roi_features @ ref_features[0]
                    similarities = similarities.clone()
                    if args.match_roi_blend is None:
                        similarities[roi_indices] = roi_sims.to(similarities.dtype)
                    else:
                        def _ranks(values):
                            order = values.argsort()
                            ranks = torch.empty_like(values, dtype=torch.float32)
                            ranks[order] = torch.arange(
                                len(values), device=values.device,
                                dtype=torch.float32)
                            return ranks / max(len(values) - 1, 1)
                        learned_ranks = _ranks(similarities[roi_indices].float())
                        roi_ranks = _ranks(roi_sims.float())
                        alpha = args.match_roi_blend
                        fused = (1.0 - alpha) * learned_ranks + alpha * roi_ranks
                        similarities[roi_indices] = fused.to(similarities.dtype)
                target = np.logical_or.reduce(
                    [m for m, c in zip(masks, categories) if c == category])
                if args.oracle_overlap_thresh is not None:
                    target_t = torch.from_numpy(target).to(pred_masks.device)
                    overlap = (pred_masks & target_t).flatten(1).sum(1)
                    precision = overlap / pred_masks.flatten(1).sum(1).clamp(min=1)
                    keep = objects & (precision >= args.oracle_overlap_thresh)
                elif args.match_top_k is not None and objects.any():
                    object_indices = objects.nonzero(as_tuple=False).flatten()
                    k = min(args.match_top_k, object_indices.numel())
                    chosen = object_indices[similarities[object_indices].topk(k).indices]
                    keep = torch.zeros_like(objects)
                    keep[chosen] = True
                elif args.match_largest_gap and objects.sum() >= 2:
                    values = similarities[objects]
                    ordered, _ = values.sort(descending=True)
                    split = (ordered[:-1] - ordered[1:]).argmax()
                    threshold = (ordered[split] + ordered[split + 1]) / 2
                    keep = objects & (similarities >= threshold)
                elif args.match_cluster and objects.sum() >= 2:
                    values = similarities[objects]
                    lo, hi = values.min(), values.max()
                    for _ in range(12):
                        midpoint = (lo + hi) / 2
                        high_group = values >= midpoint
                        if high_group.all() or (~high_group).all():
                            break
                        new_lo = values[~high_group].mean()
                        new_hi = values[high_group].mean()
                        if torch.isclose(lo, new_lo) and torch.isclose(hi, new_hi):
                            lo, hi = new_lo, new_hi
                            break
                        lo, hi = new_lo, new_hi
                    keep = objects & (similarities >= (lo + hi) / 2)
                elif args.match_margin is not None and objects.any():
                    best_similarity = similarities[objects].max()
                    keep = objects & (similarities >= best_similarity - args.match_margin)
                else:
                    keep = objects & (similarities > args.match_thresh)
                prediction = (pred_masks[keep].any(0).cpu().numpy() if keep.any()
                              else np.zeros((nh, nw), dtype=bool))
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
        "mask_thresh": args.mask_thresh,
        "match_margin": args.match_margin,
        "match_cluster": args.match_cluster,
        "match_largest_gap": args.match_largest_gap,
        "match_top_k": args.match_top_k,
        "match_roi_imagenet": args.match_roi_imagenet,
        "match_roi_fixed_patch": args.match_roi_fixed_patch,
        "match_roi_blend": args.match_roi_blend,
        "oracle_overlap_thresh": args.oracle_overlap_thresh,
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
