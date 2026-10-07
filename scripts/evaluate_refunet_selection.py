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
from refmask2former.dataset import (_normalize_chw, exclusion_mask, reference_region,
                                    render_instance_mask, sample_reference_box,
                                    scale_matched_reference)
from refmask2former.ref_unet import RefUNet
from refmask2former.ref_attn_unet import RefCrossAttnUNet
from refmask2former.ref_swin_unet import RefSwinUNet
from refmask2former.ref_dino_unet import BACKBONES as DINO_BACKBONES
from refmask2former.ref_dino_unet import RefDinoUNet

HOLDOUT = "12,16,27,7,11,25,23,1,18,2,0,3,14,24"
# Default reference boxes on the real sheets: the hand-reviewed set (moved where the
# automatic box was unrealistic, the rest kept as drawn). `--boxes auto` = automatic only.
HAND_BOXES = Path(__file__).resolve().parent.parent / "eval_boxes" / "hand_v1.json"


def hand_boxes():
    return json.loads(HAND_BOXES.read_text())


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
    if args.get("model", "unet") == "crossattn":
        model = RefCrossAttnUNet(width=int(args.get("width", 128)), pretrained=False,
                                 num_heads=int(args.get("attn_heads", 4))).to(device)
    elif str(args.get("model", "unet")) in DINO_BACKBONES:
        model = RefDinoUNet(width=int(args.get("width", 128)),
                              pretrained=False,
                              backbone=str(args.get("model")))
    elif str(args.get("model", "unet")).startswith("swin"):
        model = RefSwinUNet(width=int(args.get("width", 128)), pretrained=False,
                            backbone=str(args["model"]),
                            corr_grid=int(args.get("corr_grid", 0) or 0),
                            decoder=str(args.get("swin_decoder", "baseline")),
                            roi_ref=bool(args.get("roi_ref", False)),
                            roi_mode=str(args.get("roi_ref_mode", "replace")),
                            roi_min_cells=float(args.get("roi_min_cells", 0.0) or 0.0)).to(device)
    else:
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
                   mask_thresh=0.5, device=None, scale_matched_ref=False, tta=1,
                   save_probs=None, box_overrides=None, label_fixes=None, q_batch=8):
    """`tta` averages sigmoid maps over dihedral views of image AND reference
    (the reference crop is a piece of the same image, so it turns with it).
    `q_batch` > 0: models with `forward_cached` (swin) run the image backbone once per
    sheet and view and the questions `q_batch` at a time; 0 = one full forward each."""
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
            # synthetic pools: reference boxes stay off posts / railings / furniture (ref_exclude)
            excl = exclusion_mask(rec.get("ref_exclude") if isinstance(rec, dict) else None, h0, w0)
            # --label-fixes: extra TARGET pixels per category (missed labels); the masks
            # above, which the reference boxes are sampled from, are left as they are
            fix_masks = {}
            for fx in (label_fixes or {}).get(str(image_idx), []):
                m = np.zeros((h0, w0), np.uint8)
                cv2.fillPoly(m, [np.round(np.array(fx["polygon"], np.float64)).astype(np.int32)], 1)
                fix_masks[fx["category"]] = fix_masks.get(fx["category"], np.zeros((h0, w0), bool)) | m.astype(bool)
            scale = image_max_size / max(h0, w0)
            nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
            image = cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)
            image_tensor = _normalize_chw(image).unsqueeze(0).to(device)
            # 1) the sheet's questions: reference box per labelled instance
            questions = []
            for ref_idx, (ref_mask, category) in enumerate(zip(masks0, categories)):
                rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
                region = reference_region(ref_mask, excl)
                if region is None:      # piece wholly behind a railing / furniture: not a question
                    continue
                x, y, w, h = sample_reference_box(region, 128, 512, rng=rng)
                # --boxes: a hand-placed box (native pixels) replaces the automatic one;
                # null drops the question. The rng draw above still happens, so every
                # other question keeps its automatic box exactly.
                ov = (box_overrides or {}).get(str(image_idx), {})
                if str(ref_idx) in ov:
                    if ov[str(ref_idx)] is None:
                        continue
                    x, y, w, h = (int(round(v)) for v in ov[str(ref_idx)])
                    x, y = max(0, min(x, w0 - 1)), max(0, min(y, h0 - 1))
                    w, h = max(1, min(w, w0 - x)), max(1, min(h, h0 - y))
                crop = image0[y:y + h, x:x + w]
                if not crop.size:
                    continue
                if scale_matched_ref:
                    crop = scale_matched_reference(crop, scale, ref_size)
                else:
                    crop = cv2.resize(crop, (ref_size, ref_size),
                                      interpolation=cv2.INTER_LINEAR)
                questions.append((ref_idx, category, (x, y, w, h), _normalize_chw(crop)))

            def box_plane(x, y, w, h):
                # The anchor plane must be built the same way here as in the
                # dataset, or an anchored model is evaluated without the input it
                # was trained on. Built at native size, then resized with the
                # image so it stays pixel-aligned.
                box_native = np.zeros((h0, w0), np.uint8)
                box_native[y:y + h, x:x + w] = 1
                return torch.from_numpy(cv2.resize(box_native, (nw, nh),
                                                   interpolation=cv2.INTER_NEAREST)).float()[None, None]

            use_box = getattr(model, "anchor", False) or getattr(model, "roi_ref", False)
            # 2) probabilities. Swin models: the image backbone runs ONCE per sheet and TTA
            # view (it does not depend on the reference); questions go through the
            # reference encoder + decoder `q_batch` at a time. Others: one forward each.
            cached = q_batch > 0 and hasattr(model, "forward_cached")
            feats = {}
            targets = {}
            for c0 in range(0, len(questions), q_batch if cached else 1):
                chunk = questions[c0:c0 + (q_batch if cached else 1)]
                refs = torch.stack([q[3] for q in chunk]).to(device)
                boxes = torch.cat([box_plane(*q[2]) for q in chunk]).to(device) if use_box else None
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    acc = None
                    for hf, k in transforms:
                        img_t = _dihedral(image_tensor, hf, k)
                        ref_t = _dihedral(refs, hf, k)
                        box_t = _dihedral(boxes, hf, k) if boxes is not None else None
                        if cached:
                            if (hf, k) not in feats:
                                feats[(hf, k)] = model.features(img_t)
                            logit = model.forward_cached(feats[(hf, k)], img_t.shape[-2:], ref_t, box_t)
                        else:
                            logit = model(img_t, ref_t, ref_box=box_t)
                        prob = _dihedral_inv(logit.sigmoid().float(), hf, k)[:, 0]
                        acc = prob if acc is None else acc + prob
                    probs = acc / len(transforms)
                # 3) score each question
                for (ref_idx, category, (x, y, w, h), _), probability in zip(chunk, probs):
                    if save_probs:
                        # Inference-resolution probability, quantised to uint8, for offline
                        # ensembling / threshold sweeps (scripts/ensemble_selection.py).
                        np.savez_compressed(
                            Path(save_probs) / f"{image_idx:03d}_{ref_idx:03d}.npz",
                            prob=(probability.float().cpu().numpy() * 255).round().astype(np.uint8))
                    pred_small = probability > mask_thresh
                    prediction = cv2.resize(pred_small.cpu().numpy().astype(np.uint8),
                                            (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if category not in targets:
                        target = np.logical_or.reduce(
                            [m for m, c in zip(masks0, categories) if c == category])
                        if category in fix_masks:
                            target = target | fix_masks[category]
                        targets[category] = (target, int(target.sum()))
                    target, target_pixels = targets[category]
                    intersection = int((prediction & target).sum())
                    prediction_pixels = int(prediction.sum())
                    union = prediction_pixels + target_pixels - intersection
                    rows.append({"image_index": image_idx,
                                 "reference_instance": ref_idx,
                                 "category": category,
                                 "reference_box_native": [x, y, w, h],
                                 "iou": intersection / max(union, 1),
                                 "prediction_pixels": prediction_pixels,
                                 "target_pixels": target_pixels})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--indices", default="",
                        help="comma list or 'all'; default HF14 for the real sheets, every sheet for --local-pool")
    parser.add_argument("--image-max-size", type=int, default=1280)
    parser.add_argument("--ref-size", type=int, default=224)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--tta", type=int, default=1, choices=(1, 2, 4, 8),
                        help="dihedral test-time augmentation views to average")
    parser.add_argument("--boxes", default="hand",
                        help="'hand' (default on the real sheets) = eval_boxes/hand_v1.json; 'auto' = "
                             "automatic boxes only (the protocol before 2026-10-03); or a JSON "
                             "{image_index: {question_index: [x, y, w, h] | null}} in native pixels "
                             "replacing those questions' automatic boxes, or dropping them (null). "
                             "--local-pool uses automatic boxes unless --boxes names a file for that pool.")
    parser.add_argument("--q-batch", type=int, default=8,
                        help="questions per batch on one cached image backbone (swin models); 0 = the "
                             "old one-forward-per-question path. Lower it if 4096 runs out of GPU memory")
    parser.add_argument("--label-fixes", default="",
                        help="JSON {image_index: [{category, polygon: [[x, y], ...]}]} in native pixels: "
                             "add missed pixels to the TARGET of every question of that category "
                             "(eval_labels/hf14_fixes_v1.json). Boxes and questions are unchanged.")
    parser.add_argument("--local-pool", default="",
                        help="score a local synthetic pool (images/ + annotations/) instead of the "
                             "real-world test set; --indices then indexes its sorted sheets "
                             "('all' = every sheet)")
    parser.add_argument("--save-probs", default="",
                        help="directory: write each question's probability map "
                             "(uint8, inference resolution) as <image>_<question>.npz")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_refunet(args.checkpoint, device)
    if args.local_pool:
        pool = Path(args.local_pool)
        records = []
        for f in sorted((pool / "annotations").glob("*.json")):
            ann = json.loads(f.read_text())
            records.append({"image": (pool / "images" / ann["image"]["file_name"]).read_bytes(),
                            "annotations": ann["annotations"], "ref_exclude": ann.get("ref_exclude")})
    else:
        records = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                       config="real-world-test", split="test")
    if not args.indices:
        args.indices = "all" if args.local_pool else HOLDOUT
    indices = (list(range(len(records))) if args.indices == "all" else
               [int(value) for value in args.indices.split(",") if value.strip()])
    ckpt_args = checkpoint.get("args", {}) or {}
    if args.save_probs:
        Path(args.save_probs).mkdir(parents=True, exist_ok=True)
    # --local-pool: automatic boxes unless --boxes names a file made for that pool (its _meta.pool must
    # match, e.g. eval_boxes/gemini30_v1.json for data/eval_pools/gemini30_s20261006)
    if args.local_pool:
        boxes_path = "" if args.boxes in ("hand", "auto") else args.boxes
    else:
        boxes_path = ("" if args.boxes == "auto" else
                      "eval_boxes/hand_v1.json" if args.boxes == "hand" else args.boxes)
    box_overrides = (json.loads((HAND_BOXES if boxes_path == "eval_boxes/hand_v1.json" and args.boxes == "hand"
                                 else Path(boxes_path)).read_text()) if boxes_path else None)
    if args.local_pool and box_overrides is not None:
        want = box_overrides.get("_meta", {}).get("pool")
        if want and Path(want).resolve() != Path(args.local_pool).resolve():
            raise SystemExit(f"--boxes {boxes_path} is for pool {want}, not {args.local_pool}")
    rows = evaluate_model(model, records, indices, args.image_max_size,
                          args.ref_size, args.mask_thresh, device,
                          scale_matched_ref=bool(ckpt_args.get("scale_matched_ref")),
                          tta=args.tta, save_probs=args.save_probs or None,
                          box_overrides=box_overrides, q_batch=args.q_batch,
                          label_fixes=json.loads(Path(args.label_fixes).read_text()) if args.label_fixes else None)
    mean_iou = float(np.mean([row["iou"] for row in rows]))
    metrics = {"metric": "reference-conditioned union IoU",
               "checkpoint": args.checkpoint,
               "checkpoint_epoch": checkpoint.get("actual_epoch",
                                                    checkpoint.get("epoch")),
               "image_max_size": args.image_max_size,
               "mask_thresh": args.mask_thresh, "tta": args.tta,
               "boxes": boxes_path or "automatic",
               "label_fixes": args.label_fixes or "none",
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
