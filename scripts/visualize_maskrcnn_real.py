#!/usr/bin/env python3
"""Export GT-selected areas and Mask R-CNN predictions for heldout real plans."""

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask
from scripts.train_maskrcnn import HOLDOUT, build_model


COLORS = [
    (230, 57, 70), (29, 53, 87), (69, 123, 157), (42, 157, 143),
    (233, 196, 106), (244, 162, 97), (231, 111, 81), (131, 56, 236),
    (0, 187, 249), (0, 245, 212), (255, 0, 110), (58, 134, 255),
]


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-max-size", type=int, default=1536)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--indices", default=HOLDOUT)
    p.add_argument("--tta-flips", action="store_true",
                   help="combine original, horizontal-flip, and vertical-flip views")
    p.add_argument("--mask-nms-thresh", type=float, default=0.5)
    return p.parse_args()


def overlay(base, masks, labels, alpha=0.42):
    canvas = base.copy()
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = np.array(COLORS[i % len(COLORS)], dtype=np.uint8)
        canvas[mask] = ((1 - alpha) * canvas[mask] + alpha * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(int(x) for x in color), 3)
        ys, xs = np.where(mask)
        if len(xs):
            x, y = int(xs.min()), max(18, int(ys.min()))
            cv2.putText(canvas, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, tuple(int(x) for x in color), 2, cv2.LINE_AA)
    return canvas


def titled(image, title):
    band = 46
    out = Image.new("RGB", (image.shape[1], image.shape[0] + band), "white")
    out.paste(Image.fromarray(image), (0, band))
    ImageDraw.Draw(out).text((12, 14), title, fill="black")
    return np.array(out)


def mask_nms(masks, scores, threshold):
    if not len(masks):
        return np.zeros(0, dtype=np.int64)
    flat = masks.reshape(len(masks), -1).astype(np.uint8)
    areas = flat.sum(1)
    retained = []
    for candidate in np.argsort(-scores):
        suppress = False
        for prior in retained:
            inter = np.logical_and(flat[candidate], flat[prior]).sum()
            union = areas[candidate] + areas[prior] - inter
            if inter / max(union, 1) > threshold:
                suppress = True
                break
        if not suppress:
            retained.append(int(candidate))
    return np.asarray(retained, dtype=np.int64)


def matching_audit(base, gt, pred, rows, cols, ious):
    canvas = base.copy()
    matched_pred = set()
    pairs = []
    for gi, pi in zip(rows, cols):
        iou = float(ious[gi, pi])
        if iou < 0.5:
            continue
        matched_pred.add(int(pi))
        color = np.array(COLORS[int(gi) % len(COLORS)], dtype=np.uint8)
        canvas[pred[pi]] = (0.58 * canvas[pred[pi]] + 0.42 * color).astype(np.uint8)
        pc, _ = cv2.findContours(pred[pi].astype(np.uint8), cv2.RETR_CCOMP,
                                 cv2.CHAIN_APPROX_SIMPLE)
        gc, _ = cv2.findContours(gt[gi].astype(np.uint8), cv2.RETR_CCOMP,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, pc, -1, tuple(int(x) for x in color), 3)
        cv2.drawContours(canvas, gc, -1, (0, 0, 0), 2)
        ys, xs = np.where(gt[gi])
        if len(xs):
            cv2.putText(canvas, f"GT{gi + 1} IoU {iou:.2f}",
                        (int(xs.min()) + 3, max(18, int(ys.min()) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        pairs.append({"gt": int(gi + 1), "prediction": int(pi + 1), "iou": iou})
    for pi, mask in enumerate(pred):
        if pi in matched_pred:
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 0, 0), 2)
    return canvas, pairs, matched_pred


def main():
    args = args_parser()
    root = Path(args.out)
    for name in ("ground_truth", "predictions", "matching_audit", "comparisons"):
        (root / name).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model = build_model(False, args.image_max_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    records = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                   config="real-world-test", split="test")
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    manifest = {"checkpoint": args.checkpoint, "checkpoint_epoch": ckpt.get("epoch"),
                "image_max_size": args.image_max_size,
                "score_thresh": args.score_thresh, "tta_flips": args.tta_flips,
                "mask_nms_thresh": args.mask_nms_thresh,
                "scenes": []}

    for idx in indices:
        rec = records[idx]
        original = np.array(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        h0, w0 = original.shape[:2]
        scale = args.image_max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
        image = cv2.resize(original, (nw, nh), interpolation=cv2.INTER_LINEAR)
        gt = [cv2.resize(render_instance_mask(a["segmentation"], h0, w0),
                         (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
              for a in anns]
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255).to(device)
        with torch.inference_mode():
            if args.tta_flips:
                views = [tensor, tensor.flip(-1), tensor.flip(-2)]
                view_outputs = model(views)
                out = {
                    "masks": torch.cat([view_outputs[0]["masks"],
                                        view_outputs[1]["masks"].flip(-1),
                                        view_outputs[2]["masks"].flip(-2)]),
                    "scores": torch.cat([o["scores"] for o in view_outputs]),
                }
            else:
                out = model([tensor])[0]
        keep = out["scores"] > args.score_thresh
        scores = out["scores"][keep].cpu().numpy()
        pred = (out["masks"][keep, 0] > 0.5).cpu().numpy()
        raw_predictions = len(pred)
        retained = mask_nms(pred, scores, args.mask_nms_thresh)
        pred, scores = pred[retained], scores[retained]

        gt_arr = np.stack(gt) if gt else np.zeros((0, nh, nw), dtype=bool)
        if len(gt_arr) and len(pred):
            gm = gt_arr.reshape(len(gt_arr), -1).astype(np.float32)
            pm = pred.reshape(len(pred), -1).astype(np.float32)
            inter = gm @ pm.T
            union = gm.sum(1)[:, None] + pm.sum(1)[None] - inter
            ious = inter / np.maximum(union, 1)
            rows, cols = linear_sum_assignment(-ious)
        else:
            ious = np.zeros((len(gt_arr), len(pred)), dtype=np.float32)
            rows = cols = np.zeros(0, dtype=np.int64)

        gt_view = overlay(image, gt, [f"GT {i + 1}" for i in range(len(gt))])
        pred_view = overlay(image, pred,
                            [f"P{i + 1} {score:.2f}" for i, score in enumerate(scores)])
        audit_view, pairs, matched_pred = matching_audit(
            image, gt_arr, pred, rows, cols, ious)
        tp = len(pairs)
        fp = len(pred) - tp
        fn = len(gt) - tp
        raw_panel = titled(image, f"HF heldout {idx} - original")
        gt_panel = titled(gt_view, f"Selected/labelled areas ({len(gt)} instances)")
        pred_panel = titled(pred_view,
                            f"Predictions after mask NMS ({len(pred)} masks)")
        audit_panel = titled(audit_view,
                             f"One-to-one audit: TP={tp} FP={fp} FN={fn}; red=FP")
        Image.fromarray(gt_panel).save(root / "ground_truth" / f"hf_{idx:02d}.png")
        Image.fromarray(pred_panel).save(root / "predictions" / f"hf_{idx:02d}.png")
        Image.fromarray(audit_panel).save(root / "matching_audit" / f"hf_{idx:02d}.png")
        comparison = np.concatenate([raw_panel, gt_panel, pred_panel, audit_panel], axis=1)
        Image.fromarray(comparison).save(root / "comparisons" / f"hf_{idx:02d}.png")
        manifest["scenes"].append({
            "index": idx, "ground_truth": len(gt),
            "raw_tta_predictions": raw_predictions,
            "predictions_after_nms": len(pred), "scores": scores.tolist(),
            "tp50": tp, "fp50": fp, "fn50": fn, "matches": pairs,
        })
        print(f"saved heldout {idx}: gt={len(gt)} raw={raw_predictions} "
              f"nms={len(pred)} TP={tp} FP={fp} FN={fn}", flush=True)

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "README.md").write_text(
        "# Heldout real visualizations\n\n"
        "- `ground_truth/`: labelled areas used only for heldout evaluation.\n"
        "- `predictions/`: score-filtered masks after mask-IoU NMS.\n"
        "- `matching_audit/`: one-to-one TP masks; GT is black, unmatched FP is red.\n"
        "- `comparisons/`: original, GT, deduplicated predictions, and audit.\n"
        "- `manifest.json`: thresholds, TP/FP/FN, match IoUs, and scores.\n")


if __name__ == "__main__":
    main()
