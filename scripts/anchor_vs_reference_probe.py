#!/usr/bin/env python3
"""Does the model follow the REFERENCE crop or the ANCHOR rectangle?

Suppression is otherwise only inferred from a score drop. This observes it: the
model is given deliberately CONTRADICTORY inputs -- the anchor rectangle sits in
family A while the reference crop is taken from family B in the same image -- and
we ask which one the prediction obeys.

    follow_anchor     IoU(prediction, union of family A)   the anchor's family
    follow_reference  IoU(prediction, union of family B)   the crop's family

A model doing the intended task follows the reference. A model that has let the
anchor suppress its matching pathway follows the anchor. The un-anchored baseline
has no anchor input at all, so it can only follow the reference -- it is the
control proving the probe reads what it claims.

Also reports plain reference sensitivity: 1 - IoU(pred with crop A, pred with
crop B), holding the anchor fixed. Near 0 means the reference channel has
essentially no causal effect on the output.

    PYTHONPATH=. python3 scripts/anchor_vs_reference_probe.py \
        --checkpoint data/runs/ck_*/epoch_8.pth --labels base anchor anchordrop
"""
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
from refmask2former.dataset import _normalize_chw, render_instance_mask, sample_reference_box
from scripts.evaluate_refunet_selection import HOLDOUT, load_refunet

VALIDATION = "4,5,6,8,9,10,13,15,17,19,20,21,22,26"


def _crop_and_box(image0, mask, image_idx, ref_idx, ref_size):
    """The evaluator's deterministic rectangle for one instance."""
    rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
    x, y, w, h = sample_reference_box(mask, 128, 512, rng=rng)
    crop = image0[y:y + h, x:x + w]
    if not crop.size:
        return None, None
    crop = cv2.resize(crop, (ref_size, ref_size), interpolation=cv2.INTER_LINEAR)
    return crop, (x, y, w, h)


@torch.no_grad()
def _predict(model, image_tensor, crop, box, h0, w0, nh, nw, thresh, device):
    reference = _normalize_chw(crop).unsqueeze(0).to(device)
    ref_box = None
    if getattr(model, "anchor", False):
        x, y, w, h = box
        native = np.zeros((h0, w0), np.uint8)
        native[y:y + h, x:x + w] = 1
        ref_box = torch.from_numpy(
            cv2.resize(native, (nw, nh), interpolation=cv2.INTER_NEAREST)
        ).float()[None, None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prob = model(image_tensor, reference, ref_box=ref_box).sigmoid()[0, 0]
    small = (prob > thresh).cpu().numpy().astype(np.uint8)
    return cv2.resize(small, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)


def _iou(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / max(u, 1)


def _anchor_component(union, box):
    """The connected component of `union` that the anchor rectangle sits in."""
    n, lab = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
    if n <= 1:
        return union
    x, y, w, h = box
    cy = int(np.clip(y + h // 2, 0, lab.shape[0] - 1))
    cx = int(np.clip(x + w // 2, 0, lab.shape[1] - 1))
    lid = lab[cy, cx]
    if lid == 0:
        ys, xs = np.nonzero(lab)
        if not len(ys):
            return union
        d = (ys - cy) ** 2 + (xs - cx) ** 2
        lid = lab[ys[d.argmin()], xs[d.argmin()]]
    return lab == lid


@torch.no_grad()
def probe(model, records, indices, image_max_size, ref_size, thresh, device):
    fa, fr, sens = [], [], []
    honest, pred_local, gt_local = [], [], []
    for image_idx in indices:
        rec = records[image_idx]
        image0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = rec["annotations"]
        anns = json.loads(anns) if isinstance(anns, str) else anns
        h0, w0 = image0.shape[:2]
        masks = [render_instance_mask(a["segmentation"], h0, w0).astype(bool) for a in anns]
        cats = [a.get("category_name", "pattern") for a in anns]
        if len(set(cats)) < 2:
            continue                      # need a second family to swap in
        scale = image_max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
        image_tensor = _normalize_chw(
            cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ).unsqueeze(0).to(device)

        unions = {c: np.logical_or.reduce([m for m, cc in zip(masks, cats) if cc == c])
                  for c in set(cats)}

        for i, (mi, ca) in enumerate(zip(masks, cats)):
            others = [j for j, cb in enumerate(cats) if cb != ca]
            if not others or not mi.any():
                continue
            j = others[0]
            cb = cats[j]
            crop_a, box_a = _crop_and_box(image0, mi, image_idx, i, ref_size)
            crop_b, _ = _crop_and_box(image0, masks[j], image_idx, j, ref_size)
            if crop_a is None or crop_b is None:
                continue
            # Honest input: crop from A, anchor on A.
            pred_a = _predict(model, image_tensor, crop_a, box_a, h0, w0, nh, nw,
                              thresh, device)
            # Contradictory input: crop from B, anchor still on A.
            pred_b = _predict(model, image_tensor, crop_b, box_a, h0, w0, nh, nw,
                              thresh, device)
            fa.append(_iou(pred_b, unions[ca]))     # obeyed the anchor
            fr.append(_iou(pred_b, unions[cb]))     # obeyed the reference
            sens.append(1.0 - _iou(pred_a, pred_b))
            # Non-contradictory measures on the HONEST input (crop A, anchor A).
            # The swap above feeds a combination that never occurs in training,
            # so an anchored model may simply be off-distribution there; these
            # two are read from an input it was actually trained on.
            honest.append(_iou(pred_a, unions[ca]))
            comp = _anchor_component(unions[ca], box_a)
            # Share of the PREDICTION confined to the anchor's own component. If
            # the model is riding the anchor this runs far above the share of the
            # ground truth that lives there.
            ps = int(pred_a.sum())
            if ps:
                pred_local.append(int((pred_a & comp).sum()) / ps)
                gt_local.append(int(comp.sum()) / max(1, int(unions[ca].sum())))
    return (np.array(fa), np.array(fr), np.array(sens), np.array(honest),
            np.array(pred_local), np.array(gt_local))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--indices", default=VALIDATION,
                    help="default is the validation split, not the acceptance set")
    ap.add_argument("--image-max-size", type=int, default=1280)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--cache-dir", default="./data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_parquet_records(cache_dir=args.cache_dir,
                                   config="real-world-test", split="test")
    indices = [int(v) for v in args.indices.split(",") if v.strip()]
    labels = args.labels or [Path(c).parent.name for c in args.checkpoint]

    print("\ncontradictory input: reference crop from family B, anchor on family A")
    print(f"{'model':<28}{'n':>5}{'honest':>9}{'pred_local':>12}{'gt_local':>10}"
          f"{'follow_anchor':>15}{'follow_ref':>12}{'ref_sens':>11}")
    for ck, lab in zip(args.checkpoint, labels):
        model, _ = load_refunet(ck, device)
        fa, fr, sens, honest, pl, gl = probe(
            model, records, indices, args.image_max_size, args.ref_size,
            args.mask_thresh, device)
        del model
        torch.cuda.empty_cache()
        if not len(fa):
            print(f"{lab:<28}{0:>5}  no multi-family images")
            continue
        print(f"{lab:<28}{len(fa):>5}{honest.mean():>9.3f}{pl.mean():>12.3f}"
              f"{gl.mean():>10.3f}{fa.mean():>15.3f}{fr.mean():>12.3f}{sens.mean():>11.3f}")
    print()


if __name__ == "__main__":
    main()
