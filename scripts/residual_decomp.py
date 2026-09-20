#!/usr/bin/env python3
"""What is the missing IoU actually MADE OF?

A mean IoU of 0.78 says how much is wrong, never what kind of wrong. This
splits every error pixel (false positive or false negative) into five buckets
whose remedies are different architectural axes:

  boundary    - within `--band` px of the GT boundary. The shape is right and
                the edge is off: an output-stride / upsampling / edge problem.
  missed_region  - false negatives in a GT connected component the model barely
                found at all (recall < `--found-thresh`). It did not propagate
                the match to that region: a long-range matching problem.
  missed_inside  - false negatives away from the boundary inside a component it
                DID find. Holes in a found region: a fill / consistency problem.
  false_region   - false positives forming a component mostly outside the GT.
                It selected the wrong material: a discrimination problem.
  false_fringe   - remaining false positives (spill and speckle near found
                regions).

Also sweeps the threshold, because every reported number is a hard threshold at
0.35 while training optimises a soft dice: part of an apparent fit deficit can
be calibration rather than capacity.

Decomposition runs at the model's input resolution (the space the model
actually decides in); the headline IoU column uses the evaluator's native-
resolution chain so it stays comparable to reported numbers.
"""
import argparse
import io
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from refmask2former import load_local_records, load_parquet_records
from refmask2former.dataset import (_normalize_chw, render_instance_mask,
                                    sample_reference_box)
from scripts.evaluate_refunet_selection import HOLDOUT, load_refunet

HOLDOUT_IDX = [int(x) for x in HOLDOUT.split(",")]


def decompose(pred, target, band, found_thresh):
    """pred/target: bool arrays at input resolution. Returns pixel counts."""
    out = {}
    fp = pred & ~target
    fn = target & ~pred
    out["tp"] = int((pred & target).sum())
    out["fp"] = int(fp.sum())
    out["fn"] = int(fn.sum())
    if not (fp.any() or fn.any()):
        return out | {k: 0 for k in ("boundary", "missed_region", "missed_inside",
                                     "false_region", "false_fringe")}

    t8 = target.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(t8, k, iterations=band).astype(bool)
    ero = cv2.erode(t8, k, iterations=band).astype(bool)
    boundary_band = dil & ~ero

    # False negatives, split by whether the GT component was found at all.
    n_gt, gt_lab = cv2.connectedComponents(t8)
    missed_region = np.zeros_like(target)
    for c in range(1, n_gt):
        comp = gt_lab == c
        if comp.sum() and (pred & comp).sum() / comp.sum() < found_thresh:
            missed_region |= comp
    fn_missed = fn & missed_region
    fn_rest = fn & ~missed_region

    # False positives, split by whether the predicted component is mostly
    # outside the target (a wrong region) or hugging one (spill).
    n_p, p_lab = cv2.connectedComponents(pred.astype(np.uint8))
    false_region = np.zeros_like(pred)
    for c in range(1, n_p):
        comp = p_lab == c
        if comp.sum() and (target & comp).sum() / comp.sum() < 0.5:
            false_region |= comp
    fp_false = fp & false_region
    fp_rest = fp & ~false_region

    # Boundary wins over the others: an edge pixel is an edge pixel whichever
    # side it fell on, and counting it twice would inflate the total.
    out["boundary"] = int(((fn_rest | fp_rest) & boundary_band).sum())
    out["missed_region"] = int(fn_missed.sum())
    out["missed_inside"] = int((fn_rest & ~boundary_band).sum())
    out["false_region"] = int(fp_false.sum())
    out["false_fringe"] = int((fp_rest & ~boundary_band).sum())
    return out


def selections_hf(records, indices, image_max_size, ref_size):
    for image_index in indices:
        rec = records[image_index]
        image0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        h0, w0 = image0.shape[:2]
        masks0 = [render_instance_mask(a["segmentation"], h0, w0).astype(bool)
                  for a in anns]
        cats = [a.get("category_name", "pattern") for a in anns]
        scale = image_max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
        image = cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)
        for ref_idx, cat in enumerate(cats):
            rng = random.Random(image_index * 1_000_003 + ref_idx * 65_537 + 12_345)
            x, y, w, h = sample_reference_box(masks0[ref_idx], 128, 512, rng=rng)
            crop = image0[y:y + h, x:x + w]
            if not crop.size:
                continue
            crop = cv2.resize(crop, (ref_size, ref_size), interpolation=cv2.INTER_LINEAR)
            target0 = np.logical_or.reduce([m for m, c in zip(masks0, cats) if c == cat])
            yield (f"{image_index}:{ref_idx}", image, crop, target0, (nh, nw))


def selections_local(records, limit, image_max_size, ref_size, per_image, seed):
    rng_pick = random.Random(seed)
    for rec in records[:limit]:
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        image0 = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
        h0, w0 = image0.shape[:2]
        cats = [a.get("category_name", "pattern") for a in anns]
        masks0 = [render_instance_mask(a["segmentation"], h0, w0).astype(bool)
                  for a in anns]
        if not masks0:
            continue
        scale = image_max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
        image = cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)
        picks = list(range(len(masks0)))
        rng_pick.shuffle(picks)
        name = os.path.basename(rec["image_path"])
        for ref_idx in picks[:per_image]:
            rng = random.Random(hash(name) % 10_000 * 1_000_003 + ref_idx)
            x, y, w, h = sample_reference_box(masks0[ref_idx], 128, 512, rng=rng)
            crop = image0[y:y + h, x:x + w]
            if not crop.size:
                continue
            crop = cv2.resize(crop, (ref_size, ref_size), interpolation=cv2.INTER_LINEAR)
            cat = cats[ref_idx]
            target0 = np.logical_or.reduce([m for m, c in zip(masks0, cats) if c == cat])
            yield (f"{name}:{ref_idx}", image, crop, target0, (nh, nw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", choices=["hf14", "val14", "local"], default="hf14")
    ap.add_argument("--local-data", default="data/synthetic/v6d_train2000")
    ap.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--per-image", type=int, default=1, help="selections/image, local")
    ap.add_argument("--image-max-size", type=int, default=2560)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--sweep", default="0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7")
    ap.add_argument("--band", type=int, default=4, help="boundary half-width, px")
    ap.add_argument("--found-thresh", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_refunet(a.checkpoint, device)
    sweep = [float(s) for s in a.sweep.split(",")]

    if a.source == "local":
        records = load_local_records(a.local_data)
        items = selections_local(records, a.n, a.image_max_size, a.ref_size,
                                 a.per_image, a.seed)
        label = f"{os.path.basename(a.local_data)} (<= {a.n} images)"
    else:
        records = load_parquet_records(a.hf_repo, cache_dir=a.cache_dir,
                                       config="real-world-test", split="test")
        idx = HOLDOUT_IDX if a.source == "hf14" else \
            [i for i in range(len(records)) if i not in HOLDOUT_IDX]
        items = selections_hf(records, idx, a.image_max_size, a.ref_size)
        label = f"{a.source}"

    totals = defaultdict(int)
    ious, rows = [], []
    sweep_iou = defaultdict(list)
    with torch.inference_mode():
        for key, image, crop, target0, (nh, nw) in items:
            img_t = _normalize_chw(image).unsqueeze(0).to(device)
            ref_t = _normalize_chw(crop).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                prob = model(img_t, ref_t).sigmoid()[0, 0].float()
            prob_np = prob.cpu().numpy()
            h0, w0 = target0.shape

            for t in sweep:
                p = cv2.resize((prob_np > t).astype(np.uint8), (w0, h0),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                sweep_iou[t].append(int((p & target0).sum())
                                    / max(int((p | target0).sum()), 1))

            pred_native = cv2.resize((prob_np > a.mask_thresh).astype(np.uint8),
                                     (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
            iou = int((pred_native & target0).sum()) / max(int((pred_native | target0).sum()), 1)
            ious.append(iou)

            target_small = cv2.resize(target0.astype(np.uint8), (nw, nh),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            parts = decompose(prob_np > a.mask_thresh, target_small,
                              a.band, a.found_thresh)
            for k, v in parts.items():
                totals[k] += v
            rows.append({"key": key, "iou": iou, **parts})
            print(f"  {key:26s} iou={iou:.4f} "
                  + " ".join(f"{k}={parts[k]}" for k in
                             ("boundary", "missed_region", "missed_inside",
                              "false_region", "false_fringe")), flush=True)

    err = sum(totals[k] for k in ("boundary", "missed_region", "missed_inside",
                                  "false_region", "false_fringe"))
    print(f"\n== residual decomposition: {label}, {len(rows)} selections, "
          f"input {a.image_max_size}, threshold {a.mask_thresh}")
    print(f"   checkpoint {os.path.basename(a.checkpoint)} "
          f"(model={ck.get('args',{}).get('model','unet')}, "
          f"trained res {ck.get('args',{}).get('image_max_size')})")
    print(f"   mean IoU {np.mean(ious):.4f}   median {np.median(ious):.4f}")
    print(f"   error pixels {err:,}  (tp {totals['tp']:,})")
    for k in ("boundary", "missed_region", "missed_inside", "false_region",
              "false_fringe"):
        print(f"     {k:14s} {totals[k]:12,}   {totals[k]/max(err,1)*100:5.1f}%")
    print("   threshold sweep (native-res mean IoU):")
    for t in sweep:
        print(f"     {t:.2f}  {np.mean(sweep_iou[t]):.4f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "totals": dict(totals), "rows": rows,
                   "sweep": {str(t): float(np.mean(v)) for t, v in sweep_iou.items()}},
                  open(a.out, "w"), indent=1)
        print(f"   wrote {a.out}")


if __name__ == "__main__":
    main()
