#!/usr/bin/env python3
"""Why does swin_t fit its own training plans at ~0.74 against a ~0.90 ceiling?

Scores plans with the HF14 protocol (every labelled instance is a question,
same box sampler, resize, threshold) like `question_difficulty.py`, but breaks
each question's error down so the hypotheses can be told apart:

  iou         native-resolution union IoU (the protocol number)
  ceil        IoU of the target pushed through the same resize chain
              (target -> input size -> native), i.e. the per-question cap
  iou_in      IoU at INPUT resolution against the resized target (no chain)
  fp_other    predicted input pixels inside OTHER labelled families / target px
  fp_bg       predicted input pixels on unlabelled paper / target px
  fn          target input pixels not predicted / target px
  wrong_regs  other-family instances more than half selected
  miss_regs   target instances less than half selected
  n_fams, n_target, target_frac, thick (target thickness at input, px: 2x the
  max of the distance transform)

Options that test hypotheses rather than just measure:
  --shrink s       extra downscale of the image (training's --domain-random
                   draws 0.4-1.0, so the model mostly saw ~0.7x of nominal)
  --ref-mode       'protocol' (evaluator's box), 'largest' (largest box in the
                   family's largest instance), 'foreign' (a box from a DIFFERENT
                   family -- how much the answer depends on the reference)
  --train-only     restrict to the checkpoint's own training split (seed/split
                   read from the checkpoint args)

    PYTHONPATH=. python3 scripts/fit_diagnose.py --checkpoint ck.pth \\
        --pool data/synthetic/revit_train2000 --n-images 150 --train-only \\
        --out data/evaluations/diag/seen.json
"""
import argparse
import json
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from refmask2former import load_local_records                                   # noqa: E402
from refmask2former.dataset import _normalize_chw, render_instance_mask, sample_reference_box  # noqa: E402
from scripts.evaluate_refunet_selection import load_refunet                    # noqa: E402


def train_indices(n, seed, split):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return set(idx[:int(n * split)])


def largest_box(mask):
    m = mask.astype(np.uint8)
    dt = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_C, 3)[1:-1, 1:-1]
    half = min(int(dt.max()) - 1, 255)
    if half < 0:
        return None
    cy, cx = np.argwhere(dt >= half + 1)[0]
    return int(cx - half), int(cy - half), 2 * half + 1, 2 * half + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n-images", type=int, default=150)
    ap.add_argument("--image-max-size", type=int, default=2048)
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--ref-mode", choices=("protocol", "largest", "foreign"), default="protocol")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--save-probs", default="", help="dir: save input-res prob maps (npz) per question")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    model, ck = load_refunet(a.checkpoint, device)
    cargs = ck.get("args", {})
    recs = load_local_records(a.pool)
    ids = list(range(len(recs)))
    if a.train_only:
        tr = train_indices(len(recs), int(cargs.get("seed", 31)), float(cargs.get("train_split", 0.99)))
        ids = [i for i in ids if i in tr]
    ids = ids[:a.n_images]
    rows = []
    with torch.inference_mode():
        for ii in ids:
            rec = recs[ii]
            image0 = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
            anns = rec["annotations"]
            h0, w0 = image0.shape[:2]
            masks0 = [render_instance_mask(x["segmentation"], h0, w0).astype(bool) for x in anns]
            cats = [x.get("category_name", "pattern") for x in anns]
            if not masks0:
                continue
            scale = a.image_max_size / max(h0, w0) * a.shrink
            nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
            image = cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)
            it = _normalize_chw(image).unsqueeze(0).to(device)
            masks_in = [cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
                        for m in masks0]
            labelled = np.logical_or.reduce(masks_in)
            fams = sorted(set(cats))
            # backbone once per plan
            feats = model.features(it) if hasattr(model, "features") else None
            for ref_idx, (ref_mask, cat) in enumerate(zip(masks0, cats)):
                rng = random.Random(ii * 1_000_003 + ref_idx * 65_537 + 12_345)
                box = sample_reference_box(ref_mask, 128, 512, rng=rng)
                if a.ref_mode == "largest":
                    fam_idx = [j for j, c in enumerate(cats) if c == cat]
                    big = max(fam_idx, key=lambda j: masks0[j].sum())
                    box = largest_box(masks0[big]) or box
                elif a.ref_mode == "foreign":
                    others = [j for j, c in enumerate(cats) if c != cat]
                    if not others:
                        continue
                    j = rng.choice(others)
                    box = sample_reference_box(masks0[j], 128, 512, rng=rng)
                x, y, w, h = box
                crop = cv2.resize(image0[y:y + h, x:x + w], (224, 224), interpolation=cv2.INTER_LINEAR)
                ref = _normalize_chw(crop).unsqueeze(0).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rf = model.features(ref)
                    protos = [f.mean((-2, -1)) for f in rf]
                    if getattr(model, "roi_ref", False):
                        bm = np.zeros((nh, nw), np.float32)
                        x0, y0 = int(x * scale), int(y * scale)
                        bm[y0:max(y0 + 1, int(round((y + h) * scale))),
                           x0:max(x0 + 1, int(round((x + w) * scale)))] = 1
                        protos = model.roi_prototypes(feats, torch.from_numpy(bm)[None, None].to(device), protos)
                    logits = model.decode(feats, protos, rf)
                    prob = torch.nn.functional.interpolate(logits.float(), size=(nh, nw), mode="bilinear",
                                                           align_corners=False).sigmoid()[0, 0]
                prob = prob.cpu().numpy()
                pred_in = prob > a.mask_thresh
                tgt_idx = [j for j, c in enumerate(cats) if c == cat]
                target0 = np.logical_or.reduce([masks0[j] for j in tgt_idx])
                target_in = np.logical_or.reduce([masks_in[j] for j in tgt_idx])
                pred0 = cv2.resize(pred_in.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
                back0 = cv2.resize(target_in.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
                iou = (pred0 & target0).sum() / max((pred0 | target0).sum(), 1)
                ceil = (back0 & target0).sum() / max((back0 | target0).sum(), 1)
                tp = target_in.sum()
                other = labelled & ~target_in
                wrong = sum(1 for j, c in enumerate(cats) if c != cat and masks_in[j].sum() > 0
                            and (pred_in & masks_in[j]).sum() / masks_in[j].sum() > 0.5)
                miss = sum(1 for j in tgt_idx if masks_in[j].sum() > 0
                           and (pred_in & masks_in[j]).sum() / masks_in[j].sum() < 0.5)
                dt = cv2.distanceTransform(np.pad(target_in.astype(np.uint8), 1), cv2.DIST_L2, 3)
                rows.append({
                    "image": ii, "ref": ref_idx, "cat": cat, "iou": float(iou), "ceil": float(ceil),
                    "iou_in": float((pred_in & target_in).sum() / max((pred_in | target_in).sum(), 1)),
                    "fp_other": float((pred_in & other).sum() / max(tp, 1)),
                    "fp_bg": float((pred_in & ~labelled).sum() / max(tp, 1)),
                    "fn": float((target_in & ~pred_in).sum() / max(tp, 1)),
                    "wrong_regs": wrong, "miss_regs": miss, "n_fams": len(fams), "n_target": len(tgt_idx),
                    "n_other": len(cats) - len(tgt_idx),
                    "target_frac": float(target0.mean()), "thick": float(2 * dt.max()),
                    "box": [x, y, w, h], "mean_p_target": float(prob[target_in].mean()) if tp else 0.0,
                    "mean_p_other": float(prob[other].mean()) if other.any() else 0.0,
                    "mean_p_bg": float(prob[~labelled].mean()) if (~labelled).any() else 0.0})
                if a.save_probs:
                    import os
                    os.makedirs(a.save_probs, exist_ok=True)
                    np.savez_compressed(f"{a.save_probs}/{ii}_{ref_idx}.npz",
                                        prob=(prob * 255).astype(np.uint8))
    json.dump({"checkpoint": a.checkpoint, "pool": a.pool, "args": vars(a), "rows": rows}, open(a.out, "w"))
    m = lambda k: np.mean([r[k] for r in rows])
    print(f"{len(rows)} q  iou {m('iou'):.4f}  ceil {m('ceil'):.4f}  iou_in {m('iou_in'):.4f}  "
          f"fp_other {m('fp_other'):.3f} fp_bg {m('fp_bg'):.3f} fn {m('fn'):.3f}  "
          f"wrong/q {m('wrong_regs'):.2f} miss/q {m('miss_regs'):.2f}")


if __name__ == "__main__":
    main()
