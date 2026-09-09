#!/usr/bin/env python3
"""Score the two-model subtractive pipeline against the TRUE holed annotation.

The single model predicts the outer ring with holes filled, and is scored
against that same filled ring -- which flatters it, because the labeller needs
the holed polygon. Measured on the 29 held-out sheets:

    scored against          mean IoU   >=0.8
    outer ring (reported)     0.8818   90.8%
    true holed annotation     0.8449   77.6%
    ... holed instances only  0.6782   14.3%

So holes cost 13.2 points end to end, and on a holed region the single model is
essentially unusable. This evaluates: region = outer_model - hole_model.

Reported at MASK level (what the two decoders produce) and at POLYGON level
(what the labeller actually receives: a regularized outer ring plus one
regularized ring per hole, which is exactly the Roboflow `remove` convention).

    PYTHONPATH=. python3 scripts/sam3_holes_eval.py \
        --outer data/runs/sam3_nodihedral_s7/best.pth \
        --holes data/runs/sam3_holes_s7/best.pth
"""
import argparse, json, os, sys
import numpy as np, cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask
from scripts.regularize_polygon import regularize
from scripts.sam3_region_model import RegionModel
from scripts.train_sam3_boxseg import _hole_union, jitter


def fill(poly, H, W):
    m = np.zeros((H, W), np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def hole_polygons(hole_mask, min_area, eps_frac=0.010):
    """One regularized ring per connected hole, dropping specks."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole_mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if stats[i][4] < min_area:
            continue
        p = regularize(lab == i, eps_frac=eps_frac)
        if p is not None and len(p) >= 3:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", default="data/runs/sam3_nodihedral_s7/best.pth")
    ap.add_argument("--holes", default="data/runs/sam3_holes_s7/best.pth")
    ap.add_argument("--data", default="data/sam3_ft/val.jsonl")
    ap.add_argument("--crop-zoom", type=float, default=2.0)
    ap.add_argument("--min-hole-frac", type=float, default=0.005,
                    help="drop a predicted hole smaller than this fraction of "
                         "the region; specks cost more to delete than they save")
    ap.add_argument("--jitter", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    om = RegionModel(checkpoint=args.outer, crop_zoom=args.crop_zoom)
    hm = RegionModel(checkpoint=args.holes, crop_zoom=args.crop_zoom)
    sheets = [json.loads(l) for l in open(args.data)]
    rng = np.random.default_rng(args.seed)

    res = {k: [] for k in ("base_mask", "sub_mask", "base_poly", "sub_poly")}
    holed_flag, n_pred_holes, n_true_holes = [], 0, 0
    for s in sheets:
        im = Image.open(s["image"]).convert("RGB")
        W, H = im.size
        boxes = [jitter(i["bbox_xyxy"], W, H, args.jitter, rng) for i in s["instances"]]
        om_masks = om.masks(im, boxes)
        hm_masks = hm.masks(im, boxes)
        for k, inst in enumerate(s["instances"]):
            truth = render_instance_mask(
                [inst["outer_polygon"]] + inst.get("hole_polygons", []), H, W).astype(bool)
            outer = om_masks[k]
            holes = hm_masks[k] & outer          # a hole only exists inside its region
            if holes.sum() < args.min_hole_frac * max(1, outer.sum()):
                holes = np.zeros_like(holes)
            sub = outer & ~holes

            res["base_mask"].append(_iou(outer, truth))
            res["sub_mask"].append(_iou(sub, truth))

            # polygon level: regularized outer, minus one regularized ring per hole
            po = regularize(outer)
            base_p = fill(po, H, W)
            sub_p = base_p.copy()
            hp = hole_polygons(holes, args.min_hole_frac * max(1, outer.sum()))
            for p in hp:
                sub_p &= ~fill(p, H, W)
            res["base_poly"].append(_iou(base_p, truth))
            res["sub_poly"].append(_iou(sub_p, truth))
            holed_flag.append(inst.get("n_holes", 0) > 0)
            n_pred_holes += len(hp); n_true_holes += inst.get("n_holes", 0)

    h = np.array(holed_flag)
    print(f"val {len(h)} instances ({int(h.sum())} holed)   crop_zoom={args.crop_zoom}")
    print(f"predicted hole rings: {n_pred_holes}   true: {n_true_holes}\n")
    print(f"{'':34}{'mean':>8}{'median':>9}{'>=0.8':>8}")
    print("-" * 59)
    for lbl, key in [("outer only, mask", "base_mask"), ("outer - holes, mask", "sub_mask"),
                     ("outer only, POLYGON", "base_poly"),
                     ("outer - holes, POLYGON", "sub_poly")]:
        v = np.array(res[key])
        print(f"{lbl:<34}{v.mean():8.4f}{np.median(v):9.4f}{(v>=.8).mean():8.1%}")
    print(f"\non the {int(h.sum())} holed instances (polygon level):")
    for lbl, key in [("  outer only", "base_poly"), ("  outer - holes", "sub_poly")]:
        v = np.array(res[key])[h]
        print(f"{lbl:<34}{v.mean():8.4f}{np.median(v):9.4f}{(v>=.8).mean():8.1%}")
    print(f"\non the {int((~h).sum())} solid instances (polygon level) -- "
          f"the regression risk:")
    for lbl, key in [("  outer only", "base_poly"), ("  outer - holes", "sub_poly")]:
        v = np.array(res[key])[~h]
        print(f"{lbl:<34}{v.mean():8.4f}{np.median(v):9.4f}{(v>=.8).mean():8.1%}")


def _iou(a, b):
    return float((a & b).sum()) / max(1, int((a | b).sum()))


if __name__ == "__main__":
    main()
