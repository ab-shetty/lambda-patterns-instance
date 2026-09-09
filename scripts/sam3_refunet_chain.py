#!/usr/bin/env python3
"""Chain the labelling-assist model to RefUNet: one box -> a whole pattern family.

Today the labeller draws one rough box per region and SAM 3 traces it. On the
held-out sheets 76% of instances belong to a family of more than one, and one
sheet carries a family of 15 -- so the boxes are the cost, not the tracing.

The chain uses each model for the thing it is good at:

  1. user's rough box  -> SAM 3 (crop_zoom)   -> mask + polygon of THAT region
  2. that mask         -> sample_reference_box -> a reference crop INSIDE it
  3. reference crop    -> RefUNet              -> union of every matching region
  4. each union component's bbox -> SAM 3      -> a clean polygon per region

Step 2 matters and is easy to get wrong. RefUNet's reference is a texture patch
sampled *inside* a region; the labeller's gesture is a box drawn *around* one,
which on an L-shaped region is mostly background. Feeding the raw user box to
RefUNet evaluates it on an input it was never trained on. Taking the reference
from SAM 3's predicted mask gives each model its native input, and costs nothing
because that mask was computed in step 1 anyway.

Scored as LABELLING COST, not as IoU: how many good polygons does one box buy,
and what has to be fixed afterwards.

    PYTHONPATH=. python3 scripts/sam3_refunet_chain.py \
        --refunet data/runs/ck_refunet_real165_s7/epoch_8.pth \
        --sam3 data/runs/sam3_nodihedral_s7/best.pth
"""
import argparse, json, os, random, sys
import numpy as np, cv2, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import (_normalize_chw, render_instance_mask,
                                    sample_reference_box)
from refmask2former.ref_unet import RefUNet
from scripts.sam3_region_model import RegionModel


def load_refunet(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get("args", {}) or {}
    m = RefUNet(width=int(a.get("width", 128)), pretrained=False,
                anchor=bool(a.get("anchor", False)))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), a


@torch.no_grad()
def refunet_union(model, image0, ref_crop, device, image_max_size=1280,
                  ref_size=224, thresh=0.35):
    h0, w0 = image0.shape[:2]
    s = image_max_size / max(h0, w0)
    nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
    img = _normalize_chw(cv2.resize(image0, (nw, nh),
                                    interpolation=cv2.INTER_LINEAR)).unsqueeze(0).to(device)
    ref = _normalize_chw(cv2.resize(ref_crop, (ref_size, ref_size),
                                    interpolation=cv2.INTER_LINEAR)).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prob = model(img, ref).sigmoid().float()[0, 0]
    small = (prob > thresh).cpu().numpy().astype(np.uint8)
    return cv2.resize(small, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)


def components(mask, min_area=400, topk=0):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area or w < 8 or h < 8:
            continue
        out.append(([int(x), int(y), int(x + w - 1), int(y + h - 1)], int(a)))
    out = sorted(out, key=lambda t: -t[1])
    return out[:topk] if topk else out


def iou(a, b):
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def poly_mask(p, H, W):
    m = np.zeros((H, W), np.uint8)
    if p is None or len(p) < 3:
        return m.astype(bool)
    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refunet", default=None)
    ap.add_argument("--oracle", action="store_true",
                    help="replace RefUNet with the ground-truth family union -- "
                         "measures the CEILING of the chain, i.e. what a perfect "
                         "matcher would buy once components are split into boxes")
    ap.add_argument("--sam3", default=None, help="default = the published decoder")
    ap.add_argument("--data", default="data/sam3_ft/val.jsonl")
    ap.add_argument("--crop-zoom", type=float, default=2.0)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--image-max-size", type=int, default=1280)
    ap.add_argument("--accept", type=float, default=0.8,
                    help="polygon IoU at which a proposal is 'accept with a nudge'")
    ap.add_argument("--max-family", type=int, default=40)
    ap.add_argument("--min-frac", type=float, default=0.25,
                    help="drop a component smaller than this fraction of the "
                         "prompted region's area. This is the filter that makes "
                         "the chain pay: it cuts false proposals 4.7 -> 1.0 per "
                         "box for ~0.03 of recall, because RefUNet's spurious "
                         "fragments are small ones.")
    ap.add_argument("--topk", type=int, default=8,
                    help="keep at most this many proposals, largest first")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.oracle and not args.refunet:
        sys.exit("pass --refunet CHECKPOINT, or --oracle for the ceiling")
    ref_model, ra = (None, {}) if args.oracle else load_refunet(args.refunet, dev)
    rm = RegionModel(checkpoint=args.sam3, crop_zoom=args.crop_zoom)
    sheets = [json.loads(l) for l in open(args.data)]
    print(f"matcher: {'ORACLE (gt family union)' if args.oracle else args.refunet}\n"
          f"sam3: {args.sam3 or 'published'} "
          f"crop_zoom={args.crop_zoom}\nsheets: {len(sheets)}\n")

    rows = []
    for si, s in enumerate(sheets):
        image0 = cv2.cvtColor(cv2.imread(s["image"]), cv2.COLOR_BGR2RGB)
        H, W = image0.shape[:2]
        gts = [render_instance_mask([i["outer_polygon"]], H, W).astype(bool)
               for i in s["instances"]]
        cats = [i["category_name"] for i in s["instances"]]
        for k, inst in enumerate(s["instances"]):
            fam = [j for j, c in enumerate(cats) if c == cats[k]]
            if len(fam) > args.max_family:
                continue
            # 1. the user's box -> SAM 3
            m1 = rm.masks(_pil(image0), [inst["bbox_xyxy"]])[0]
            if m1.sum() < 100:
                continue
            # 2. a reference crop INSIDE that predicted mask
            try:
                x, y, w, h = sample_reference_box(m1, 128, 512,
                                                  rng=random.Random(si * 7919 + k))
            except Exception:
                continue
            crop = image0[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            # 3. RefUNet -> union of matching regions
            if args.oracle:
                union = np.logical_or.reduce([gts[j] for j in fam])
            else:
                union = refunet_union(ref_model, image0, crop, dev,
                                      args.image_max_size,
                                      int(ra.get("ref_size", 224)),
                                      args.mask_thresh)
            # 4. components -> SAM 3 -> polygons
            comps = components(union, max(400, int(args.min_frac * m1.sum())),
                               args.topk)
            boxes = [b for b, _ in comps][:args.max_family]
            polys = ([_poly(rm, image0, b) for b in boxes] if boxes else [])
            preds = [poly_mask(p, H, W) for p in polys if p is not None]
            preds = [p for p in preds if p.sum() >= 100]
            # the region the user pointed at is always delivered by step 1
            p1 = poly_mask(rm.polygons(_pil(image0), [inst["bbox_xyxy"]])[0], H, W)
            preds = [p1] + preds

            # greedy match: each GT family member to its best unused proposal
            used, found = set(), 0
            per = []
            for j in fam:
                best, bi = 0.0, -1
                for pi_, pm in enumerate(preds):
                    if pi_ in used:
                        continue
                    v = iou(pm, gts[j])
                    if v > best:
                        best, bi = v, pi_
                if bi >= 0 and best >= args.accept:
                    used.add(bi); found += 1
                per.append(best)
            fp = len(preds) - len(used)
            rows.append(dict(sheet=si, inst=k, family=len(fam), found=found,
                             proposals=len(preds), fp=fp,
                             best_iou=float(np.mean(per)) if per else 0.0))

    if not rows:
        print("no rows"); return
    fam = np.array([r["family"] for r in rows])
    found = np.array([r["found"] for r in rows])
    fp = np.array([r["fp"] for r in rows])
    multi = fam > 1
    print(f"{'':22} {'all prompts':>13} {'families >1':>13}")
    print("-" * 50)
    def line(lbl, v):
        print(f"{lbl:22} {v[np.ones_like(fam,bool)].mean():13.3f} "
              f"{v[multi].mean() if multi.any() else float('nan'):13.3f}")
    line("family size", fam.astype(float))
    line("regions found (>=%.1f)" % args.accept, found.astype(float))
    line("false proposals", fp.astype(float))
    rec = found / np.maximum(1, fam)
    line("family recall", rec)
    # labelling cost: baseline draws one box per member; the chain draws one.
    saved = np.maximum(0, found - 1)
    line("boxes saved / prompt", saved.astype(float))
    print(f"\nprompts: {len(rows)}  ({int(multi.sum())} on families >1)")
    print(f"baseline cost: {fam.sum()} boxes for {fam.sum()} regions")
    print(f"chain cost   : {len(rows)} boxes -> {found.sum()} regions at >={args.accept} IoU, "
          f"{fp.sum()} to delete")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"-> {args.out}")


def _pil(arr):
    from PIL import Image
    return Image.fromarray(arr)


def _poly(rm, image0, box):
    p = rm.polygons(_pil(image0), [box])
    return p[0] if p else None


if __name__ == "__main__":
    main()
