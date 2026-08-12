#!/usr/bin/env python3
"""Pairwise appearance similarity between pattern families.

The 2026-08-10 finding was that confusable materials, not localisation or
boundaries, are where the score is lost: every image scoring under 0.65 in
either split is multi-family, every single-family image scores 0.92+, and the
worst multi-family image in each split is the one holding a near-duplicate pair
(HF14 img 14 at 0.908 -> IoU 0.246; validation img 13 at 0.886 -> 0.385).

That measurement was described in `synth_progress.md` but its script was never
committed. This is it, so the numbers can be re-derived and so the same
similarity definition can be reused when generating confusable pairs.

Descriptor: ImageNet ResNet50 (V1 weights) global-average-pool features over
64px patches sampled inside each region, L2-normalised per patch, averaged per
family, renormalised. Similarity is cosine between family descriptors.

Modes
-----
--metrics JSON   reproduce the per-image corr(IoU, max inter-family sim) from an
                 `evaluate_refunet_selection.py` metrics file. Use this to check
                 the implementation against the documented -0.709 / -0.250
                 before trusting any similarity number it produces.
--tiles DIR      pairwise similarity over the curated generator tiles, i.e. does
                 the tile pool even contain confusable pairs to draw from.
--local-data DIR distribution of per-image max inter-family similarity over a
                 local-data pool. This is the measurement that matters for the
                 generator claim: the generator samples distinct tile *paths*,
                 which is not the same as distinct *appearances*, so whether
                 rendered synthetic sheets actually lack confusable pairs has to
                 be measured on the sheets rather than assumed.
"""

import argparse
import io
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchvision
from PIL import Image

PATCH = 64
N_PATCHES = 32
SEED = 0


def build_encoder(device):
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return model, mean, std


@torch.no_grad()
def describe(patches, model, mean, std, device):
    """[n, PATCH, PATCH, 3] uint8 -> one L2-normalised descriptor."""
    if len(patches) == 0:
        return None
    x = torch.from_numpy(np.stack(patches)).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    f = model(x)
    f = torch.nn.functional.normalize(f, dim=1)
    d = torch.nn.functional.normalize(f.mean(0), dim=0)
    return d.cpu().numpy()


def sample_patches_in_mask(image, mask, rng, n=N_PATCHES, patch=PATCH):
    """Random patches whose centre lies inside the mask."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    half = patch // 2
    out = []
    for _ in range(n * 8):
        if len(out) >= n:
            break
        i = rng.randrange(ys.size)
        cy, cx = int(ys[i]), int(xs[i])
        y0, x0 = cy - half, cx - half
        if y0 < 0 or x0 < 0 or y0 + patch > h or x0 + patch > w:
            continue
        out.append(image[y0:y0 + patch, x0:x0 + patch])
    if not out:                                  # region smaller than a patch
        y0 = max(0, min(h - patch, int(ys.mean()) - half))
        x0 = max(0, min(w - patch, int(xs.mean()) - half))
        crop = image[y0:y0 + patch, x0:x0 + patch]
        if crop.shape[0] == patch and crop.shape[1] == patch:
            out.append(crop)
    return out


def sample_patches_in_image(image, rng, n=N_PATCHES, patch=PATCH):
    """Random patches anywhere, for small standalone images such as tiles."""
    h, w = image.shape[:2]
    if h < patch or w < patch:
        scale = max(patch / h, patch / w)
        im = Image.fromarray(image).resize(
            (max(patch, int(round(w * scale))), max(patch, int(round(h * scale)))),
            Image.BICUBIC)
        image = np.asarray(im)
        h, w = image.shape[:2]
    return [image[y:y + patch, x:x + patch]
            for y, x in ((rng.randrange(h - patch + 1), rng.randrange(w - patch + 1))
                         for _ in range(n))]


def pairwise(descriptors):
    """{name: vec} -> sorted [(sim, a, b)] descending."""
    names = list(descriptors)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out.append((float(descriptors[names[i]] @ descriptors[names[j]]),
                        names[i], names[j]))
    out.sort(reverse=True)
    return out


# ------------------------------------------------------------------ real mode

def run_metrics(args, model, mean, std, device):
    from refmask2former import load_parquet_records
    from refmask2former.dataset import render_instance_mask

    metrics = json.loads(open(args.metrics).read())
    per_image = defaultdict(list)
    for row in metrics["selections"]:
        per_image[row["image_index"]].append(row["iou"])

    ds = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                              config="real-world-test", split="test")
    rows = []
    for image_idx in sorted(per_image):
        rec = ds[image_idx]
        image = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        h, w = image.shape[:2]

        families = defaultdict(list)
        for a in anns:
            families[a.get("category_name", "pattern")].append(
                render_instance_mask(a["segmentation"], h, w).astype(bool))

        rng = random.Random(SEED + image_idx)
        descs = {}
        intra = {}
        for name, masks in families.items():
            union = np.logical_or.reduce(masks)
            d = describe(sample_patches_in_mask(image, union, rng),
                         model, mean, std, device)
            if d is not None:
                descs[name] = d
            # Intra-family spread: how alike are two instances the labeller called
            # the SAME material? If a family's own instances agree less than it
            # agrees with a different family, appearance cannot decide the answer
            # for this image and no amount of data or texture feature will.
            if len(masks) >= 2:
                per_inst = [describe(sample_patches_in_mask(image, m, rng),
                                     model, mean, std, device) for m in masks]
                per_inst = [v for v in per_inst if v is not None]
                if len(per_inst) >= 2:
                    intra[name] = min(float(a @ b)
                                      for i, a in enumerate(per_inst)
                                      for b in per_inst[i + 1:])

        # Background: everything no family claims.
        all_fg = np.logical_or.reduce([np.logical_or.reduce(m)
                                       for m in families.values()])
        bg = describe(sample_patches_in_mask(image, ~all_fg, rng),
                      model, mean, std, device)

        iou = float(np.mean(per_image[image_idx]))
        pairs = pairwise(descs)
        max_inter = pairs[0][0] if pairs else float("nan")
        max_bg = (max(float(d @ bg) for d in descs.values())
                  if bg is not None and descs else float("nan"))
        rows.append({"image": image_idx, "iou": iou, "n_families": len(descs),
                     "max_inter": max_inter, "max_bg": max_bg,
                     "min_intra": min(intra.values()) if intra else float("nan"),
                     "top_pair": pairs[0][1:] if pairs else None})

    print(f"{'img':>4} {'IoU':>6} {'fams':>5} {'maxsim':>7} {'intra':>7} "
          f"{'margin':>7}  top pair")
    for r in rows:
        pair = f"{r['top_pair'][0]}/{r['top_pair'][1]}" if r["top_pair"] else "-"
        margin = r["min_intra"] - r["max_inter"]
        flag = "  <-- ill-posed" if margin < 0 else ""
        print(f"{r['image']:>4} {r['iou']:>6.3f} {r['n_families']:>5} "
              f"{r['max_inter']:>7.3f} {r['min_intra']:>7.3f} {margin:>+7.3f}  "
              f"{pair}{flag}")

    # An image is appearance-decidable only if every family's own instances agree
    # more than that family agrees with a different one.
    both = [r for r in rows if r["n_families"] > 1 and r["min_intra"] == r["min_intra"]]
    if both:
        bad = [r for r in both if r["min_intra"] < r["max_inter"]]
        print(f"\nmulti-family images with an intra-family measurement: {len(both)}")
        print(f"  intra < inter (target not decidable from appearance): "
              f"{len(bad)} ({len(bad) / len(both) * 100:.0f}%)")
        if bad:
            print(f"  their mean IoU: {np.mean([r['iou'] for r in bad]):.3f}   "
                  f"others: {np.mean([r['iou'] for r in both if r not in bad]):.3f}")

    multi = [r for r in rows if r["n_families"] > 1]
    if len(multi) > 2:
        iou = np.array([r["iou"] for r in multi])
        sim = np.array([r["max_inter"] for r in multi])
        bg = np.array([r["max_bg"] for r in multi])
        print(f"\nn multi-family = {len(multi)}")
        print(f"corr(IoU, max inter-family sim) = {np.corrcoef(iou, sim)[0, 1]:+.3f}")
        print(f"corr(IoU, max background sim)   = {np.corrcoef(iou, bg)[0, 1]:+.3f}")
    single = [r for r in rows if r["n_families"] == 1]
    if single:
        print(f"single-family images: n={len(single)} "
              f"min IoU={min(r['iou'] for r in single):.3f}")


# ----------------------------------------------------------------- tiles mode

def run_tiles(args, model, mean, std, device):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from generate_synthetic_v5 import load_curated_tiles

    tiles = load_curated_tiles(args.tiles)
    rng = random.Random(SEED)
    descs = {}
    for t in tiles:
        image = np.asarray(Image.open(t["path"]).convert("RGB"))
        descs[t["name"]] = describe(sample_patches_in_image(image, rng),
                                    model, mean, std, device)

    pairs = pairwise(descs)
    sims = np.array([p[0] for p in pairs])
    print(f"{len(tiles)} tiles, {len(pairs)} pairs")
    print(f"similarity: mean={sims.mean():.3f} median={np.median(sims):.3f} "
          f"p90={np.percentile(sims, 90):.3f} max={sims.max():.3f}")
    for thresh in (0.80, 0.85, 0.875, 0.90):
        n = int((sims >= thresh).sum())
        print(f"  pairs >= {thresh:.3f}: {n:>5}  ({n / len(pairs) * 100:.1f}%)")
    print("\nHF14 img14's confusable pair sits at 0.908, validation img13 at 0.886.")
    print(f"\ntop {args.top} most confusable tile pairs:")
    for sim, a, b in pairs[:args.top]:
        print(f"  {sim:.3f}  {a}\n         {b}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"tiles": [t["name"] for t in tiles],
                       "pairs": [{"sim": s, "a": a, "b": b} for s, a, b in pairs]},
                      f, indent=2)
        print(f"\nwrote {args.out}")


# ------------------------------------------------------------ local-data mode

def run_local(args, model, mean, std, device):
    import glob
    from generate_synthetic_v5 import rasterize_mask

    files = sorted(glob.glob(os.path.join(args.local_data, "annotations", "*.json")))
    pick = random.Random(SEED)
    if args.limit and len(files) > args.limit:
        files = pick.sample(files, args.limit)

    sims, gaps, rgb_gaps, n_multi = [], [], [], 0
    for k, path in enumerate(files):
        rec = json.loads(open(path).read())
        anns = rec["annotations"]
        families = defaultdict(list)
        for a in anns:
            families[a.get("category_name", "pattern")].append(a["segmentation"])
        if len(families) < 2:
            continue
        n_multi += 1
        image = np.asarray(Image.open(os.path.join(
            args.local_data, "images", rec["image"]["file_name"])).convert("RGB"))
        h, w = image.shape[:2]
        masks = {name: np.logical_or.reduce(
                     [rasterize_mask(s, h, w).astype(bool) for s in segs])
                 for name, segs in families.items()}

        # A fixed 64px patch covers very different fractions of a pattern on a
        # 4505px synthetic sheet and a 640px scraped plan, so pools are only
        # comparable at the resolution the model actually trains on.
        if args.resize:
            scale = args.resize / max(h, w)
            if scale < 1.0:
                nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
                image = np.asarray(Image.fromarray(image).resize((nw, nh),
                                                                 Image.BILINEAR))
                masks = {name: np.asarray(Image.fromarray(m.astype(np.uint8))
                                          .resize((nw, nh), Image.NEAREST)).astype(bool)
                         for name, m in masks.items()}

        rng = random.Random(SEED + k)
        descs = {}
        for name, union in masks.items():
            d = describe(sample_patches_in_mask(image, union, rng),
                         model, mean, std, device)
            if d is not None:
                descs[name] = d
        pairs = pairwise(descs)
        if pairs:
            sims.append(pairs[0][0])
            # Is the most texture-confusable pair still trivially separable by
            # fill brightness/colour? `cat_style` in generate_synthetic_v5.py
            # assigns each family an independent fill, so a synthetic pair can be
            # texture-confusable and still carry a colour cue real plans lack.
            _, a, b = pairs[0]
            ma, mb = masks[a], masks[b]
            if ma.any() and mb.any():
                lum = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                pa, pb = image[ma].astype(np.float32), image[mb].astype(np.float32)
                gaps.append(abs(float((pa * lum).sum(1).mean())
                                - float((pb * lum).sum(1).mean())))
                rgb_gaps.append(float(np.linalg.norm(pa.mean(0) - pb.mean(0))))

    sims = np.array(sims)
    gaps, rgb_gaps = np.array(gaps), np.array(rgb_gaps)
    label = args.label or args.local_data
    print(f"\n{label}")
    print(f"  images scanned={len(files)} multi-family={n_multi} scored={len(sims)}")
    if len(sims):
        print(f"  max inter-family sim: mean={sims.mean():.3f} "
              f"median={np.median(sims):.3f} p90={np.percentile(sims, 90):.3f} "
              f"max={sims.max():.3f}")
        for thresh in (0.85, 0.875, 0.90):
            print(f"    images with a pair >= {thresh:.3f}: "
                  f"{int((sims >= thresh).sum()):>4} / {len(sims)} "
                  f"({(sims >= thresh).mean() * 100:.1f}%)")
        # Among the texture-confusable ones, can fill brightness alone separate
        # the pair? If yes the model never has to learn texture discrimination.
        hard = sims >= args.confusable_thresh
        if hard.sum():
            g, c = gaps[hard], rgb_gaps[hard]
            print(f"  of the {int(hard.sum())} pairs >= {args.confusable_thresh:.3f}, "
                  f"brightness gap between the pair:")
            print(f"    luminance |delta|: mean={g.mean():.1f} median={np.median(g):.1f} "
                  f"p90={np.percentile(g, 90):.1f}")
            print(f"    mean-RGB distance: mean={c.mean():.1f} median={np.median(c):.1f}")
            for t in (10.0, 20.0, 40.0):
                print(f"    separable by luminance alone (|delta| > {t:>4.0f}): "
                      f"{(g > t).mean() * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", help="evaluate_refunet_selection.py metrics JSON")
    ap.add_argument("--tiles", help="curated tiles directory")
    ap.add_argument("--local-data", help="local-data pool directory")
    ap.add_argument("--limit", type=int, default=300,
                    help="sample this many images in --local-data mode (0 = all)")
    ap.add_argument("--label", help="display name for --local-data output")
    ap.add_argument("--resize", type=int, default=1280,
                    help="longest side before patch sampling, matching training "
                         "resolution (0 = native)")
    ap.add_argument("--confusable-thresh", type=float, default=0.875,
                    help="texture similarity at/above which a pair counts as "
                         "confusable (validation img13 sits at 0.886)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", help="write the tile pair table here")
    args = ap.parse_args()
    if not any((args.metrics, args.tiles, args.local_data)):
        ap.error("pass --metrics, --tiles or --local-data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = build_encoder(device)
    if args.metrics:
        run_metrics(args, model, mean, std, device)
    if args.tiles:
        run_tiles(args, model, mean, std, device)
    if args.local_data:
        run_local(args, model, mean, std, device)


if __name__ == "__main__":
    main()
