#!/usr/bin/env python3
"""Can a frozen backbone tell these materials apart at all?

The residual decomposition says 24-42% of the missing IoU is `false_region`:
the model selects a region whose material is not the reference's. That is a
DISCRIMINATION failure, and discrimination starts in the representation. This
measures it with no training whatsoever.

Within one plan, crop patches inside each labelled family. Embed every patch
with a frozen pretrained backbone. Same-family pairs should be more similar
than different-family pairs. The separability (ROC AUC over within-image pairs)
is an upper bound on how well ANY decoder reading those features can group
regions by material -- if the representation does not separate them, no
conditioning mechanism downstream can recover it.

Pairs are formed WITHIN an image only, because that is the task: `pattern1` in
one plan has no relation to `pattern1` in another (`PROJECT_UNDERSTANDING.md`).

    python3 scripts/material_separability.py --backbone resnet50 --source hf14
"""
import argparse
import io
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, ".")
from refmask2former import load_local_records, load_parquet_records
from refmask2former.dataset import (_normalize_chw, render_instance_mask,
                                    sample_reference_box)
from scripts.cache_backbone_features import StagedBackbone

HOLDOUT = [12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24]


def auc(pos, neg):
    """ROC AUC via the rank statistic; no sklearn dependency."""
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def patches_for_image(image0, masks0, cats, patch, per_family, rng):
    """Square crops guaranteed inside their family's mask."""
    by_family = {}
    for m, c in zip(masks0, cats):
        by_family.setdefault(c, []).append(m)
    out = []
    for fam, masks in by_family.items():
        got = 0
        for _ in range(per_family * 6):
            if got >= per_family:
                break
            m = masks[rng.randrange(len(masks))]
            x, y, w, h = sample_reference_box(m, patch, patch, rng=rng)
            crop = image0[y:y + h, x:x + w]
            if crop.size == 0 or min(crop.shape[:2]) < 8:
                continue
            out.append((fam, cv2.resize(crop, (224, 224),
                                        interpolation=cv2.INTER_LINEAR)))
            got += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--source", choices=["hf14", "val14", "local"], default="hf14")
    ap.add_argument("--local-data", default="data/synthetic/v6d_train2000")
    ap.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--n", type=int, default=60, help="images, local source")
    ap.add_argument("--patch", type=int, default=224)
    ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--levels", default="3", help="feature levels to pool, e.g. 2,3")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device("cuda")
    model = StagedBackbone(a.backbone, pretrained=True).to(device).eval()
    levels = [int(x) for x in a.levels.split(",")]
    rng = random.Random(a.seed)

    if a.source == "local":
        records = load_local_records(a.local_data)[:a.n]
        items = [(os.path.basename(r["image_path"]),
                  np.asarray(Image.open(r["image_path"]).convert("RGB")),
                  r["annotations"]) for r in records]
    else:
        ds = load_parquet_records(a.hf_repo, cache_dir=a.cache_dir,
                                  config="real-world-test", split="test")
        idx = HOLDOUT if a.source == "hf14" else \
            [i for i in range(len(ds)) if i not in HOLDOUT]
        items = []
        for i in idx:
            anns = ds[i]["annotations"]
            items.append((str(i),
                          np.asarray(Image.open(io.BytesIO(ds[i]["image"])).convert("RGB")),
                          json.loads(anns) if isinstance(anns, str) else anns))

    pos_all, neg_all, per_image = [], [], []
    with torch.inference_mode():
        for name, image0, anns in items:
            h0, w0 = image0.shape[:2]
            cats = [x.get("category_name", "pattern") for x in anns]
            if len(set(cats)) < 2:
                continue  # separability is undefined with one family
            masks0 = [render_instance_mask(x["segmentation"], h0, w0).astype(bool)
                      for x in anns]
            patches = patches_for_image(image0, masks0, cats, a.patch,
                                        a.per_family, rng)
            if len(patches) < 4:
                continue
            batch = torch.stack([_normalize_chw(p) for _, p in patches]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = model(batch)
            emb = torch.cat([feats[l].float().mean((-2, -1)) for l in levels], 1)
            emb = F.normalize(emb, dim=1)
            sim = (emb @ emb.T).cpu().numpy()
            fams = np.array([f for f, _ in patches])
            iu = np.triu_indices(len(fams), k=1)
            same = fams[iu[0]] == fams[iu[1]]
            s = sim[iu]
            p, n = s[same], s[~same]
            if len(p) and len(n):
                per_image.append({"image": name, "n_families": len(set(fams)),
                                  "auc": auc(p, n),
                                  "same_mean": float(p.mean()),
                                  "diff_mean": float(n.mean())})
                pos_all.append(p)
                neg_all.append(n)
            print(f"  {name:>22s} fams={len(set(fams)):2d} patches={len(fams):3d} "
                  f"auc={per_image[-1]['auc']:.3f} same={p.mean():+.3f} "
                  f"diff={n.mean():+.3f}", flush=True)

    p = np.concatenate(pos_all)
    n = np.concatenate(neg_all)
    macro = float(np.nanmean([r["auc"] for r in per_image]))
    print(f"\n== material separability: backbone {a.backbone}, {a.source}, "
          f"levels {levels}, patch {a.patch}")
    print(f"   images with >=2 families: {len(per_image)}")
    print(f"   same-family cos  {p.mean():+.4f}   different-family cos {n.mean():+.4f}"
          f"   margin {p.mean()-n.mean():+.4f}")
    print(f"   AUC pooled {auc(p, n):.4f}   per-image mean {macro:.4f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "auc_pooled": auc(p, n), "auc_macro": macro,
                   "same_mean": float(p.mean()), "diff_mean": float(n.mean()),
                   "per_image": per_image}, open(a.out, "w"), indent=1)
        print(f"   wrote {a.out}")


if __name__ == "__main__":
    main()
