#!/usr/bin/env python3
"""Union IoU on synthetic plans the model never saw.

With a large single-pass pool the train/fresh gap separates two very different
limits: a LARGE gap means memorisation (data-bound), a SMALL gap with a low
absolute score means the model cannot fit its own training distribution
(capacity- or optimisation-bound). On 2026-09-18 the 100k RefUNet run scored
0.837 train / 0.798 fresh / 0.713 HF14 -- underfitting, not overfitting.
"""
import argparse, os, sys
import numpy as np, torch
from functools import partial
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from refmask2former import build_datasets, collate_fn, load_local_records
from refmask2former.ref_unet import RefUNet
from refmask2former.ref_attn_unet import RefCrossAttnUNet

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--all-pool", default="data/synthetic/v6d_4000")
ap.add_argument("--trained-pool", default="data/synthetic/v6d_100000")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--image-max-size", type=int, default=2048)
ap.add_argument("--mask-thresh", type=float, default=0.35)
a = ap.parse_args()

ck = torch.load(a.checkpoint, map_location="cpu")
args = ck.get("args", {})
width = args.get("width", 128)
model = (RefCrossAttnUNet(width, pretrained=False,
                          num_heads=args.get("attn_heads", 4))
         if args.get("model") == "crossattn" else RefUNet(width, pretrained=False))
model.load_state_dict(ck["model"]); model = model.cuda().eval()

used = set(os.listdir(f"{a.trained_pool}/annotations"))
recs = [r for r in load_local_records(a.all_pool)
        if os.path.basename(r["image_path"]).replace(".png", ".json") not in used]
print(f"unseen plans: {len(recs)}; evaluating {min(a.n, len(recs))}")
ds, _ = build_datasets(recs[:a.n], image_max_size=a.image_max_size, ref_size=224,
                       train_split=1.0, seed=7, domain_random=False)
dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=8,
                collate_fn=partial(collate_fn, size_divisible=512, union_only=True))
ious = []
with torch.no_grad():
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["images"].cuda(), b["references"].cuda())
        pred = torch.sigmoid(logits.squeeze(1).float()) > a.mask_thresh
        tgt = b["union"].squeeze(1).cuda().bool()
        inter = (pred & tgt).flatten(1).sum(1).float()
        union = (pred | tgt).flatten(1).sum(1).float()
        ious += (inter / union.clamp(min=1)).cpu().tolist()
print(f"model={args.get('model','unet')} width={width} "
      f"fresh-synthetic union IoU (n={len(ious)}): {np.mean(ious):.4f}")
