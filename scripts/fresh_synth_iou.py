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
from refmask2former.ref_swin_unet import RefSwinUNet
from refmask2former.ref_dino_unet import BACKBONES as DINO_BACKBONES
from refmask2former.ref_dino_unet import RefDinoUNet

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--all-pool", default="data/synthetic/v6d_4000")
ap.add_argument("--trained-pool", default="data/synthetic/v6d_100000")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--image-max-size", type=int, default=2048)
ap.add_argument("--mask-thresh", type=float, default=0.35)
ap.add_argument("--seen", action="store_true",
                help="evaluate the records the model DID train on (hard train "
                     "IoU). Every other train IoU in this repo is soft-dice "
                     "derived and not comparable to it.")
a = ap.parse_args()

ck = torch.load(a.checkpoint, map_location="cpu")
args = ck.get("args", {})
width = args.get("width", 128)
_m = args.get("model", "unet")
if _m == "crossattn":
    model = RefCrossAttnUNet(width, pretrained=False,
                             num_heads=args.get("attn_heads", 4))
elif str(_m) in DINO_BACKBONES:
    model = RefDinoUNet(width, pretrained=False, backbone=str(_m))
elif str(_m).startswith("swin"):
    model = RefSwinUNet(width, pretrained=False, backbone=str(_m))
else:
    model = RefUNet(width, pretrained=False)
model.load_state_dict(ck["model"]); model = model.cuda().eval()

used = set(os.listdir(f"{a.trained_pool}/annotations"))
recs = [r for r in load_local_records(a.all_pool)
        if (os.path.basename(r["image_path"]).replace(".png", ".json") in used)
        == bool(a.seen)]
print(f"{'SEEN (train)' if a.seen else 'unseen'} plans: {len(recs)}; "
      f"evaluating {min(a.n, len(recs))}")
ds, _ = build_datasets(recs[:a.n], image_max_size=a.image_max_size, ref_size=224,
                       train_split=1.0, seed=7, domain_random=False)
dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=8,
                collate_fn=partial(collate_fn, size_divisible=512, union_only=True))
# Two numbers, because they can differ enormously. `collate_fn` pads every
# batch to a common size and the padding is ~40-45% of the canvas here. Scoring
# it asks the model a question about zero pixels that no user ever asks, and
# backbones answer it very differently: measured 2026-09-20 on the 100k pool,
# mean P(padding) is 0.071 for swin_t but 0.422 for dinov3_s -- a plain ViT's
# global attention assigns meaning to uniform black where windowed attention
# cannot. That alone moved dinov3_s from 0.86 to 0.33 and made it look like the
# worst arm on the board when on real pixels it ties the best.
#   full  -- the legacy number, scored over the padded canvas. Every hard train
#            IoU recorded before 2026-09-20 (the 0.8022 reference included) is
#            this one, so it is kept for comparability.
#   valid -- scored only where `pixel_mask` says real image lives. This is the
#            honest number and the one to quote for new work.
ious, ious_valid = [], []
with torch.no_grad():
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["images"].cuda(), b["references"].cuda())
        prob = torch.sigmoid(logits.squeeze(1).float())
        pred = prob > a.mask_thresh
        tgt = b["union"].squeeze(1).cuda().bool()
        inter = (pred & tgt).flatten(1).sum(1).float()
        union = (pred | tgt).flatten(1).sum(1).float()
        ious += (inter / union.clamp(min=1)).cpu().tolist()
        vm = b["pixel_mask"].cuda().bool()
        pv, tv = pred & vm, tgt & vm
        iv = (pv & tv).flatten(1).sum(1).float()
        uv = (pv | tv).flatten(1).sum(1).float()
        ious_valid += (iv / uv.clamp(min=1)).cpu().tolist()
print(f"  valid-pixel (pixel_mask) union IoU: "
      f"{sum(ious_valid)/max(1,len(ious_valid)):.4f}   <- quote this one")
print(f"model={args.get('model','unet')} width={width} "
      f"{'TRAIN (hard)' if a.seen else 'fresh-synthetic'} union IoU "
      f"(n={len(ious)}): {np.mean(ious):.4f}")
