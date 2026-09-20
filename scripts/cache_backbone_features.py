#!/usr/bin/env python3
"""Freeze the backbone, run it ONCE, keep the features.

Every architecture question the project has open -- decoder width, dense
correlation, cross-attention conditioning, an actual transformer decoder -- is
downstream of `RefUNet.features()`. The backbone is ~25.6M of 28.0M parameters
and essentially all of the per-step cost, so re-running it for every variant is
what makes an architecture sweep cost GPU-hours.

Cache `c1..c4` for the image and for the reference crop once, and every variant
afterwards trains against tensors already resident on the GPU: no image decode,
no dataloader, no backbone forward. A variant then costs seconds.

What this can and cannot answer. It holds the representation fixed, so it ranks
DECODERS and says whether a frozen ImageNet ResNet50 representation is
sufficient. It cannot evaluate a different backbone or the effect of
finetuning one. Survivors still need a real run.

Samples are padded to one square grid with a validity mask, exactly as
`collate_fn` pads batches, so variants can train batched.

    python3 scripts/cache_backbone_features.py \
        --local-data data/synthetic/v6d_train2000 --n 250 \
        --image-max-size 1024 --out data/cache/v6d_train_1024
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
from refmask2former.ref_unet import RefUNet


class StagedBackbone(torch.nn.Module):
    """Any of several pretrained backbones exposed as one 4-scale pyramid.

    ResNet50 (the incumbent), ConvNeXt (a modern CNN) and Swin (hierarchical
    attention) all emit strides 4/8/16/32, so the same decoder consumes any of
    them and a bake-off is apples-to-apples. Swin returns NHWC; it is permuted
    so every backbone hands back NCHW.
    """

    CHANNELS = {"resnet50": (256, 512, 1024, 2048),
                "convnext_base": (128, 256, 512, 1024),
                "convnext_small": (96, 192, 384, 768),
                "swin_b": (128, 256, 512, 1024),
                "swin_t": (96, 192, 384, 768)}

    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        self.name = name
        if name == "resnet50":
            self.net = RefUNet(width=128, pretrained=pretrained)
        else:
            import torchvision.models as tvm
            weights = "DEFAULT" if pretrained else None
            self.net = getattr(tvm, name)(weights=weights).features
            self.nhwc = name.startswith("swin")

    def forward(self, x):
        if self.name == "resnet50":
            return self.net.features(x)
        feats, t = [], x
        for i, block in enumerate(self.net):
            t = block(t)
            if i in (1, 3, 5, 7):
                feats.append(t.permute(0, 3, 1, 2).contiguous() if self.nhwc else t)
        return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local", "hf14", "val14"], default="local")
    ap.add_argument("--local-data", default="data/synthetic/v6d_train2000")
    ap.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--n", type=int, default=250, help="images to cache")
    ap.add_argument("--per-image", type=int, default=1)
    ap.add_argument("--image-max-size", type=int, default=1024)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backbone", default="resnet50",
                    choices=list(StagedBackbone.CHANNELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda")
    # pretrained=True: for resnet50 this is the representation every run in
    # this repo starts from; for the others it is their ImageNet checkpoint.
    model = StagedBackbone(a.backbone, pretrained=True).to(device).eval()

    if a.source == "local":
        records = load_local_records(a.local_data)
        HOLDOUT = None
    else:
        ds = load_parquet_records(a.hf_repo, cache_dir=a.cache_dir,
                                  config="real-world-test", split="test")
        hold = [12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24]
        idx = hold if a.source == "hf14" else [i for i in range(len(ds))
                                               if i not in hold]
        records = [{"hf_index": i, "image": ds[i]["image"],
                    "annotations": ds[i]["annotations"]} for i in idx]
        HOLDOUT = True
    rng_pick = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    G = a.image_max_size  # every sample is padded into a G x G input grid

    kept, meta = 0, []
    with torch.inference_mode():
        for rec in records[:a.n]:
            anns = rec["annotations"]
            if isinstance(anns, str):
                anns = json.loads(anns)
            if "image_path" in rec:
                image0 = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
                name = os.path.basename(rec["image_path"]).replace(".png", "")
            else:
                image0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
                name = str(rec["hf_index"])
            h0, w0 = image0.shape[:2]
            cats = [x.get("category_name", "pattern") for x in anns]
            masks0 = [render_instance_mask(x["segmentation"], h0, w0).astype(bool)
                      for x in anns]
            if not masks0:
                continue
            scale = G / max(h0, w0)
            nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
            image = cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)

            picks = list(range(len(masks0)))
            if HOLDOUT:
                # Real plans: keep ALL selections and the evaluator's own RNG,
                # so this cache is the 52 fixed questions, not a sample of them.
                pass
            else:
                rng_pick.shuffle(picks)
                picks = picks[:a.per_image]
            for ref_idx in picks:
                if HOLDOUT:
                    rng = random.Random(int(name) * 1_000_003 + ref_idx * 65_537
                                        + 12_345)
                else:
                    rng = random.Random(a.seed * 1_000_003 + kept)
                x, y, w, h = sample_reference_box(masks0[ref_idx], 128, 512, rng=rng)
                crop = image0[y:y + h, x:x + w]
                if not crop.size:
                    continue
                crop = cv2.resize(crop, (a.ref_size, a.ref_size),
                                  interpolation=cv2.INTER_LINEAR)
                cat = cats[ref_idx]
                target0 = np.logical_or.reduce(
                    [m for m, c in zip(masks0, cats) if c == cat])
                target = cv2.resize(target0.astype(np.uint8), (nw, nh),
                                    interpolation=cv2.INTER_NEAREST)

                img_pad = np.zeros((G, G, 3), image.dtype)
                img_pad[:nh, :nw] = image
                tgt_pad = np.zeros((G, G), np.uint8)
                tgt_pad[:nh, :nw] = target
                valid = np.zeros((G, G), np.uint8)
                valid[:nh, :nw] = 1

                img_t = _normalize_chw(img_pad).unsqueeze(0).to(device)
                ref_t = _normalize_chw(crop).unsqueeze(0).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fi = model(img_t)
                    fr = model(ref_t)
                sample = {
                    "image_feats": [f[0].to(torch.float16).cpu() for f in fi],
                    "ref_feats": [f[0].to(torch.float16).cpu() for f in fr],
                    "target": torch.from_numpy(tgt_pad).bool(),
                    "valid": torch.from_numpy(valid).bool(),
                    "key": f"{name}:{ref_idx}", "input_hw": [nh, nw],
                    "native_hw": [h0, w0],
                }
                torch.save(sample, os.path.join(a.out, f"{kept:05d}.pt"))
                meta.append({"idx": kept, "key": sample["key"],
                             "input_hw": [nh, nw], "native_hw": [h0, w0],
                             "target_px": int(target.sum())})
                kept += 1
            if kept and kept % 50 == 0:
                print(f"  cached {kept}", flush=True)

    shapes = [tuple(t.shape) for t in sample["image_feats"]]
    json.dump({"args": vars(a), "n": kept, "grid": G, "backbone": a.backbone,
               "channels": list(StagedBackbone.CHANNELS[a.backbone]),
               "feat_shapes": [list(s) for s in shapes], "items": meta},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    size = sum(os.path.getsize(os.path.join(a.out, f))
               for f in os.listdir(a.out)) / 1e9
    print(f"cached {kept} samples -> {a.out}  ({size:.1f} GB)")
    print(f"  feature shapes per sample: {shapes}")


if __name__ == "__main__":
    main()
