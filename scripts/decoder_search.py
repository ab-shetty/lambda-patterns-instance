#!/usr/bin/env python3
"""Rank reference-conditioned decoders against cached backbone features.

`cache_backbone_features.py` froze the ResNet50 and stored `c1..c4` for image
and reference. Everything here trains only the part downstream of that, so a
variant costs seconds instead of an hour and the whole sweep fits between two
sips of coffee.

What each variant is testing:

  baseline    RefUNet's own decode path (`ConditionBlock` x4: a globally
              average-pooled reference vector broadcast over the image, fused
              by convs). The thing every result in this repo used.
  w256/w384   Same mechanism, more channels -- the decoder-capacity question
              `run_capacity_probe.sh` was written to ask, minus the backbone
              cost, plus a valid hard-threshold metric.
  deep        Same width, more depth. Separates "too few channels" from "too
              few layers".
  corr4       Baseline + dense correlation: the reference kept as 4x4 spatial
              tokens, every image location cosine-matched against all of them.
  crossattn   Multi-head cross-attention conditioning at the two coarsest
              scales (image tokens query reference tokens), as
              `ref_attn_unet.py` does.
  xattn_all   Cross-attention at ALL four scales.
  selfattn    Cross-attention conditioning PLUS self-attention over image
              tokens at 1/32 and 1/16. This is the one the repo has never
              tested: `RefCrossAttnUNet` is not a transformer and adds no
              capacity (27.9M vs RefUNet's 28.0M), so "a transformer might fix
              the long-range failures" has never actually been tried.

The reported metric is hard union IoU at the training threshold, measured on
the cached TRAIN split (can this decoder fit at all?) and on a held-out cached
split (does the fit transfer within the same distribution?), and -- with
`--hf14-cache` -- on the 52 real selections, which is the transfer that
matters. The printed `ceiling` is the NAIVE area-pooled bound: a floor on what
is reachable, not a cap. A good decoder legitimately exceeds it by finding a
sharper stride-4 encoding than area-pooling (`label_ceiling.py --strides`
optimises the grid properly; on the 1024 probe pool naive reads 0.9108 against
a true 0.9886).

Frozen backbone: this ranks decoders, and -- pointed at caches built by
`cache_backbone_features.py --backbone ...` -- ranks backbones too, since
ResNet50, ConvNeXt and Swin all expose the same stride-4/8/16/32 pyramid. It
cannot measure the effect of FINETUNING a backbone, which production does at
`backbone_lr_mult 0.1`. Survivors need a real run.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from refmask2former.ref_unet import ConditionBlock

CHANNELS = (256, 512, 1024, 2048)


class CrossAttnCondition(nn.Module):
    """Image locations query the reference's spatial tokens."""

    def __init__(self, image_channels, ref_channels, width, num_heads=4,
                 self_attn=False):
        super().__init__()
        self.width, self.num_heads = width, num_heads
        self.image = nn.Conv2d(image_channels, width, 1)
        self.reference = nn.Linear(ref_channels, width)
        self.kv = nn.Conv2d(ref_channels, width * 2, 1)
        self.attn_out = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.self_attn = self_attn
        if self_attn:
            self.qkv_self = nn.Conv2d(width, width * 3, 1)
            self.self_out = nn.Conv2d(width, width, 1)
            self.self_norm = nn.GroupNorm(8, width)
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU())

    def _heads(self, t, b, n):
        return t.reshape(b, self.num_heads, self.width // self.num_heads, n)

    def forward(self, image, reference_vector, reference_features):
        x = self.image(image)
        b, c, h, w = x.shape
        # Reference tokens are pooled to a small grid: attention cost then does
        # not depend on the reference crop's size.
        ref = F.adaptive_avg_pool2d(reference_features, 7)
        k, v = self.kv(ref).chunk(2, dim=1)
        q = self._heads(x.flatten(2), b, h * w)
        k = self._heads(k.flatten(2), b, 49)
        v = self._heads(v.flatten(2), b, 49)
        a = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1),
                                           v.transpose(-2, -1))
        a = a.transpose(-2, -1).reshape(b, c, h, w)
        x = self.norm(x + self.attn_out(a))
        if self.self_attn:
            q2, k2, v2 = self.qkv_self(x).chunk(3, dim=1)
            n = h * w
            q2 = self._heads(q2.flatten(2), b, n).transpose(-2, -1)
            k2 = self._heads(k2.flatten(2), b, n).transpose(-2, -1)
            v2 = self._heads(v2.flatten(2), b, n).transpose(-2, -1)
            s = F.scaled_dot_product_attention(q2, k2, v2)
            s = s.transpose(-2, -1).reshape(b, c, h, w)
            x = self.self_norm(x + self.self_out(s))
        # Keep the broadcast prototype alongside the attended features, so this
        # differs from `ConditionBlock` only by ADDING attention -- otherwise a
        # win could just be the missing global term.
        r = self.reference(reference_vector)[:, :, None, None].expand_as(x)
        return self.fuse(torch.cat([x, r], dim=1))


class Decoder(nn.Module):
    """One reference-conditioned decoder over frozen c1..c4."""

    def __init__(self, width=128, corr_grid=0, depth=2, attn_levels=(),
                 self_attn_levels=(), num_heads=4, channels=CHANNELS):
        super().__init__()
        self.attn_levels = set(attn_levels)
        blocks = []
        for i, c in enumerate(channels):
            if i in self.attn_levels:
                blocks.append(CrossAttnCondition(c, c, width, num_heads,
                                                 self_attn=i in set(self_attn_levels)))
            else:
                blocks.append(ConditionBlock(c, c, width,
                                             corr_grid=(0 if i == 0 else corr_grid)))
        self.condition = nn.ModuleList(blocks)
        def smooth_stack():
            layers = []
            for _ in range(depth):
                layers += [nn.Conv2d(width, width, 3, padding=1, bias=False),
                           nn.GroupNorm(8, width), nn.GELU()]
            return nn.Sequential(*layers)
        self.smooth = nn.ModuleList(smooth_stack() for _ in range(3))
        head = []
        for _ in range(depth):
            head += [nn.Conv2d(width, width, 3, padding=1, bias=False),
                     nn.GroupNorm(8, width), nn.GELU()]
        head += [nn.Conv2d(width, 1, 1)]
        self.head = nn.Sequential(*head)

    def forward(self, image_feats, ref_feats):
        prototypes = [f.mean((-2, -1)) for f in ref_feats]
        conditioned = []
        for i, block in enumerate(self.condition):
            if i in self.attn_levels:
                conditioned.append(block(image_feats[i], prototypes[i], ref_feats[i]))
            else:
                conditioned.append(block(image_feats[i], prototypes[i], ref_feats[i]))
        pyramid = conditioned[-1]
        for level in range(2, -1, -1):
            pyramid = F.interpolate(pyramid, size=conditioned[level].shape[-2:],
                                    mode="bilinear", align_corners=False)
            pyramid = self.smooth[level](pyramid + conditioned[level])
        return self.head(pyramid)


VARIANTS = {
    "baseline":  dict(width=128),
    "w256":      dict(width=256),
    "w384":      dict(width=384),
    "deep":      dict(width=128, depth=4),
    "corr4":     dict(width=128, corr_grid=4),
    "crossattn": dict(width=128, attn_levels=(2, 3)),
    "xattn_all": dict(width=128, attn_levels=(0, 1, 2, 3)),
    "selfattn":  dict(width=128, attn_levels=(2, 3), self_attn_levels=(2, 3)),
}


def load_cache(path, limit, device):
    man = json.load(open(os.path.join(path, "manifest.json")))
    n = min(limit, man["n"])
    channels = tuple(man.get("channels", CHANNELS))
    out = []
    for i in range(n):
        s = torch.load(os.path.join(path, f"{i:05d}.pt"), map_location="cpu")
        out.append({
            "image_feats": [f.to(device, torch.bfloat16) for f in s["image_feats"]],
            "ref_feats": [f.to(device, torch.bfloat16) for f in s["ref_feats"]],
            "target": s["target"].to(device),
            "valid": s["valid"].to(device),
            "key": s["key"],
        })
    return out, channels


def batch(samples, idx, device):
    image_feats = [torch.stack([samples[i]["image_feats"][l] for i in idx])
                   for l in range(4)]
    ref_feats = [torch.stack([samples[i]["ref_feats"][l] for i in idx])
                 for l in range(4)]
    target = torch.stack([samples[i]["target"] for i in idx]).float()[:, None]
    valid = torch.stack([samples[i]["valid"] for i in idx]).float()[:, None]
    return image_feats, ref_feats, target, valid


def loss_fn(logits, target, valid, bce_weight=1.0, dice_weight=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = logits.sigmoid() * valid
    t = target * valid
    inter = (p * t).flatten(1).sum(1)
    den = p.flatten(1).sum(1) + t.flatten(1).sum(1)
    dice = 1.0 - ((2 * inter + 1) / (den + 1)).mean()
    return bce_weight * bce + dice_weight * dice


@torch.no_grad()
def hard_iou(model, samples, device, thresh, bs=2):
    model.eval()
    ious = []
    for s in range(0, len(samples), bs):
        idx = list(range(s, min(s + bs, len(samples))))
        image_feats, ref_feats, target, valid = batch(samples, idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(image_feats, ref_feats)
        logits = F.interpolate(logits.float(), size=target.shape[-2:],
                               mode="bilinear", align_corners=False)
        pred = (logits.sigmoid() > thresh) & valid.bool()
        tgt = target.bool() & valid.bool()
        inter = (pred & tgt).flatten(1).sum(1).float()
        union = (pred | tgt).flatten(1).sum(1).float()
        ious += (inter / union.clamp(min=1)).cpu().tolist()
    model.train()
    return float(np.mean(ious))


@torch.no_grad()
def probe_ceiling(samples, device, thresh, stride=4):
    """What the exact area-pooled target scores in this same setup."""
    ious = []
    for s in samples:
        t = s["target"].float()[None, None]
        v = s["valid"].bool()[None, None]
        gh, gw = t.shape[-2] // stride, t.shape[-1] // stride
        grid = F.interpolate(t, size=(gh, gw), mode="area")
        up = F.interpolate(grid, size=t.shape[-2:], mode="bilinear",
                           align_corners=False)
        pred = (up > thresh) & v
        tgt = t.bool() & v
        inter = (pred & tgt).sum().float()
        union = (pred | tgt).sum().float()
        ious.append((inter / union.clamp(min=1)).item())
    return float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="data/cache/v6d_train_1024")
    ap.add_argument("--fresh-cache", default="data/cache/v6d_fresh_1024")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-fresh", type=int, default=64)
    ap.add_argument("--hf14-cache", default="",
                    help="cached real plans; gives synthetic->real transfer")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(a.seed)
    print(f"loading cache ...", flush=True)
    train, channels = load_cache(a.train_cache, a.n_train, device)
    fresh, _ = load_cache(a.fresh_cache, a.n_fresh, device)
    hf14 = load_cache(a.hf14_cache, 10_000, device)[0] if a.hf14_cache else None
    ceil_tr = probe_ceiling(train, device, a.mask_thresh)
    ceil_fr = probe_ceiling(fresh, device, a.mask_thresh)
    print(f"  train {len(train)}  fresh {len(fresh)}  channels {channels}\n"
          f"  naive area-pooled ceiling (a FLOOR on the true bound, which the\n"
          f"  optimised-grid analysis puts near 0.99): train {ceil_tr:.4f}  "
          f"fresh {ceil_fr:.4f}\n")

    names = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    results = []
    for name in names:
        torch.manual_seed(a.seed)
        model = Decoder(channels=channels, **VARIANTS[name]).to(device)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
        rng = np.random.default_rng(a.seed)
        t0 = time.time()
        for step in range(a.steps):
            idx = rng.choice(len(train), a.batch_size, replace=False).tolist()
            image_feats, ref_feats, target, valid = batch(train, idx, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(image_feats, ref_feats)
            logits = F.interpolate(logits.float(), size=target.shape[-2:],
                                   mode="bilinear", align_corners=False)
            loss = loss_fn(logits, target, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (step + 1) % a.eval_every == 0:
                tr = hard_iou(model, train, device, a.mask_thresh)
                print(f"  {name:10s} step {step+1:5d}  loss {loss.item():.4f}  "
                      f"train IoU {tr:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        tr = hard_iou(model, train, device, a.mask_thresh)
        fr = hard_iou(model, fresh, device, a.mask_thresh)
        rl = hard_iou(model, hf14, device, a.mask_thresh) if hf14 else float("nan")
        dt = time.time() - t0
        results.append({"variant": name, "params": n_par, "train_iou": tr,
                        "fresh_iou": fr, "hf14_iou": rl, "seconds": dt})
        print(f"== {name:10s} {n_par/1e6:5.2f}M params  train {tr:.4f}  "
              f"fresh {fr:.4f}  hf14 {rl:.4f}  ({dt:.0f}s)\n", flush=True)

    print(f"\n== decoder sweep, {len(train)} cached train / {len(fresh)} fresh, "
          f"{a.steps} steps, threshold {a.mask_thresh}")
    print(f"   probe ceiling  train {ceil_tr:.4f}   fresh {ceil_fr:.4f}")
    print(f"   {'variant':11s} {'params':>9s} {'train':>8s} {'fresh':>8s} "
          f"{'hf14':>8s} {'sec':>5s}")
    for r in sorted(results, key=lambda r: -(r["hf14_iou"] if hf14 else r["fresh_iou"])):
        print(f"   {r['variant']:11s} {r['params']/1e6:8.2f}M {r['train_iou']:8.4f} "
              f"{r['fresh_iou']:8.4f} {r['hf14_iou']:8.4f} {r['seconds']:5.0f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "ceiling": {"train": ceil_tr, "fresh": ceil_fr},
                   "results": results}, open(a.out, "w"), indent=1)
        print(f"   wrote {a.out}")


if __name__ == "__main__":
    main()
