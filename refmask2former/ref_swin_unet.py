"""Vision-transformer backbone variant of `RefUNet`.

`RefUNet` and `RefCrossAttnUNet` both sit on the same ImageNet ResNet50. The
2026-09-18 note in `codex_doc.md` says the transformer hypothesis "has never
actually been tested here" -- `RefCrossAttnUNet` is a ResNet with attention
bolted onto its two coarsest conditioning blocks, at 27.9M parameters against
RefUNet's 28.0M. This swaps the BACKBONE instead, and changes nothing else:
same `ConditionBlock` conditioning, same FPN top-down fusion, same head, same
direct-union objective.

Swin is the transformer that fits here without redesigning the decoder. It is
hierarchical (strides 4/8/16/32, exactly the pyramid `decode` consumes) and its
self-attention runs inside shifted 7x7 windows, so it keeps a locality prior and
trains on ImageNet-scale data rather than needing JFT-scale. A plain ViT would
give one stride-16 scale and no pyramid.

Measured motivation (frozen-feature probe, `scripts/decoder_search.py`,
2026-09-20): with the identical decoder on cached features, Swin-B beat
ResNet50 on the 52 real selections by +0.084 at 1024 px and +0.190 at 2048 px,
while fitting the synthetic training data WORSE (0.75-0.83 against 0.91-0.93).
`swin_t`, at 28.3M against ResNet50's 25.6M, held most of that, so the effect
is architectural rather than a parameter count. What the probe could not test
is finetuning -- every probe number holds the backbone frozen, and this module
exists to close that gap.

`--backbone-lr-mult` applies to the Swin stages exactly as it does to the
ResNet stages, via `parameter_groups`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_condition import CrossAttnCondition
from .ref_unet import ConditionBlock

# Stage output channels after blocks 1/3/5/7 of torchvision's `features`.
SWIN_CHANNELS = {"swin_t": (96, 192, 384, 768),
                 "swin_s": (96, 192, 384, 768),
                 "swin_b": (128, 256, 512, 1024)}


class RefSwinUNet(nn.Module):
    """Swin backbone + the unchanged reference-conditioned FPN decoder."""

    def __init__(self, width=128, pretrained=True, backbone="swin_b",
                 corr_grid=0, decoder="baseline", roi_ref=False, roi_mode="replace",
                 roi_min_cells=0.0):
        super().__init__()
        # roi_ref (2026-09-27, diagnostic): take each level's prototype from
        # the IMAGE's own features inside the user's box instead of from the
        # separately encoded 224 px crop. The crop is magnified ~5x relative
        # to the plan, so the siamese comparison is across scales; pooling
        # from the image puts reference and targets at one scale and context.
        self.roi_ref = bool(roi_ref)
        # roi_mode "add": keep the crop prototype and ADD a zero-initialised
        # projection of the image-ROI prototype (starts identical to baseline).
        if roi_mode not in ("replace", "add"):
            raise ValueError(f"unknown roi_mode {roi_mode!r}")
        self.roi_mode = roi_mode
        # roi_min_cells > 0: per level, scale the ROI term by
        # min(1, box coverage in feature cells / roi_min_cells). A box smaller
        # than a coarse cell pools mostly its surroundings (real plans: many
        # 20-60 px boxes at 2048 input), so there the crop prototype wins.
        self.roi_min_cells = float(roi_min_cells)
        import torchvision.models as tvm
        if backbone not in SWIN_CHANNELS:
            raise ValueError(f"unknown swin backbone {backbone!r}")
        self.backbone_name = backbone
        net = getattr(tvm, backbone)(weights="DEFAULT" if pretrained else None)
        # torchvision's swin `features` alternates stage / patch-merging; the
        # stage outputs are at indices 1,3,5,7 and come back NHWC.
        self.stages = nn.ModuleList(net.features)
        self._taps = (1, 3, 5, 7)
        channels = SWIN_CHANNELS[backbone]
        self.corr_grid = corr_grid
        # decoder="selfattn" (2026-09-27): the two coarsest scales (1/16, 1/32)
        # condition by cross-attention to the reference's spatial tokens, then
        # self-attention over the image, so regions can be compared with each
        # other. Ranked first by the frozen-backbone screen (2026-09-20) and
        # never run unfrozen. "baseline" is the historical decoder.
        if decoder not in ("baseline", "selfattn"):
            raise ValueError(f"unknown decoder {decoder!r}")
        self.decoder = decoder
        attn = (2, 3) if decoder == "selfattn" else ()
        self.condition = nn.ModuleList(
            CrossAttnCondition(c, c, width, num_heads=4, self_attn=True) if i in attn
            else ConditionBlock(c, c, width, corr_grid=(0 if i == 0 else corr_grid))
            for i, c in enumerate(channels))
        if self.roi_ref and roi_mode == "add":
            self.roi_proj = nn.ModuleList(nn.Linear(c, c) for c in channels)
            for lin in self.roi_proj:
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.smooth = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
                          nn.GroupNorm(8, width), nn.GELU())
            for _ in range(3))
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, 1, 1))

    def features(self, x):
        """Four NCHW feature maps at strides 4/8/16/32, matching `RefUNet`."""
        out, t = [], x
        for i, stage in enumerate(self.stages):
            t = stage(t)
            if i in self._taps:
                out.append(t.permute(0, 3, 1, 2).contiguous())
        return out

    def decode(self, image_features, prototypes, reference_features):
        conditioned = [block(x, r, rf) for block, x, r, rf in zip(
            self.condition, image_features, prototypes, reference_features)]
        pyramid = conditioned[-1]
        for level in range(2, -1, -1):
            pyramid = F.interpolate(pyramid, size=conditioned[level].shape[-2:],
                                    mode="bilinear", align_corners=False)
            pyramid = self.smooth[level](pyramid + conditioned[level])
        return self.head(pyramid)

    def roi_prototypes(self, image_features, boxes, fallback):
        """Box-weighted mean of each image feature level; `boxes` is
        [N, 1, H, W] at input scale. Falls back to the crop prototype where a
        box covers no feature cell."""
        out = []
        for f, fb in zip(image_features, fallback):
            w = F.interpolate(boxes.to(f.dtype), size=f.shape[-2:], mode="area")
            mass = w.flatten(1).sum(1)
            p = (f * w).flatten(2).sum(2) / mass[:, None].clamp(min=1e-6)
            ok = (mass > 1e-3).to(f.dtype)[:, None]
            if getattr(self, "roi_min_cells", 0.0) > 0:
                ok = ok * (mass / self.roi_min_cells).clamp(max=1.0).to(f.dtype)[:, None]
            out.append(p * ok + fb * (1 - ok))
        if getattr(self, "roi_mode", "replace") == "add":
            return [fb + proj(p - fb) for fb, proj, p in zip(fallback, self.roi_proj, out)]
        return out

    def forward(self, image, reference, ref_box=None, return_embeddings=False,
                return_aux=False):
        # `ref_box` is ignored unless roi_ref: this variant is anchor-free,
        # which is the documented default (see "Deliberately anchor-free" in
        # startup.md). Keeping the argument lets the training loop and the
        # evaluator call every model the same way.
        image_size = image.shape[-2:]
        image_features = self.features(image)
        reference_features = self.features(reference)
        prototypes = [f.mean((-2, -1)) for f in reference_features]
        if self.roi_ref and ref_box is not None:
            prototypes = self.roi_prototypes(image_features, ref_box, prototypes)
        logits = self.decode(image_features, prototypes, reference_features)
        out = F.interpolate(logits, size=image_size, mode="bilinear",
                            align_corners=False)
        if return_aux:
            return out, None
        return out

    def forward_multi(self, image, references, k, boxes=None):
        """K questions per image (`--refs-per-image`): `references` is
        [B*K, 3, R, R], plan-major. The image runs through the backbone ONCE;
        only the conditioning and decoder repeat per question."""
        image_size = image.shape[-2:]
        image_features = [f.repeat_interleave(k, 0) for f in self.features(image)]
        reference_features = self.features(references)
        prototypes = [f.mean((-2, -1)) for f in reference_features]
        if self.roi_ref and boxes is not None:
            prototypes = self.roi_prototypes(image_features, boxes, prototypes)
        logits = self.decode(image_features, prototypes, reference_features)
        return F.interpolate(logits, size=image_size, mode="bilinear",
                             align_corners=False)

    def forward_cached(self, image_features, image_size, references, boxes=None):
        """Evaluation: K questions on one image whose `features` were computed ONCE
        (batch 1). Same maths as `forward` per question; the backbone is skipped."""
        k = references.shape[0]
        image_features = [f.expand(k, -1, -1, -1) for f in image_features]
        reference_features = self.features(references)
        prototypes = [f.mean((-2, -1)) for f in reference_features]
        if self.roi_ref and boxes is not None:
            prototypes = self.roi_prototypes(image_features, boxes, prototypes)
        logits = self.decode(image_features, prototypes, reference_features)
        return F.interpolate(logits, size=image_size, mode="bilinear",
                             align_corners=False)

    def parameter_groups(self, lr, backbone_lr_mult=0.1):
        backbone_ids = {id(p) for p in self.stages.parameters()}
        backbone = [p for p in self.parameters() if id(p) in backbone_ids]
        task = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [{"params": backbone, "lr": lr * backbone_lr_mult},
                {"params": task, "lr": lr}]
