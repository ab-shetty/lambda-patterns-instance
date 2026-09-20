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

from .ref_unet import ConditionBlock

# Stage output channels after blocks 1/3/5/7 of torchvision's `features`.
SWIN_CHANNELS = {"swin_t": (96, 192, 384, 768),
                 "swin_s": (96, 192, 384, 768),
                 "swin_b": (128, 256, 512, 1024)}


class RefSwinUNet(nn.Module):
    """Swin backbone + the unchanged reference-conditioned FPN decoder."""

    def __init__(self, width=128, pretrained=True, backbone="swin_b",
                 corr_grid=0):
        super().__init__()
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
        self.condition = nn.ModuleList(
            ConditionBlock(c, c, width, corr_grid=(0 if i == 0 else corr_grid))
            for i, c in enumerate(channels))
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

    def forward(self, image, reference, ref_box=None, return_embeddings=False,
                return_aux=False):
        # `ref_box` is accepted and ignored: this variant is anchor-free, which
        # is the documented default (see "Deliberately anchor-free" in
        # startup.md). Keeping the argument lets the training loop and the
        # evaluator call every model the same way.
        image_size = image.shape[-2:]
        image_features = self.features(image)
        reference_features = self.features(reference)
        prototypes = [f.mean((-2, -1)) for f in reference_features]
        logits = self.decode(image_features, prototypes, reference_features)
        out = F.interpolate(logits, size=image_size, mode="bilinear",
                            align_corners=False)
        if return_aux:
            return out, None
        return out

    def parameter_groups(self, lr, backbone_lr_mult=0.1):
        backbone_ids = {id(p) for p in self.stages.parameters()}
        backbone = [p for p in self.parameters() if id(p) in backbone_ids]
        task = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [{"params": backbone, "lr": lr * backbone_lr_mult},
                {"params": task, "lr": lr}]
