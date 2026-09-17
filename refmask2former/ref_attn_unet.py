"""Cross-attention variant of `RefUNet`.

Same backbone, same FPN top-down fusion, same direct-union training objective
as `ref_unet.RefUNet` -- the only change is how the two COARSEST scales
condition on the reference. `RefUNet.ConditionBlock` reduces the reference to
one globally-averaged vector per scale, which the docstring there already
flags as discarding "stroke orientation, spacing, and arrangement". `--corr-grid`
tried to recover that with per-location cosine similarity to a small reference
grid and made things monotonically worse (baseline 0.7672 -> g4 0.7300 -> g8
0.7179, `synth_progress.md` 2026-08-06). That mechanism only ever produced
similarity SCORES as extra input channels; it never aggregated the reference's
own feature VALUES per image location. `AttnConditionBlock` does: standard
multi-head cross-attention, image tokens as queries, reference tokens as
keys/values, so each image location receives a learned, softmax-weighted blend
of the reference's actual features -- not just a similarity hint. Whether that
distinction matters, or the finding above just repeats, is exactly the
open question this model tests.

Applied only at the two coarsest scales (c3 stride16, c4 stride32) for the same
reason `--corr-grid` skips the finest level: "deciding WHICH regions match is
semantic and resolved at the coarser levels", and full attention over
stride-4 image tokens against even a small reference grid is not tractable at
training resolution. The two finest scales (c1, c2) keep `ConditionBlock`
unchanged, so this is a single-variable swap, not a new architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from .ref_unet import ConditionBlock


class AttnConditionBlock(nn.Module):
    """Image tokens attend to reference tokens; fuse with the same
    [x, attended, |x-attended|] pattern `ConditionBlock` uses for
    [x, r, x*r, |x-r|] (dropping the product term: attended already IS a
    reference-derived quantity per location, not a broadcast constant, so a
    location-wise product doesn't carry the same "agreement" signal)."""

    def __init__(self, image_channels, ref_channels, width, num_heads=4):
        super().__init__()
        self.image_proj = nn.Conv2d(image_channels, width, 1)
        self.ref_proj = nn.Conv2d(ref_channels, width, 1)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU())

    def forward(self, image_features, reference_features):
        x = self.image_proj(image_features)                  # [B, width, H, W]
        r = self.ref_proj(reference_features)                 # [B, width, h, w]
        b, c, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)                      # [B, HW, width]
        kv = r.flatten(2).transpose(1, 2)                     # [B, hw, width]
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        parts = [x, attended, (x - attended).abs()]
        return self.fuse(torch.cat(parts, dim=1))


class RefCrossAttnUNet(nn.Module):
    """`RefUNet` with cross-attention conditioning at the two coarsest scales.

    Deliberately omits RefUNet's ablation extras (anchor, self-support,
    dynamic-filter, metric embeddings) that this repo already screened: this
    model isolates ONE new variable (attention vs. global-average
    conditioning), not a combination of untested ones.
    """

    def __init__(self, width=128, pretrained=True, num_heads=4):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1,
                                  backbone.relu, backbone.maxpool)
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        channels = (256, 512, 1024, 2048)
        self.condition = nn.ModuleList([
            ConditionBlock(channels[0], channels[0], width, corr_grid=0),
            ConditionBlock(channels[1], channels[1], width, corr_grid=0),
            AttnConditionBlock(channels[2], channels[2], width, num_heads),
            AttnConditionBlock(channels[3], channels[3], width, num_heads),
        ])
        self.smooth = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
                          nn.GroupNorm(8, width), nn.GELU())
            for _ in range(3))
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, 1, 1))

    def features(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4

    def decode(self, image_features, reference_features):
        conditioned = []
        for i, (block, x, r) in enumerate(
                zip(self.condition, image_features, reference_features)):
            if i < 2:
                prototype = r.mean((-2, -1))
                conditioned.append(block(x, prototype, r))
            else:
                conditioned.append(block(x, r))
        pyramid = conditioned[-1]
        for level in range(2, -1, -1):
            pyramid = F.interpolate(pyramid, size=conditioned[level].shape[-2:],
                                    mode="bilinear", align_corners=False)
            pyramid = self.smooth[level](pyramid + conditioned[level])
        return self.head(pyramid)

    def forward(self, image, reference, ref_box=None, return_embeddings=False,
                return_aux=False):
        # ref_box/return_embeddings/return_aux accepted, unused: kept so this
        # model is a drop-in for `RefUNet`'s call sites in the shared training
        # and evaluation scripts without branching on model type there.
        image_size = image.shape[-2:]
        image_features = self.features(image)
        reference_features = self.features(reference)
        logits = self.decode(image_features, reference_features)
        out = F.interpolate(logits, size=image_size, mode="bilinear",
                            align_corners=False)
        if return_aux:
            return out, None
        return out

    def parameter_groups(self, lr, backbone_lr_mult=0.1):
        backbone_modules = (self.stem, self.layer1, self.layer2,
                            self.layer3, self.layer4)
        backbone_ids = {id(p) for module in backbone_modules
                        for p in module.parameters()}
        backbone = [p for p in self.parameters() if id(p) in backbone_ids]
        task = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [{"params": backbone, "lr": lr * backbone_lr_mult},
                {"params": task, "lr": lr}]
