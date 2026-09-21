"""Self-supervised plain-ViT backbone variant of `RefUNet` (DINOv2 / DINOv3).

`ref_swin_unet.py` tested the first transformer BACKBONE here and won
(+0.0707 HF14, one seed, 2026-09-20). It deliberately picked the transformer
that needed no decoder redesign: Swin is hierarchical, so it already emits the
stride-4/8/16/32 pyramid `decode` consumes. Its own docstring names what that
choice left untested -- "a plain ViT would give one stride-16 scale and no
pyramid" -- and `codex_doc.md` lists a DINOv2-class backbone as the open arm.

Two reasons to expect a plain self-supervised ViT to do better here, both
specific to this task rather than to ImageNet leaderboards:

* The task is dense semantic CORRESPONDENCE -- "here is a patch, find every
  region made of the same material" -- and matching patches across an image is
  the property DINOv2's self-distillation objective optimises directly.
  Supervised ImageNet features (what ResNet50, ConvNeXt and Swin all carry
  here) are trained to discard within-class variation, which is the signal
  this task needs kept.
* `residual_decomp.py` puts 46-56% of the remaining error at REGION level
  (wrong region selected, interior not filled) and at most a fifth on
  boundaries. Region-level errors are correspondence errors.

Against that, one caveat this repo has already been bitten by: these are
architectural line drawings, not natural images, and `material_separability.py`
found the backbone with the WORST within-image family separability (swin_b)
had the BEST transfer. Natural-image intuition has not predicted this domain
well, so this module exists to measure, not to assume.

WHAT THE ADAPTER DOES. A plain ViT emits one feature map at stride `patch`
(14 for DINOv2, 16 for DINOv3). The decoder wants four. Rather than resample
one final map four times -- which gives four views of identical information --
this taps four evenly spaced transformer blocks and resamples each to one
pyramid level, the arrangement DPT uses for dense prediction: early blocks
carry finer, more local detail and late blocks carry semantics, so the levels
differ in content and not merely in resolution.

Channels are projected to `(96, 192, 384, 768)` -- swin_t's exact widths -- so
`ConditionBlock`, the FPN fusion and the head are bit-identical in shape to the
winning Swin arm and the comparison stays a backbone comparison.

DINOv3 SWAP. `BACKBONES` maps a short name to an HF repo; DINOv3 entries are
present but its repos are gated `manual`, so they 403 until the account behind
`HF_TOKEN` accepts the license. Nothing else changes: DINOv3 is patch 16, which
divides 1024 and 224 exactly, so it needs less resampling than DINOv2, not more.

COST WARNING. Self-attention is quadratic in tokens and this backbone is NOT
windowed. At 1024 px that is (1022/14)^2 = 5,329 tokens; at 2048 it is ~21,300,
about 16x the attention cost. Training resolution is this project's largest
recorded lever (+0.05 to +0.07), so check throughput before assuming this arm
can be run where the ResNet and Swin arms are run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ref_unet import ConditionBlock

# Short name -> (HF repo, patch size). DINOv3 is gated; see module docstring.
BACKBONES = {
    "dinov2_s": ("facebook/dinov2-small", 14),
    "dinov2": ("facebook/dinov2-base", 14),
    "dinov2_l": ("facebook/dinov2-large", 14),
    "dinov3_s": ("facebook/dinov3-vits16-pretrain-lvd1689m", 16),
    "dinov3": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 16),
    "dinov3_l": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 16),
}

# Matches swin_t, so the decoder is unchanged across the two transformer arms.
PYRAMID_CHANNELS = (96, 192, 384, 768)
PYRAMID_STRIDES = (4, 8, 16, 32)


class RefDinoUNet(nn.Module):
    """Plain ViT backbone + simple-FPN adapter + the unchanged decoder."""

    def __init__(self, width=128, pretrained=True, backbone="dinov2",
                 corr_grid=0):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown dino backbone {backbone!r}")
        repo, patch = BACKBONES[backbone]
        self.backbone_name = backbone
        self.patch = patch

        from transformers import AutoConfig, AutoModel
        if pretrained:
            self.trunk = AutoModel.from_pretrained(repo)
        else:
            # `--init-from` and every evaluator build the architecture and then
            # load our own weights, so do not reach for the hub here.
            self.trunk = AutoModel.from_config(AutoConfig.from_pretrained(repo))
        dim = self.trunk.config.hidden_size
        depth = self.trunk.config.num_hidden_layers
        # DINOv2 interpolates learned position embeddings for off-size inputs
        # and must be TOLD to; DINOv3 uses RoPE, handles any size natively and
        # does not need it. Neither the signature nor a probe can distinguish
        # them -- both route the flag through `**kwargs` -- so this asks the
        # only question that matters: is passing it safe? It is needed on
        # DINOv2 and an ignored no-op on DINOv3. (Inspecting the signature
        # instead returns False for BOTH, which would silently run the DINOv2
        # arm on un-interpolated position embeddings.)
        self._interp_pos = self._accepts_interp_pos(patch)

        # Four evenly spaced blocks, last one included (DPT-style taps).
        self.taps = tuple(int(round((i + 1) * depth / 4)) - 1 for i in range(4))

        # One 1x1 projection per level; the resample itself is parameter-free.
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(dim, c, 1, bias=False),
                          nn.GroupNorm(8, c), nn.GELU())
            for c in PYRAMID_CHANNELS)

        self.corr_grid = corr_grid
        self.condition = nn.ModuleList(
            ConditionBlock(c, c, width, corr_grid=(0 if i == 0 else corr_grid))
            for i, c in enumerate(PYRAMID_CHANNELS))
        self.smooth = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
                          nn.GroupNorm(8, width), nn.GELU())
            for _ in range(3))
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, 1, 1))

    @torch.no_grad()
    def _accepts_interp_pos(self, patch):
        probe = torch.zeros(1, 3, patch * 2, patch * 2)
        try:
            self.trunk(pixel_values=probe, interpolate_pos_encoding=True)
            return True
        except (TypeError, ValueError):
            return False

    def features(self, x):
        """Four NCHW maps at strides 4/8/16/32 of `x`, matching `RefUNet`.

        The ViT only accepts whole patches, so `x` is resized to the nearest
        multiple of the patch size; the pyramid is then built at strides of the
        ORIGINAL size, so callers never see the adjustment.
        """
        h, w = x.shape[-2:]
        p = self.patch
        hp, wp = max(p, round(h / p) * p), max(p, round(w / p) * p)
        if (hp, wp) != (h, w):
            x = F.interpolate(x, size=(hp, wp), mode="bilinear",
                              align_corners=False)
        gh, gw = hp // p, wp // p

        kw = {"output_hidden_states": True}
        if self._interp_pos:
            kw["interpolate_pos_encoding"] = True
        out = self.trunk(pixel_values=x, **kw)
        # hidden_states[0] is the embedding output, so block i is index i + 1.
        levels = []
        for tap, proj, stride in zip(self.taps, self.project, PYRAMID_STRIDES):
            tokens = out.hidden_states[tap + 1]
            # Drop CLS and any register tokens (DINOv2 has 1, DINOv3 has 5 =
            # CLS + 4 registers) by keeping only the last gh*gw tokens.
            tokens = tokens[:, tokens.shape[1] - gh * gw:]
            grid = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, gh, gw)
            size = (max(1, -(-h // stride)), max(1, -(-w // stride)))
            levels.append(proj(F.interpolate(grid.float(), size=size,
                                             mode="bilinear",
                                             align_corners=False)))
        return levels

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
        # `ref_box` is accepted and ignored: anchor-free is the documented
        # default, exactly as in `ref_swin_unet.py`.
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
        backbone_ids = {id(p) for p in self.trunk.parameters()}
        backbone = [p for p in self.parameters() if id(p) in backbone_ids]
        task = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [{"params": backbone, "lr": lr * backbone_lr_mult},
                {"params": task, "lr": lr}]
