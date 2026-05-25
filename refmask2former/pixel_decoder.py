"""FPN-style pixel decoder.

Takes multi-scale backbone features and produces:
  - mask_features: high-resolution (stride 4) per-pixel embeddings used to
    generate masks via dot product with per-query mask embeddings.
  - multi-scale transformer features at strides 8/16/32, used as memory for
    the transformer decoder's cross-attention.

This is the simpler FPN variant of the Mask2Former pixel decoder (no
multi-scale deformable attention), chosen so the whole model uses only
standard PyTorch ops (no custom CUDA kernels).
"""

import torch.nn as nn
import torch.nn.functional as F


def _conv_gn_relu(in_ch, out_ch, k=3, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
              nn.GroupNorm(32, out_ch)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class FPNPixelDecoder(nn.Module):
    def __init__(self, in_channels_per_level, conv_dim=256, mask_dim=256):
        """
        Args:
            in_channels_per_level: dict like {"res2":256,"res3":512,"res4":1024,"res5":2048}
            conv_dim: channel dim of the FPN / transformer features
            mask_dim: channel dim of the final per-pixel mask embedding
        """
        super().__init__()
        self.in_features = ["res2", "res3", "res4", "res5"]

        # Lateral 1x1 convs to bring every level to conv_dim.
        self.lateral_convs = nn.ModuleDict()
        # Output 3x3 convs after top-down fusion.
        self.output_convs = nn.ModuleDict()
        for name in self.in_features:
            self.lateral_convs[name] = _conv_gn_relu(
                in_channels_per_level[name], conv_dim, k=1, act=False)
            self.output_convs[name] = _conv_gn_relu(conv_dim, conv_dim, k=3, act=True)

        # Final projection to mask embedding space (at stride 4 / res2).
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1)

        # Levels fed to the transformer decoder (low->high stride order set in model).
        self.transformer_levels = ["res5", "res4", "res3"]
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim

    def forward(self, features):
        # Top-down pathway, coarse (res5) -> fine (res2).
        prev = None
        fused = {}
        for name in reversed(self.in_features):  # res5, res4, res3, res2
            lat = self.lateral_convs[name](features[name])
            if prev is not None:
                lat = lat + F.interpolate(prev, size=lat.shape[-2:],
                                          mode="nearest")
            out = self.output_convs[name](lat)
            fused[name] = out
            prev = out

        mask_features = self.mask_features(fused["res2"])  # [B, mask_dim, H/4, W/4]
        multi_scale = [fused[n] for n in self.transformer_levels]  # res5, res4, res3
        return mask_features, multi_scale
