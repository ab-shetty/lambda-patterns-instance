"""ResNet50 backbone producing multi-scale feature maps (C2..C5).

A `stem_pool` knob controls the stride-2 downsample inside the stem:

  - "max"  (default): the stock ResNet50 3x3 stride-2 max-pool.
  - "blur": an anti-aliased binomial blur-pool (line-art variant). On dark
    lines drawn on a white background, max-pooling a 2x2 window keeps the
    *brightest* pixel and therefore erases thin strokes at the very first
    stride-4 downsample. A blur-pool low-pass filters then subsamples, so a
    thin dark line darkens its window instead of being overwritten by the
    background -- the high-frequency line structure these plans carry survives
    into the feature pyramid. Every ImageNet-pretrained conv weight is kept;
    only the pooling op changes (zero new learnable params).
  - "avg":  plain 3x3 stride-2 average pool (cheaper cousin of "blur").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class BlurPool2d(nn.Module):
    """Anti-aliased stride-2 downsample: depthwise binomial blur, then subsample.

    Uses a fixed 3x3 binomial kernel [[1,2,1],[2,4,2],[1,2,1]] / 16 (no learnable
    params). Reflect padding avoids darkening image borders.
    """

    def __init__(self, channels, stride=2):
        super().__init__()
        self.stride = stride
        k = torch.tensor([1.0, 2.0, 1.0])
        kernel = (k[:, None] * k[None, :])
        kernel = kernel / kernel.sum()
        # [channels, 1, 3, 3] for depthwise conv (groups=channels).
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.kernel, stride=self.stride, groups=self.channels)


class ResNetBackbone(nn.Module):
    """Wraps a torchvision ResNet50 and returns stride 4/8/16/32 features.

    Output channels: res2=256, res3=512, res4=1024, res5=2048.
    Strides are unchanged across all `stem_pool` options, so the pixel decoder
    and transformer are agnostic to the choice -- this is a drop-in swap.
    """

    out_channels = {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048}
    out_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    def __init__(self, pretrained=True, stem_pool="max"):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        if stem_pool == "max":
            pool = resnet.maxpool  # stock 3x3 stride-2 max-pool
        elif stem_pool == "blur":
            # 64 channels out of conv1/bn1/relu at this point.
            pool = BlurPool2d(channels=64, stride=2)
        elif stem_pool == "avg":
            pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"unknown stem_pool: {stem_pool!r}")
        self.stem_pool = stem_pool

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, pool)
        self.layer1 = resnet.layer1  # stride 4
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16
        self.layer4 = resnet.layer4  # stride 32

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"res2": c2, "res3": c3, "res4": c4, "res5": c5}
