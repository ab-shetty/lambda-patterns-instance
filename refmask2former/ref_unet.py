"""Direct reference-conditioned semantic mask model.

The output is one binary union mask for the pattern shown in ``reference``.
Connected components can be converted to user-facing instances downstream.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ConditionBlock(nn.Module):
    def __init__(self, image_channels, ref_channels, width):
        super().__init__()
        self.image = nn.Conv2d(image_channels, width, 1)
        self.reference = nn.Linear(ref_channels, width)
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 4, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU())

    def forward(self, image, reference):
        x = self.image(image)
        r = self.reference(reference)[:, :, None, None]
        r = r.expand_as(x)
        return self.fuse(torch.cat((x, r, x * r, (x - r).abs()), dim=1))


class RefUNet(nn.Module):
    """Shared ResNet features + multiscale reference-conditioned FPN."""

    def __init__(self, width=128, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1,
                                  backbone.relu, backbone.maxpool)
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        channels = (256, 512, 1024, 2048)
        self.condition = nn.ModuleList(
            ConditionBlock(c, c, width) for c in channels)
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

    def forward(self, image, reference):
        image_size = image.shape[-2:]
        image_features = self.features(image)
        reference_features = self.features(reference)
        reference_vectors = [f.mean((-2, -1)) for f in reference_features]
        conditioned = [block(x, r) for block, x, r in zip(
            self.condition, image_features, reference_vectors)]
        pyramid = conditioned[-1]
        for level in range(2, -1, -1):
            pyramid = F.interpolate(pyramid, size=conditioned[level].shape[-2:],
                                    mode="bilinear", align_corners=False)
            pyramid = self.smooth[level](pyramid + conditioned[level])
        logits = self.head(pyramid)
        return F.interpolate(logits, size=image_size, mode="bilinear",
                             align_corners=False)

    def parameter_groups(self, lr, backbone_lr_mult=0.1):
        backbone_modules = (self.stem, self.layer1, self.layer2,
                            self.layer3, self.layer4)
        backbone_ids = {id(p) for module in backbone_modules
                        for p in module.parameters()}
        backbone = [p for p in self.parameters() if id(p) in backbone_ids]
        task = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [{"params": backbone, "lr": lr * backbone_lr_mult},
                {"params": task, "lr": lr}]
