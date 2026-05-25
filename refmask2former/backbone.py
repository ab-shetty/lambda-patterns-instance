"""ResNet50 backbone producing multi-scale feature maps (C2..C5)."""

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNetBackbone(nn.Module):
    """Wraps a torchvision ResNet50 and returns stride 4/8/16/32 features.

    Output channels: res2=256, res3=512, res4=1024, res5=2048.
    """

    out_channels = {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048}
    out_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
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
