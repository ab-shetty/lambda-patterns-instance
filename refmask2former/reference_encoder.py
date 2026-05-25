"""Encodes a reference patch into a normalized embedding.

The embedding lives in the same space as the per-query reference embeddings
produced by the transformer decoder, so cosine similarity between them scores
whether a detected instance matches the user-selected pattern.
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ReferenceEncoder(nn.Module):
    def __init__(self, ref_dim=128, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> [B, 2048, 1, 1]
        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, ref_dim),
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)   # [B, 2048]
        emb = self.proj(feat)                # [B, ref_dim]
        return F.normalize(emb, dim=-1)
