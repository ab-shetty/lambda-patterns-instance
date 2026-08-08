"""Direct reference-conditioned semantic mask model.

The output is one binary union mask for the pattern shown in ``reference``.
Connected components can be converted to user-facing instances downstream.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ConditionBlock(nn.Module):
    """Fuse image features with the reference.

    `corr_grid > 0` adds dense correlation ("learned template matching"): the
    reference is kept as a `corr_grid x corr_grid` set of spatial tokens and
    every image location is cosine-matched against all of them, instead of being
    compared only to one globally-averaged reference vector.

    Global averaging discards stroke orientation, spacing, and arrangement --
    precisely what distinguishes one hatch from another -- which is why region
    identification, not boundary placement, limits the model. Hand-engineered
    matching does carry signal here but only weakly (Gabor-energy cosine maps
    reach AUC 0.674, naive NCC 0.433 i.e. below chance because hatch is periodic
    and NCC is phase-sensitive), so the correlation is computed on learned
    features and left for the network to interpret.
    """

    def __init__(self, image_channels, ref_channels, width, corr_grid=4):
        super().__init__()
        self.corr_grid = corr_grid
        self.image = nn.Conv2d(image_channels, width, 1)
        self.reference = nn.Linear(ref_channels, width)
        extra = 0
        if corr_grid > 0:
            # per-token similarity maps + max/mean summaries
            self.corr_proj = nn.Conv2d(corr_grid * corr_grid + 2, width, 1)
            extra = width
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 4 + extra, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU())

    def correlate(self, image_features, reference_features):
        """Cosine similarity of every image location to every reference token."""
        g = self.corr_grid
        tokens = F.adaptive_avg_pool2d(reference_features, g)      # [B, C, g, g]
        b, c, h, w = image_features.shape
        x = F.normalize(image_features.flatten(2), dim=1)          # [B, C, HW]
        t = F.normalize(tokens.flatten(2), dim=1)                  # [B, C, g*g]
        corr = torch.einsum("bcn,bcm->bmn", x, t)                  # [B, g*g, HW]
        corr = corr.reshape(b, g * g, h, w)
        summary = torch.cat((corr.amax(1, keepdim=True),
                             corr.mean(1, keepdim=True)), dim=1)
        return self.corr_proj(torch.cat((corr, summary), dim=1))

    def forward(self, image, reference, reference_features=None):
        x = self.image(image)
        r = self.reference(reference)[:, :, None, None]
        r = r.expand_as(x)
        parts = [x, r, x * r, (x - r).abs()]
        if self.corr_grid > 0 and reference_features is not None:
            parts.append(self.correlate(image, reference_features))
        return self.fuse(torch.cat(parts, dim=1))


class RefUNet(nn.Module):
    """Shared ResNet features + multiscale reference-conditioned FPN."""

    def __init__(self, width=128, pretrained=True, corr_grid=0, metric_dim=0,
                 anchor=False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1,
                                  backbone.relu, backbone.maxpool)
        # Anchor channel: WHERE the user drew. The rectangle is always inside an
        # instance of the target pattern, so those pixels are guaranteed
        # positives -- but only the crop was ever passed in, so the model had to
        # re-locate the pattern from scratch. The extra input plane is
        # zero-initialised, making the model bit-identical to the baseline at
        # step 0 and leaving it free to learn how much to trust the anchor.
        self.anchor = anchor
        if anchor:
            old = self.stem[0]
            stem_conv = nn.Conv2d(4, old.out_channels, old.kernel_size,
                                  stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                stem_conv.weight[:, :3] = old.weight
                stem_conv.weight[:, 3:].zero_()
            self.stem[0] = stem_conv
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        channels = (256, 512, 1024, 2048)
        self.corr_grid = corr_grid
        # Correlation is skipped at the stride-4 level: it costs the most memory
        # there and matters least, since deciding WHICH regions match is semantic
        # and resolved at the coarser levels.
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
        # Optional metric head for the auxiliary hard-pair ranking loss. Off by
        # default, so a model built without it is bit-identical to the baseline.
        # Fine + coarse rather than coarse alone: the query-model lineage
        # measured res3+res5 well above res5 only (0.5195 vs 0.4578).
        self.metric_dim = metric_dim
        if metric_dim > 0:
            self.metric_proj = nn.Sequential(
                nn.Conv2d(channels[1] + channels[3], 256, 1, bias=False),
                nn.GroupNorm(8, 256), nn.GELU(),
                nn.Conv2d(256, metric_dim, 1))

    def features(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4

    def metric_embeddings(self, image_features, reference_features):
        """Siamese projection of image locations and the reference into one
        normalised space, so a dot product is a same-material score."""
        coarse = image_features[3]
        fine = F.interpolate(image_features[1], size=coarse.shape[-2:],
                             mode="bilinear", align_corners=False)
        image_embedding = F.normalize(
            self.metric_proj(torch.cat([fine, coarse], 1)), dim=1)
        reference_vector = torch.cat([reference_features[1].mean((-2, -1)),
                                      reference_features[3].mean((-2, -1))], 1)
        reference_embedding = F.normalize(
            self.metric_proj(reference_vector[:, :, None, None])[:, :, 0, 0], dim=1)
        return image_embedding, reference_embedding

    def forward(self, image, reference, ref_box=None, return_embeddings=False):
        image_size = image.shape[-2:]
        if self.anchor:
            if ref_box is None:
                ref_box = image.new_zeros((image.shape[0], 1) + tuple(image_size))
            image = torch.cat([image, ref_box], 1)
            # The reference crop IS the rectangle, so its anchor plane is all-ones
            # -- the same siamese backbone then sees a consistent meaning for the
            # channel on both inputs.
            reference = torch.cat(
                [reference, reference.new_ones(reference.shape[0], 1,
                                               *reference.shape[-2:])], 1)
        image_features = self.features(image)
        reference_features = self.features(reference)
        reference_vectors = [f.mean((-2, -1)) for f in reference_features]
        conditioned = [block(x, r, rf) for block, x, r, rf in zip(
            self.condition, image_features, reference_vectors, reference_features)]
        pyramid = conditioned[-1]
        for level in range(2, -1, -1):
            pyramid = F.interpolate(pyramid, size=conditioned[level].shape[-2:],
                                    mode="bilinear", align_corners=False)
            pyramid = self.smooth[level](pyramid + conditioned[level])
        logits = self.head(pyramid)
        out = F.interpolate(logits, size=image_size, mode="bilinear",
                            align_corners=False)
        if return_embeddings and self.metric_dim > 0:
            return (out,) + self.metric_embeddings(image_features, reference_features)
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
