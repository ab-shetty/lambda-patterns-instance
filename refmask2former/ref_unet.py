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
                 anchor=False, anchor_dropout=0.0, anchor_ref_plane=0.0,
                 self_support=0.0, self_support_thresh=0.7,
                 dynamic_filter=False):
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
        # Anchor dropout: hide the anchor on a fraction of TRAINING samples so the
        # model has to stay able to solve the task from the reference crop alone.
        # Without it the anchor is the cheap route -- it settles ~84% of the
        # training pool on its own -- so loss falls down that path and the
        # reference-matching pathway gets correspondingly little gradient and
        # never trains to baseline quality. That is why adding the anchor made
        # the model WORSE than omitting it, even though the extra input plane is
        # zero-initialised and could simply have been left unused.
        self.anchor_dropout = float(anchor_dropout)
        # Value of the anchor channel on the REFERENCE branch. It defaults to 1.0
        # ("this crop is the target"), but the backbone is siamese: the same
        # filters then see a sparse rectangle-in-a-field-of-zeros on the image
        # branch and a constant 1.0 on the reference branch. Those are very
        # different input statistics through shared weights, which can corrupt
        # the reference features themselves -- the observed failure is a model
        # whose matching pathway is still fully active (reference sensitivity
        # 0.92) but whose masks are much worse, which is what that would look
        # like. 0.0 matches the image plane's dominant value instead, and is
        # the default: the 1.0 path costs 0.069 HF14 and has no measured use, so
        # it is kept only so an older anchored checkpoint still reloads exactly
        # (`load_refunet` reads this value from the checkpoint's saved args, not
        # from this default).
        self.anchor_ref_plane = float(anchor_ref_plane)
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
        # Dynamic filter / hypernetwork conditioning. Hossler, "Where's Waldo? A
        # Deep Learning approach to Template Matching" (CS231n 2017) feeds the
        # template through a hypernetwork that GENERATES the last conv layer's
        # weights for the query branch, and the same idea is the "conditional
        # networks" family in Catalano & Matteucci's FSS review (sec. 3.1), where
        # the support set produces a parameter set theta used as a per-pixel
        # classifier. This model instead concatenates a reference vector into
        # every level and lets a FIXED classifier read the result, so the decision
        # boundary itself has never been reference-dependent. The generated 1x1
        # kernel is zero-initialised, so the model starts bit-identical to the
        # baseline and has to earn any use of the path.
        self.dynamic_filter = bool(dynamic_filter)
        if self.dynamic_filter:
            self.filter_gen = nn.Sequential(
                nn.Linear(channels[3], width), nn.GELU(),
                nn.Linear(width, width + 1))
            nn.init.zeros_(self.filter_gen[-1].weight)
            nn.init.zeros_(self.filter_gen[-1].bias)
        # Self-support prototype refinement (Fan et al., via the FSS review sec.
        # 3.2: "update the prototypes by selecting the query features that
        # self-match the prototypes with high confidence"). The reference is one
        # small rectangle, so its prototype describes that rectangle's appearance,
        # not the family's across the whole sheet -- the gap the review names as
        # the core limitation of single-prototype models. A first pass predicts a
        # mask, its confident pixels are pooled out of the QUERY's own features,
        # and that self-prototype is mixed into the reference prototype for a
        # second pass. It stays a global-average prototype throughout, so it does
        # not reintroduce the spatial correspondence that `--corr-grid` showed to
        # be harmful here (-0.037 at g=4, -0.049 at g=8).
        self.self_support = float(self_support)
        self.self_support_thresh = float(self_support_thresh)
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

    def decode(self, image_features, prototypes, reference_features):
        """Condition every level on `prototypes`, fuse top-down, emit logits."""
        conditioned = [block(x, r, rf) for block, x, r, rf in zip(
            self.condition, image_features, prototypes, reference_features)]
        pyramid = conditioned[-1]
        for level in range(2, -1, -1):
            pyramid = F.interpolate(pyramid, size=conditioned[level].shape[-2:],
                                    mode="bilinear", align_corners=False)
            pyramid = self.smooth[level](pyramid + conditioned[level])
        logits = self.head(pyramid)
        if self.dynamic_filter:
            generated = self.filter_gen(prototypes[-1])
            kernel, bias = generated[:, :-1], generated[:, -1]
            logits = logits + (torch.einsum("bc,bchw->bhw", kernel, pyramid)
                               + bias[:, None, None])[:, None]
        return logits

    def self_prototypes(self, image_features, prototypes, logits):
        """Pool the query's own confident pixels into a refined prototype.

        Falls back to the reference prototype for any sample whose first pass
        found nothing confident -- an empty prediction must not silently become a
        prototype pooled over zero pixels.
        """
        probability = logits.detach().sigmoid()
        refined = []
        for features, prototype in zip(image_features, prototypes):
            weight = F.interpolate(probability, size=features.shape[-2:],
                                   mode="bilinear", align_corners=False)
            weight = weight * (weight >= self.self_support_thresh)
            mass = weight.flatten(1).sum(1)                            # [B]
            pooled = ((features * weight).flatten(2).sum(2)
                      / mass[:, None].clamp(min=1e-4))
            found = (mass > 1.0).to(features.dtype)[:, None]
            pooled = pooled * found + prototype * (1.0 - found)
            refined.append((1.0 - self.self_support) * prototype
                           + self.self_support * pooled)
        return refined

    def forward(self, image, reference, ref_box=None, return_embeddings=False,
                return_aux=False):
        image_size = image.shape[-2:]
        if self.anchor:
            if ref_box is None:
                ref_box = image.new_zeros((image.shape[0], 1) + tuple(image_size))
            keep = None
            if self.training and self.anchor_dropout > 0.0:
                keep = (torch.rand(image.shape[0], 1, 1, 1, device=image.device)
                        >= self.anchor_dropout).to(image.dtype)
                ref_box = ref_box * keep
            image = torch.cat([image, ref_box], 1)
            # The reference crop IS the rectangle, so its anchor plane is all-ones
            # -- the same siamese backbone then sees a consistent meaning for the
            # channel on both inputs. When the anchor is dropped it must go to
            # zero here too, or the pair is inconsistent: the reference would
            # still assert "this is the target" while the image denies knowing
            # where that is.
            ref_plane = reference.new_full((reference.shape[0], 1,
                                            *reference.shape[-2:]),
                                           self.anchor_ref_plane)
            if keep is not None:
                ref_plane = ref_plane * keep
            reference = torch.cat([reference, ref_plane], 1)
        image_features = self.features(image)
        reference_features = self.features(reference)
        prototypes = [f.mean((-2, -1)) for f in reference_features]
        logits = self.decode(image_features, prototypes, reference_features)
        auxiliary = None
        if self.self_support > 0.0:
            auxiliary = logits
            logits = self.decode(
                image_features,
                self.self_prototypes(image_features, prototypes, logits),
                reference_features)
        out = F.interpolate(logits, size=image_size, mode="bilinear",
                            align_corners=False)
        if return_embeddings and self.metric_dim > 0:
            return (out,) + self.metric_embeddings(image_features, reference_features)
        if return_aux:
            if auxiliary is not None:
                auxiliary = F.interpolate(auxiliary, size=image_size,
                                          mode="bilinear", align_corners=False)
            return out, auxiliary
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
