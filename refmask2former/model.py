"""RefMask2Former: reference-conditioned, class-agnostic instance segmentation.

Pipeline:
  image --> ResNet50 backbone --> FPN pixel decoder --> {mask_features, multi-scale}
  multi-scale + mask_features --> transformer decoder --> per-query {class, mask, ref_emb}
  reference patch --> reference encoder --> ref_emb

At inference, instances are the queries classified as foreground; the ones whose
ref_emb matches the encoded reference patch (cosine sim above a threshold) are the
instances of the user-selected pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNetBackbone
from .pixel_decoder import FPNPixelDecoder
from .transformer_decoder import TransformerDecoder
from .reference_encoder import ReferenceEncoder


def _downsample_mask(padding_mask, size):
    """padding_mask: [B, H, W] bool (True=pad) -> [B, h, w] bool at `size`."""
    m = F.interpolate(padding_mask[:, None].float(), size=size, mode="nearest")
    return m[:, 0].bool()


class RefMask2Former(nn.Module):
    def __init__(self, num_queries=100, hidden_dim=256, mask_dim=256, ref_dim=128,
                 nheads=8, dec_layers=9, dim_feedforward=2048, pretrained=True,
                 stem_pool="max", ref_pool_features=False,
                 ref_siamese_backbone=False, ref_siamese_level="res5",
                 ref_siamese_stats=False, ref_texture_backbone=False):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained, stem_pool=stem_pool)
        self.pixel_decoder = FPNPixelDecoder(
            self.backbone.out_channels, conv_dim=hidden_dim, mask_dim=mask_dim)
        self.transformer = TransformerDecoder(
            d_model=hidden_dim, nhead=nheads, num_layers=dec_layers,
            num_queries=num_queries, num_feature_levels=3,
            mask_dim=mask_dim, ref_dim=ref_dim, dim_feedforward=dim_feedforward,
            ref_pool_features=ref_pool_features)
        self.reference_encoder = ReferenceEncoder(ref_dim=ref_dim, pretrained=pretrained)
        self.ref_siamese_backbone = ref_siamese_backbone
        self.ref_siamese_level = ref_siamese_level
        self.ref_siamese_stats = ref_siamese_stats
        self.ref_texture_backbone = ref_texture_backbone
        self.ref_pairwise_head = False
        if ref_siamese_backbone:
            self._siamese_levels = (ref_siamese_level.split("+")
                                    if "+" in ref_siamese_level
                                    else [ref_siamese_level])
            siamese_channels = sum(self.backbone.out_channels[level]
                                   for level in self._siamese_levels)
            if ref_siamese_stats:
                siamese_channels *= 2
            self.siamese_ref_projector = nn.Sequential(
                nn.Linear(siamese_channels, 512), nn.ReLU(inplace=True),
                nn.Dropout(0.2), nn.Linear(512, ref_dim))
            if ref_texture_backbone:
                self.texture_backbone = ResNetBackbone(
                    pretrained=pretrained, stem_pool=stem_pool)
                self.texture_backbone.requires_grad_(False)

    def enable_pairwise_match_head(self, ref_dim=128):
        """Install an expressive reference/query comparison head."""
        self.ref_pairwise_head = True
        self.siamese_match_head = nn.Sequential(
            nn.Linear(ref_dim * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1))

        # Strides of the levels the pixel decoder feeds to the transformer.
        self._level_strides = [self.backbone.out_strides[n]
                               for n in self.pixel_decoder.transformer_levels]
        self._mask_stride = self.backbone.out_strides["res2"]

    def forward(self, images, pixel_mask, reference=None):
        """
        images: [B, 3, H, W]
        pixel_mask: [B, H, W] bool, True = real pixel (valid), False = padding
        reference: [B, 3, Rh, Rw] or None
        """
        padding = ~pixel_mask  # True = pad

        feats = self.backbone(images)
        mask_features, multi_scale = self.pixel_decoder(feats)

        ms_padding = [_downsample_mask(padding, f.shape[-2:]) for f in multi_scale]
        mask_features_padding = _downsample_mask(padding, mask_features.shape[-2:])

        out = self.transformer(multi_scale, ms_padding, mask_features,
                               mask_features_padding)

        if reference is not None:
            if self.ref_siamese_backbone:
                # Encode reference and candidate regions with the same backbone
                # and projection. This makes cosine similarity compare appearance
                # in one genuine Siamese feature space instead of aligning an FPN
                # query token with an independently trained reference ResNet.
                if self.ref_texture_backbone:
                    self.texture_backbone.eval()
                    with torch.no_grad():
                        reference_maps = self.texture_backbone(reference)
                        texture_maps = self.texture_backbone(images)
                else:
                    reference_maps = self.backbone(reference)
                    texture_maps = feats
                reference_parts = []
                for level in self._siamese_levels:
                    fmap = reference_maps[level]
                    if self.ref_siamese_stats:
                        # Style-stat gradients (especially sqrt(var)) are poorly
                        # conditioned in bf16. Keep segmentation backbone updates
                        # driven by mask losses and train the texture projector on
                        # detached, stable feature statistics.
                        fmap = fmap.detach()
                    reference_parts.append(fmap.mean(dim=(-2, -1)))
                    if self.ref_siamese_stats:
                        reference_parts.append(fmap.var(dim=(-2, -1), unbiased=False).sqrt())
                reference_feat = torch.cat(reference_parts, dim=1)
                out["reference_emb"] = F.normalize(
                    self.siamese_ref_projector(reference_feat), dim=-1)

                def _region_embeddings(mask_logits):
                    pooled_levels = []
                    for level in self._siamese_levels:
                        image_map = texture_maps[level]
                        if self.ref_siamese_stats:
                            image_map = image_map.detach()
                        level_padding = _downsample_mask(
                            padding, image_map.shape[-2:])
                        weights = F.interpolate(
                            mask_logits, size=image_map.shape[-2:],
                            mode="bilinear", align_corners=False)
                        weights = weights.sigmoid().detach()
                        weights = weights.masked_fill(
                            level_padding[:, None], 0.0)
                        denom = weights.flatten(2).sum(-1).clamp(min=1e-6)
                        pooled_level = torch.einsum(
                            "bqhw,bchw->bqc", weights, image_map)
                        mean = pooled_level / denom[..., None]
                        pooled_levels.append(mean)
                        if self.ref_siamese_stats:
                            second = torch.einsum(
                                "bqhw,bchw->bqc", weights, image_map.square())
                            second = second / denom[..., None]
                            pooled_levels.append(
                                (second - mean.square()).clamp(min=1e-6).sqrt())
                    pooled = torch.cat(pooled_levels, dim=-1)
                    return F.normalize(self.siamese_ref_projector(pooled), dim=-1)

                out["pred_ref"] = _region_embeddings(out["pred_masks"])
                for aux in out.get("aux_outputs", []):
                    aux["pred_ref"] = _region_embeddings(aux["pred_masks"])
                if self.ref_pairwise_head:
                    def _pair_logits(query_emb):
                        ref_emb = out["reference_emb"][:, None].expand_as(query_emb)
                        pair = torch.cat([query_emb, ref_emb,
                                          (query_emb - ref_emb).abs(),
                                          query_emb * ref_emb], dim=-1)
                        return self.siamese_match_head(pair).squeeze(-1)
                    out["pred_match_logits"] = _pair_logits(out["pred_ref"])
                    for aux in out.get("aux_outputs", []):
                        aux["pred_match_logits"] = _pair_logits(aux["pred_ref"])
            else:
                out["reference_emb"] = self.reference_encoder(reference)  # [B, ref_dim]
        return out

    @torch.no_grad()
    def predict(self, images, pixel_mask, reference, score_thresh=0.5,
                match_thresh=0.0, mask_thresh=0.5):
        """Convenience inference: returns per-image matched instances.

        Returns a list (len B) of dicts with keys: masks [n, H, W] (bool),
        scores [n], match_sim [n]. Instances are those with foreground score >
        score_thresh AND cosine similarity to the reference > match_thresh.

        match_thresh defaults to 0.0 — the orthogonality boundary for the
        normalized embeddings. Empirically true-match sims center near +0.32 and
        non-match near -0.43 (AUC ~0.99), so 0.0 separates them cleanly; the old
        0.5 default sat above the match cluster and rejected most true matches.
        """
        self.eval()
        out = self.forward(images, pixel_mask, reference)
        # Foreground prob = softmax over [no-object, pattern] -> idx 1.
        probs = out["pred_logits"].softmax(-1)[..., 1]     # [B, Q]
        masks = out["pred_masks"]                          # [B, Q, Hm, Wm]
        ref_q = out["pred_ref"]                            # [B, Q, ref_dim]
        ref_g = out["reference_emb"]                       # [B, ref_dim]
        sim = torch.einsum("bqd,bd->bq", ref_q, ref_g)     # cosine (both normalized)

        H, W = images.shape[-2:]
        masks_up = F.interpolate(masks, size=(H, W), mode="bilinear",
                                 align_corners=False).sigmoid()

        results = []
        for b in range(images.shape[0]):
            keep = (probs[b] > score_thresh) & (sim[b] > match_thresh)
            results.append({
                "masks": (masks_up[b][keep] > mask_thresh),
                "scores": probs[b][keep],
                "match_sim": sim[b][keep],
            })
        return results
