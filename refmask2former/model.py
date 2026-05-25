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
                 nheads=8, dec_layers=9, dim_feedforward=2048, pretrained=True):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.pixel_decoder = FPNPixelDecoder(
            self.backbone.out_channels, conv_dim=hidden_dim, mask_dim=mask_dim)
        self.transformer = TransformerDecoder(
            d_model=hidden_dim, nhead=nheads, num_layers=dec_layers,
            num_queries=num_queries, num_feature_levels=3,
            mask_dim=mask_dim, ref_dim=ref_dim, dim_feedforward=dim_feedforward)
        self.reference_encoder = ReferenceEncoder(ref_dim=ref_dim, pretrained=pretrained)

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
            out["reference_emb"] = self.reference_encoder(reference)  # [B, ref_dim]
        return out

    @torch.no_grad()
    def predict(self, images, pixel_mask, reference, score_thresh=0.5,
                match_thresh=0.5, mask_thresh=0.5):
        """Convenience inference: returns per-image matched instances.

        Returns a list (len B) of dicts with keys: masks [n, H, W] (bool),
        scores [n], match_sim [n]. Instances are those with foreground score >
        score_thresh AND cosine similarity to the reference > match_thresh.
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
