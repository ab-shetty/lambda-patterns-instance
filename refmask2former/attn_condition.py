"""Attention conditioning for the reference-conditioned decoders.

`CrossAttnCondition` is the `selfattn` / `crossattn` block from
`scripts/decoder_search.py` (moved here so `RefSwinUNet` can use it too):
image locations attend to the reference's 7x7 spatial tokens, and optionally
image tokens then attend to each other -- the only mechanism in this repo
that lets two distant regions of a sheet be compared directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnCondition(nn.Module):
    """Image locations query the reference's spatial tokens."""

    def __init__(self, image_channels, ref_channels, width, num_heads=4,
                 self_attn=False):
        super().__init__()
        self.width, self.num_heads = width, num_heads
        self.image = nn.Conv2d(image_channels, width, 1)
        self.reference = nn.Linear(ref_channels, width)
        self.kv = nn.Conv2d(ref_channels, width * 2, 1)
        self.attn_out = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.self_attn = self_attn
        if self_attn:
            self.qkv_self = nn.Conv2d(width, width * 3, 1)
            self.self_out = nn.Conv2d(width, width, 1)
            self.self_norm = nn.GroupNorm(8, width)
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU())

    def _heads(self, t, b, n):
        return t.reshape(b, self.num_heads, self.width // self.num_heads, n)

    def forward(self, image, reference_vector, reference_features):
        x = self.image(image)
        b, c, h, w = x.shape
        # Reference tokens are pooled to a small grid: attention cost then does
        # not depend on the reference crop's size.
        ref = F.adaptive_avg_pool2d(reference_features, 7)
        k, v = self.kv(ref).chunk(2, dim=1)
        q = self._heads(x.flatten(2), b, h * w)
        k = self._heads(k.flatten(2), b, 49)
        v = self._heads(v.flatten(2), b, 49)
        # .contiguous(): a transposed (non-contiguous last dim) input rules out
        # the flash / memory-efficient kernels and SDPA materialises the full
        # attention matrix (27 GiB for self-attention at 1/16 of a 2048 sheet).
        a = F.scaled_dot_product_attention(q.transpose(-2, -1).contiguous(),
                                           k.transpose(-2, -1).contiguous(),
                                           v.transpose(-2, -1).contiguous())
        a = a.transpose(-2, -1).reshape(b, c, h, w)
        x = self.norm(x + self.attn_out(a))
        if self.self_attn:
            q2, k2, v2 = self.qkv_self(x).chunk(3, dim=1)
            n = h * w
            q2 = self._heads(q2.flatten(2), b, n).transpose(-2, -1).contiguous()
            k2 = self._heads(k2.flatten(2), b, n).transpose(-2, -1).contiguous()
            v2 = self._heads(v2.flatten(2), b, n).transpose(-2, -1).contiguous()
            s = F.scaled_dot_product_attention(q2, k2, v2)
            s = s.transpose(-2, -1).reshape(b, c, h, w)
            x = self.self_norm(x + self.self_out(s))
        # Keep the broadcast prototype alongside the attended features, so this
        # differs from `ConditionBlock` only by ADDING attention -- otherwise a
        # win could just be the missing global term.
        r = self.reference(reference_vector)[:, :, None, None].expand_as(x)
        return self.fuse(torch.cat([x, r], dim=1))
