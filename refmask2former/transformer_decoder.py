"""Mask2Former-style transformer decoder with masked attention.

N learnable queries iteratively attend to multi-scale image features. Each
decoder layer restricts cross-attention to the foreground region predicted by
the previous layer (masked attention). Three prediction heads are attached to
every layer output (deep supervision):
  - class head: pattern vs. no-object (class-agnostic, single foreground class)
  - mask head: per-query mask embedding, dotted with the pixel-decoder features
  - reference head: a normalized embedding used to match queries to a reference
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    """2D sine positional embedding, normalized over the valid (non-pad) region."""

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, valid_mask):
        # valid_mask: [B, H, W] bool, True = real pixel.
        not_mask = valid_mask.to(torch.float32)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32,
                             device=valid_mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [B, 2*npf, H, W]
        return pos


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query_pos):
        q = k = tgt + query_pos
        tgt2 = self.attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_pos, query_pos, attn_mask):
        tgt2 = self.attn(query=tgt + query_pos,
                         key=memory + memory_pos,
                         value=memory,
                         attn_mask=attn_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=9, num_queries=100,
                 num_feature_levels=3, mask_dim=256, ref_dim=128,
                 dim_feedforward=2048):
        super().__init__()
        self.num_heads = nhead
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels

        self.pe_layer = PositionEmbeddingSine(d_model // 2)

        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead))
            self.ffn_layers.append(FFNLayer(d_model, dim_feedforward))

        self.decoder_norm = nn.LayerNorm(d_model)

        self.query_feat = nn.Embedding(num_queries, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)  # positional
        self.level_embed = nn.Embedding(num_feature_levels, d_model)

        # Project each input level to d_model (in case channels differ).
        self.input_proj = nn.ModuleList(
            nn.Conv2d(d_model, d_model, kernel_size=1) for _ in range(num_feature_levels))

        # Prediction heads (shared across layers).
        self.class_embed = nn.Linear(d_model, 2)          # [no-object=0? -> we use idx1=pattern]
        self.mask_embed = MLP(d_model, d_model, mask_dim, 3)
        self.ref_embed = MLP(d_model, d_model, ref_dim, 3)

    def _pred_heads(self, output, mask_features, mask_padding, target_size):
        """Compute class/mask/ref for current queries and the next attn mask.

        output: [Q, B, C]; mask_features: [B, mask_dim, Hm, Wm];
        mask_padding: [B, Hm, Wm] bool True=pad; target_size: (h, w) of next level.
        """
        decoder_output = self.decoder_norm(output).transpose(0, 1)  # [B, Q, C]
        cls = self.class_embed(decoder_output)                      # [B, Q, 2]
        mask_emb = self.mask_embed(decoder_output)                  # [B, Q, mask_dim]
        ref = F.normalize(self.ref_embed(decoder_output), dim=-1)   # [B, Q, ref_dim]
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_emb, mask_features)

        # Build masked-attention mask at the target level resolution.
        attn = F.interpolate(outputs_mask, size=target_size, mode="bilinear",
                             align_corners=False)  # [B, Q, h, w]
        # Block (True) where predicted foreground prob < 0.5, OR padded.
        pad = F.interpolate(mask_padding[:, None].float(), size=target_size,
                            mode="nearest")[:, 0].bool()  # [B, h, w]
        attn_mask = (attn.sigmoid() < 0.5)
        attn_mask = attn_mask | pad[:, None]               # [B, Q, h, w]
        B, Q = attn_mask.shape[:2]
        attn_mask = attn_mask.flatten(2)                   # [B, Q, h*w]
        # Unblock any query whose entire row is masked (avoids NaN attention).
        fully_masked = attn_mask.all(dim=-1, keepdim=True)
        attn_mask = attn_mask & ~fully_masked
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attn_mask = attn_mask.flatten(0, 1)                # [B*nheads, Q, h*w]
        return cls, outputs_mask, ref, attn_mask.detach()

    def forward(self, multi_scale_feats, multi_scale_padding, mask_features,
                mask_features_padding):
        """
        multi_scale_feats: list of [B, C, H_l, W_l] (coarse->fine, len=num_levels)
        multi_scale_padding: list of [B, H_l, W_l] bool, True = pad
        mask_features: [B, mask_dim, Hm, Wm]
        mask_features_padding: [B, Hm, Wm] bool, True = pad
        """
        B = mask_features.shape[0]
        srcs, poss, sizes = [], [], []
        for l in range(self.num_feature_levels):
            feat = self.input_proj[l](multi_scale_feats[l])
            valid = ~multi_scale_padding[l]
            pos = self.pe_layer(valid)  # [B, C, H, W]
            h, w = feat.shape[-2:]
            sizes.append((h, w))
            src = feat.flatten(2).permute(2, 0, 1)            # [HW, B, C]
            pos = pos.flatten(2).permute(2, 0, 1)             # [HW, B, C]
            pos = pos + self.level_embed.weight[l][None, None]
            srcs.append(src)
            poss.append(pos)

        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)   # [Q, B, C]
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)   # [Q, B, C]

        predictions_class, predictions_mask, predictions_ref = [], [], []

        # Initial prediction (gives attn mask for the first layer).
        cls, mask, ref, attn_mask = self._pred_heads(
            query_feat, mask_features, mask_features_padding, sizes[0])
        predictions_class.append(cls)
        predictions_mask.append(mask)
        predictions_ref.append(ref)

        output = query_feat
        for i in range(self.num_layers):
            level = i % self.num_feature_levels
            output = self.cross_attn_layers[i](
                output, srcs[level], poss[level], query_pos, attn_mask)
            output = self.self_attn_layers[i](output, query_pos)
            output = self.ffn_layers[i](output)

            next_level = (i + 1) % self.num_feature_levels
            cls, mask, ref, attn_mask = self._pred_heads(
                output, mask_features, mask_features_padding, sizes[next_level])
            predictions_class.append(cls)
            predictions_mask.append(mask)
            predictions_ref.append(ref)

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_ref": predictions_ref[-1],
            "aux_outputs": [
                {"pred_logits": c, "pred_masks": m, "pred_ref": r}
                for c, m, r in zip(predictions_class[:-1],
                                   predictions_mask[:-1],
                                   predictions_ref[:-1])
            ],
        }
        return out
