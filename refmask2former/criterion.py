"""Set criterion: class + mask (BCE/Dice) + reference-matching losses.

Matching is recomputed for every decoder layer (deep supervision). Mask losses
use shared random point sampling to stay memory-bounded at high resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .point_utils import point_sample, sample_uniform_points


def _dice_loss(pred_logits, gt):
    pred = pred_logits.sigmoid()
    num = 2 * (pred * gt).sum(-1)
    den = pred.sum(-1) + gt.sum(-1)
    return (1 - (num + 1) / (den + 1)).mean()


def _strip_aux(key):
    # 'aux3_loss_mask' -> 'loss_mask'
    if key.startswith("aux"):
        return key.split("_", 1)[1]
    return key


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, eos_coef=0.1, num_points=12544,
                 ref_scale=10.0):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.num_points = num_points
        self.ref_scale = ref_scale
        self.register_buffer("empty_weight", torch.tensor([eos_coef, 1.0]))

    def _loss_labels(self, outputs, targets, indices):
        src = outputs["pred_logits"]                     # [B, Q, 2]
        B, Q, _ = src.shape
        target_classes = torch.zeros(B, Q, dtype=torch.long, device=src.device)
        for b, (pi, gi) in enumerate(indices):
            target_classes[b, pi] = 1                    # foreground = pattern
        return F.cross_entropy(src.transpose(1, 2), target_classes,
                               weight=self.empty_weight)

    def _loss_masks(self, outputs, targets, indices):
        pred_pts, gt_pts = [], []
        for b, (pi, gi) in enumerate(indices):
            if len(pi) == 0:
                continue
            pred = outputs["pred_masks"][b][pi]          # [m, Hm, Wm]
            gt = targets[b]["masks"][gi].float()         # [m, H, W]
            pts = sample_uniform_points(self.num_points, pred.device)
            m = len(pi)
            pred_pts.append(point_sample(pred.unsqueeze(1), pts[None].expand(m, -1, -1)))
            gt_pts.append(point_sample(gt.unsqueeze(1), pts[None].expand(m, -1, -1)))
        if not pred_pts:
            zero = outputs["pred_masks"].sum() * 0.0
            return zero, zero
        pred_pts = torch.cat(pred_pts, 0)                # [M, P]
        gt_pts = torch.cat(gt_pts, 0)
        bce = F.binary_cross_entropy_with_logits(pred_pts, gt_pts)
        dice = _dice_loss(pred_pts, gt_pts)
        return bce, dice

    def _loss_ref(self, outputs, targets, indices):
        if "reference_emb" not in outputs:
            return outputs["pred_ref"].sum() * 0.0
        ref_g = outputs["reference_emb"]                 # [B, ref_dim]
        pred_ref = outputs["pred_ref"]                   # [B, Q, ref_dim]
        logits, labels = [], []
        for b, (pi, gi) in enumerate(indices):
            if len(pi) == 0:
                continue
            sim = (pred_ref[b][pi] * ref_g[b][None]).sum(-1)   # cosine [m]
            logits.append(self.ref_scale * sim)
            labels.append(targets[b]["ref_match"][gi].float())
        if not logits:
            return pred_ref.sum() * 0.0
        return F.binary_cross_entropy_with_logits(torch.cat(logits), torch.cat(labels))

    def _compute(self, outputs, targets, indices, prefix=""):
        ce = self._loss_labels(outputs, targets, indices)
        bce, dice = self._loss_masks(outputs, targets, indices)
        ref = self._loss_ref(outputs, targets, indices)
        return {
            f"{prefix}loss_ce": ce,
            f"{prefix}loss_mask": bce,
            f"{prefix}loss_dice": dice,
            f"{prefix}loss_ref": ref,
        }

    def forward(self, outputs, targets):
        losses = {}
        indices = self.matcher(outputs, targets)
        losses.update(self._compute(outputs, targets, indices))

        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux = dict(aux)
            if "reference_emb" in outputs:
                aux["reference_emb"] = outputs["reference_emb"]
            aux_indices = self.matcher(aux, targets)
            losses.update(self._compute(aux, targets, aux_indices, prefix=f"aux{i}_"))

        total = sum(losses[k] * self.weight_dict[_strip_aux(k)] for k in losses)
        return total, losses
