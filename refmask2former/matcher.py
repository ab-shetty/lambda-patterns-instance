"""Hungarian matcher for class-agnostic instance segmentation.

Matches predicted queries to ground-truth instances using a class cost
(foreground probability) plus mask costs (sigmoid-BCE and Dice), both evaluated
on a shared set of randomly sampled points to bound memory.
"""

import torch
from scipy.optimize import linear_sum_assignment

from .point_utils import point_sample, sample_uniform_points


def _batch_sigmoid_bce_cost(pred, gt):
    """pred: [Q, P] logits; gt: [G, P] in {0,1}. returns [Q, G] mean BCE."""
    pos = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, torch.ones_like(pred), reduction="none")
    neg = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, torch.zeros_like(pred), reduction="none")
    cost = torch.einsum("qp,gp->qg", pos, gt) + torch.einsum("qp,gp->qg", neg, 1 - gt)
    return cost / pred.shape[1]


def _batch_dice_cost(pred, gt):
    """pred: [Q, P] logits; gt: [G, P]. returns [Q, G] dice cost."""
    pred = pred.sigmoid()
    numerator = 2 * torch.einsum("qp,gp->qg", pred, gt)
    denom = pred.sum(-1)[:, None] + gt.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denom + 1)


class HungarianMatcher:
    def __init__(self, cost_class=2.0, cost_mask=5.0, cost_dice=5.0, num_points=12544):
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def __call__(self, outputs, targets):
        """
        outputs: dict with 'pred_logits' [B,Q,2], 'pred_masks' [B,Q,Hm,Wm]
        targets: list (len B) of dicts with 'masks' [G, Hm, Wm] (float, mask res)
        returns: list of (pred_idx LongTensor, gt_idx LongTensor)
        """
        B, Q = outputs["pred_logits"].shape[:2]
        indices = []
        for b in range(B):
            tgt_mask = targets[b]["masks"]            # [G, Hm, Wm]
            G = tgt_mask.shape[0]
            if G == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            fg_prob = outputs["pred_logits"][b].softmax(-1)[:, 1]   # [Q]
            cost_class = -fg_prob[:, None].expand(Q, G)             # [Q, G]

            pred_mask = outputs["pred_masks"][b]                    # [Q, Hm, Wm]
            device = pred_mask.device
            points = sample_uniform_points(self.num_points, device)  # [P, 2]
            pred_pts = point_sample(pred_mask.unsqueeze(1),
                                    points[None].expand(Q, -1, -1))  # [Q, P]
            gt_pts = point_sample(tgt_mask.unsqueeze(1).float(),
                                  points[None].expand(G, -1, -1))    # [G, P]

            cost_mask = _batch_sigmoid_bce_cost(pred_pts, gt_pts)    # [Q, G]
            cost_dice = _batch_dice_cost(pred_pts, gt_pts)           # [Q, G]

            C = (self.cost_class * cost_class
                 + self.cost_mask * cost_mask
                 + self.cost_dice * cost_dice)
            C = C.cpu().numpy()
            row, col = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row, dtype=torch.long),
                            torch.as_tensor(col, dtype=torch.long)))
        return indices
