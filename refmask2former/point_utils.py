"""Point sampling helpers (memory-bounded mask losses, Mask2Former style)."""

import torch
import torch.nn.functional as F


def point_sample(masks, points):
    """Sample masks at normalized point coords.

    masks: [K, 1, H, W] float; points: [K, P, 2] in [0, 1].
    returns: [K, P]
    """
    coords = 2.0 * points - 1.0                       # [-1, 1] for grid_sample
    grid = coords.unsqueeze(2)                         # [K, P, 1, 2]
    out = F.grid_sample(masks, grid, align_corners=False, mode="bilinear",
                        padding_mode="border")         # [K, 1, P, 1]
    return out[:, 0, :, 0]


def sample_uniform_points(num_points, device):
    """Random uniform points in the unit square: [P, 2]."""
    return torch.rand(num_points, 2, device=device)
