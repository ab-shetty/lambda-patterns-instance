"""Dataset + collation for the floz-synth-v5 instance-segmentation data.

Loads the HuggingFace parquet (image stored as PNG bytes, annotations as a JSON
string). Each annotation is one instance. For each sample we also pick a "target
pattern" (a category_name present in the image) and crop a reference patch from
one of its instances; every instance is labelled match/no-match against it, which
trains the reference-matching head.

Preprocessing preserves aspect ratio (longest side -> image_max_size); the
collate fn pads a batch to a common size divisible by 32 and returns a pixel mask.
"""

import io
import json
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Mask + reference helpers
# --------------------------------------------------------------------------- #
def render_instance_mask(segmentation, height, width):
    """COCO polygons -> binary mask. First polygon is outer, the rest are holes."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not segmentation:
        return mask
    outer = np.array(segmentation[0], dtype=np.float32).reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(mask, [outer], 1)
    for hole in segmentation[1:]:
        hp = np.array(hole, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [hp], 0)
    return mask


def sample_reference_box(mask, min_size=128, max_size=512):
    """Random box fully inside a single instance mask. Returns (x, y, w, h)."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, 0, 32, 32)
    ys, xs = coords[:, 0], coords[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pw, ph = x1 - x0 + 1, y1 - y0 + 1
    max_size = max(min(max_size, int(pw * 0.7), int(ph * 0.7)), min_size)

    for _ in range(200):
        size = int(min_size + (max_size - min_size) * (random.random() ** 0.5))
        cy, cx = coords[random.randint(0, len(coords) - 1)]
        x = int(np.clip(cx - size // 2, 0, mask.shape[1] - size))
        y = int(np.clip(cy - size // 2, 0, mask.shape[0] - size))
        if mask[y:y + size, x:x + size].mean() == 1.0:
            return (x, y, size, size)

    for frac in (0.6, 0.5, 0.4, 0.3, 0.2):
        tw, th = max(min_size, int(pw * frac)), max(min_size, int(ph * frac))
        for _ in range(50):
            x = random.randint(x0, max(x0, x1 - tw))
            y = random.randint(y0, max(y0, y1 - th))
            if mask[y:y + th, x:x + tw].mean() == 1.0:
                return (x, y, tw, th)

    cy, cx = coords.mean(0).astype(int)
    return (max(0, cx - 16), max(0, cy - 16), 32, 32)


def _spatial_augment(image, masks):
    """Apply the same flip/rot90 to image [H,W,3] and masks [G,H,W]."""
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        masks = np.ascontiguousarray(masks[:, :, ::-1])
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[::-1])
        masks = np.ascontiguousarray(masks[:, ::-1])
    k = random.randint(0, 3)
    if k:
        image = np.ascontiguousarray(np.rot90(image, k))
        masks = np.ascontiguousarray(np.rot90(masks, k, axes=(1, 2)))
    return image, masks


def _augment_reference(patch):
    """Orientation augmentation on the reference patch (independent of the image)."""
    if random.random() < 0.5:
        patch = patch[:, ::-1]
    if random.random() < 0.5:
        patch = patch[::-1]
    k = random.randint(0, 3)
    if k:
        patch = np.rot90(patch, k)
    return np.ascontiguousarray(patch)


def _normalize_chw(img_uint8):
    img = img_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class InstanceSegDataset(Dataset):
    def __init__(self, records, indices, image_max_size=1024, ref_size=224,
                 augment=True, min_patch=128, max_patch=512):
        self.records = records
        self.indices = list(indices)
        self.image_max_size = image_max_size
        self.ref_size = ref_size
        self.augment = augment
        self.min_patch = min_patch
        self.max_patch = max_patch

    def __len__(self):
        return len(self.indices)

    def _load(self, i):
        rec = self.records[self.indices[i]]
        image = np.array(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = json.loads(rec["annotations"])
        return image, anns

    def __getitem__(self, i):
        image, anns = self._load(i)
        h0, w0 = image.shape[:2]

        # Render instance masks at original resolution.
        masks = [render_instance_mask(a["segmentation"], h0, w0) for a in anns]
        cats = [a.get("category_name", "pattern") for a in anns]

        # Pick a target pattern and a reference patch cropped from one of its instances.
        if len(anns) > 0:
            target_cat = random.choice(list(dict.fromkeys(cats)))
            cand = [j for j, c in enumerate(cats) if c == target_cat]
            ref_idx = random.choice(cand)
            bx, by, bw, bh = sample_reference_box(masks[ref_idx], self.min_patch, self.max_patch)
            ref_patch = image[by:by + bh, bx:bx + bw].copy()
            ref_match = np.array([1.0 if c == target_cat else 0.0 for c in cats], np.float32)
        else:
            target_cat = None
            ref_patch = image[:min(h0, 224), :min(w0, 224)].copy()
            ref_match = np.zeros((0,), np.float32)

        # Aspect-preserving resize of image + masks.
        scale = self.image_max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
        image_r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if masks:
            masks_arr = np.stack([cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
                                  for m in masks], 0)
        else:
            masks_arr = np.zeros((0, nh, nw), np.uint8)

        # Reference patch resize.
        ref_r = cv2.resize(ref_patch, (self.ref_size, self.ref_size),
                           interpolation=cv2.INTER_LINEAR)

        if self.augment:
            image_r, masks_arr = _spatial_augment(image_r, masks_arr)
            ref_r = _augment_reference(ref_r)

        return {
            "image": _normalize_chw(image_r),                       # [3, nh, nw]
            "masks": torch.from_numpy(masks_arr).to(torch.uint8),   # [G, nh, nw]
            "ref_match": torch.from_numpy(ref_match),               # [G]
            "reference": _normalize_chw(ref_r),                     # [3, R, R]
        }


# --------------------------------------------------------------------------- #
# Collation (dynamic padding + pixel mask)
# --------------------------------------------------------------------------- #
def collate_fn(batch, size_divisible=32):
    heights = [b["image"].shape[1] for b in batch]
    widths = [b["image"].shape[2] for b in batch]

    def _round_up(x):
        return int(np.ceil(x / size_divisible) * size_divisible)

    maxH, maxW = _round_up(max(heights)), _round_up(max(widths))
    B = len(batch)

    images = torch.zeros(B, 3, maxH, maxW)
    pixel_mask = torch.zeros(B, maxH, maxW, dtype=torch.bool)
    references = torch.stack([b["reference"] for b in batch], 0)

    targets = []
    for b, sample in enumerate(batch):
        _, h, w = sample["image"].shape
        images[b, :, :h, :w] = sample["image"]
        pixel_mask[b, :h, :w] = True

        g = sample["masks"].shape[0]
        padded = torch.zeros(g, maxH, maxW, dtype=torch.uint8)
        if g > 0:
            padded[:, :h, :w] = sample["masks"]
        targets.append({
            "masks": padded,                                  # [G, maxH, maxW]
            "labels": torch.ones(g, dtype=torch.long),        # all foreground
            "ref_match": sample["ref_match"],                 # [G]
        })

    return {"images": images, "pixel_mask": pixel_mask,
            "references": references, "targets": targets}


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def load_parquet_records(repo_id="abshetty/floz-synth-v5", cache_dir=None):
    """Load the full parquet split (image bytes + annotation strings)."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, split="train", cache_dir=cache_dir)
    return ds


def build_datasets(records, image_max_size=1024, ref_size=224, train_split=0.9,
                   seed=42):
    n = len(records)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(n * train_split)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_ds = InstanceSegDataset(records, train_idx, image_max_size, ref_size,
                                  augment=True)
    val_ds = InstanceSegDataset(records, val_idx, image_max_size, ref_size,
                                augment=False)
    return train_ds, val_ds
