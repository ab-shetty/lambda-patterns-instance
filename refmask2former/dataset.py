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


def sample_reference_box(mask, min_size=128, max_size=512, rng=None):
    """Random box fully inside a single instance mask. Returns (x, y, w, h).
    Pass `rng` (a random.Random) to make the box DETERMINISTIC (eval); None uses
    the global RNG (training).

    `min_size` is a preference, not a floor. A Chebyshev distance transform gives
    the half-width of the largest square that fits at every pixel, so the largest
    feasible box is known up front instead of being searched for: the box size is
    drawn from what actually fits and the centre is drawn only from pixels deep
    enough to hold it. The returned box is therefore always fully inside the mask.

    The previous version clamped size to `max(..., min_size)` in both the random
    search and the shrink ladder, so any region thinner than `min_size` -- and any
    region whose feasible sizes were a sliver of the sampled range -- exhausted all
    450 attempts and fell through to an unchecked `centroid +- 16` box. On
    ring-shaped regions that centroid sits in the hole, producing reference crops
    containing none of the pattern (94% of real-plan instances hit this path). Use
    `sample_reference_box_legacy` to reproduce numbers measured before the fix."""
    r = rng if rng is not None else random
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if not m.any():
        return (0, 0, 32, 32)

    # Zero border so the transform never reports room that runs off the image.
    padded = np.pad(m, 1, mode="constant", constant_values=0)
    dt = cv2.distanceTransform(padded, cv2.DIST_C, 3)[1:-1, 1:-1]

    # dt[p] == d means every pixel within Chebyshev distance d-1 of p is inside,
    # so p can host a square of half-width d-1, i.e. side 2*(d-1)+1.
    max_half = int(dt.max()) - 1
    if max_half < 0:
        ys, xs = np.nonzero(m)
        i = r.randint(0, len(ys) - 1)
        return (int(xs[i]), int(ys[i]), 1, 1)

    # Sample the half-width and derive an odd side, so the box is exactly
    # symmetric about its centre; an even side would extend one pixel further
    # right/down than `half` and could cross the mask edge.
    hi_half = min((max_size - 1) // 2, max_half)
    lo_half = min((min_size - 1) // 2, hi_half)
    half = (int(lo_half + (hi_half - lo_half) * (r.random() ** 0.5))
            if hi_half > lo_half else hi_half)
    half = max(0, min(half, hi_half))

    centers = np.argwhere(dt >= half + 1)
    cy, cx = centers[r.randint(0, len(centers) - 1)]
    size = 2 * half + 1
    return (int(cx - half), int(cy - half), size, size)


def sample_reference_box_legacy(mask, min_size=128, max_size=512, rng=None):
    """Pre-fix sampler, kept only to reproduce historical measurements.

    Can return a box that is partially or entirely outside `mask`; see
    `sample_reference_box` for why."""
    r = rng if rng is not None else random
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, 0, 32, 32)
    ys, xs = coords[:, 0], coords[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pw, ph = x1 - x0 + 1, y1 - y0 + 1
    max_size = max(min(max_size, int(pw * 0.7), int(ph * 0.7)), min_size)

    for _ in range(200):
        size = int(min_size + (max_size - min_size) * (r.random() ** 0.5))
        cy, cx = coords[r.randint(0, len(coords) - 1)]
        x = int(np.clip(cx - size // 2, 0, mask.shape[1] - size))
        y = int(np.clip(cy - size // 2, 0, mask.shape[0] - size))
        if mask[y:y + size, x:x + size].mean() == 1.0:
            return (x, y, size, size)

    for frac in (0.6, 0.5, 0.4, 0.3, 0.2):
        tw, th = max(min_size, int(pw * frac)), max(min_size, int(ph * frac))
        for _ in range(50):
            x = r.randint(x0, max(x0, x1 - tw))
            y = r.randint(y0, max(y0, y1 - th))
            if mask[y:y + th, x:x + tw].mean() == 1.0:
                return (x, y, tw, th)

    cy, cx = coords.mean(0).astype(int)
    return (max(0, cx - 16), max(0, cy - 16), 32, 32)


def fit_canvas(patch, size, pad_value=255):
    """Centre a patch on a `size x size` canvas without changing its scale.

    Centre-crops when the patch is larger. Pads with paper-white otherwise, which
    is what actually surrounds a material region on a plan.
    """
    h, w = patch.shape[:2]
    if h > size:
        top = (h - size) // 2
        patch = patch[top:top + size]
        h = size
    if w > size:
        left = (w - size) // 2
        patch = patch[:, left:left + size]
        w = size
    if h == size and w == size:
        return patch
    canvas = np.full((size, size) + patch.shape[2:], pad_value, patch.dtype)
    top, left = (size - h) // 2, (size - w) // 2
    canvas[top:top + h, left:left + w] = patch
    return canvas


def scale_matched_reference(patch, scale, size, min_side=64):
    """Resize a native-resolution reference crop to the IMAGE's pixel scale.

    The reference used to be resized to `size x size` regardless of its native
    extent, so the hatch appeared a median of 5.6x larger in the reference than
    in the plan the model sees (75% of selections were off by more than 3x).
    Template matching -- engineered or learned -- cannot work across that gap.
    Applying the image's own scale factor puts both at the same pixel scale;
    `min_side` stops very small crops from collapsing to a few pixels.
    """
    h, w = patch.shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if max(nh, nw) < min_side:
        boost = min_side / max(nh, nw, 1)
        nh, nw = max(1, round(nh * boost)), max(1, round(nw * boost))
    patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA
                       if scale < 1 else cv2.INTER_LINEAR)
    return fit_canvas(patch, size)


def _sample_fliprot():
    """Sample one flip/rot90 transform: (hflip, vflip, k_quarter_turns)."""
    return random.random() < 0.5, random.random() < 0.5, random.randint(0, 3)


def _apply_fliprot_image(image, hflip, vflip, k):
    """Apply a flip/rot90 transform to an image [H, W, C]."""
    if hflip:
        image = image[:, ::-1]
    if vflip:
        image = image[::-1]
    if k:
        image = np.rot90(image, k)
    return np.ascontiguousarray(image)


def _apply_fliprot_masks(masks, hflip, vflip, k):
    """Apply the same flip/rot90 transform to masks [G, H, W]."""
    if hflip:
        masks = masks[:, :, ::-1]
    if vflip:
        masks = masks[:, ::-1]
    if k:
        masks = np.rot90(masks, k, axes=(1, 2))
    return np.ascontiguousarray(masks)


def _realism_degrade(img_uint8, rng):
    """Mild, randomized degradations that mimic a real PDF-export rasterization.

    Measured: real excerpts carry MORE fine high-frequency detail / softer edges
    than synth (edge-density 0.080 vs 0.063), i.e. synth renders too clean. A
    model trained on crisp synthetic edges overfits them and stumbles on the
    blurrier, noisier, JPEG-compressed real plans. Applying these degradations to
    the synth scene + reference at train time closes that low-level appearance
    gap without touching real-eval (which is already 'degraded'). Kept gentle so
    the hatch texture the matching head relies on survives.
    """
    img = img_uint8
    # brightness / contrast jitter
    if rng.random() < 0.7:
        a = rng.uniform(0.85, 1.15)
        b = rng.uniform(-12, 12)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    # rasterization softness
    if rng.random() < 0.5:
        sigma = rng.uniform(0.4, 1.2)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    # sensor / scan noise
    if rng.random() < 0.5:
        std = rng.uniform(2.0, 8.0)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # JPEG compression artifacts
    if rng.random() < 0.7:
        q = int(rng.uniform(40, 90))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img)


def _to_gray3(img_uint8):
    """RGB uint8 [H, W, 3] -> luminance replicated across 3 channels.

    Keeping 3 channels (R=G=B) lets the ImageNet-pretrained backbone and its
    per-channel normalization apply unchanged; the only thing removed is colour.
    Architectural plans are near-monochrome line/hatch art, so colour is largely
    a domain-discriminating nuisance (synth fills vs. real scan tints / markup);
    dropping it forces both synth and real onto the same structural appearance.
    """
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _normalize_chw(img_uint8):
    img = img_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class InstanceSegDataset(Dataset):
    def __init__(self, records, indices, image_max_size=1024, ref_size=224,
                 augment=True, min_patch=128, max_patch=512, grayscale=False,
                 realism_aug=False, domain_random=False, scale_matched_ref=False,
                 repeat_reference_prob=0.0):
        self.records = records
        self.indices = list(indices)
        self.image_max_size = image_max_size
        self.ref_size = ref_size
        self.augment = augment
        self.min_patch = min_patch
        self.max_patch = max_patch
        self.grayscale = grayscale
        self.realism_aug = realism_aug
        self.domain_random = domain_random
        self.scale_matched_ref = scale_matched_ref
        self.repeat_reference_prob = repeat_reference_prob
        # When not augmenting (val / real eval), the reference patch is chosen
        # DETERMINISTICALLY per image so the metric measures the MODEL, not a
        # random "reference lottery" (the dominant epoch-to-epoch noise source).
        # ref_seed selects WHICH fixed reference; the eval harness sweeps it over
        # several values and averages for a robust, low-variance number.
        self.ref_seed = 0

    def __len__(self):
        return len(self.indices)

    def _load(self, i):
        rec = self.records[self.indices[i]]
        if "image_path" in rec:
            # Lazy local-data: decode from disk on demand (records hold only paths,
            # not bytes) so large datasets don't load all image bytes into RAM.
            image = np.array(Image.open(rec["image_path"]).convert("RGB"))
        else:
            image = np.array(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        # Training config stores annotations as a JSON string; the real-world-test
        # config stores them as an already-parsed list of structs.
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        return image, anns

    def __getitem__(self, i):
        image, anns = self._load(i)
        h0, w0 = image.shape[:2]

        # Render instance masks at original resolution.
        masks = [render_instance_mask(a["segmentation"], h0, w0) for a in anns]
        cats = [a.get("category_name", "pattern") for a in anns]

        # Pick a target pattern and a reference patch cropped from one of its
        # instances. Training (augment) uses the global RNG for reference variety;
        # eval (not augment) uses a per-image deterministic RNG keyed on (ref_seed,
        # i) so the SAME reference is used every epoch/run -> reproducible metric.
        ref_rng = None if self.augment else random.Random(
            (self.ref_seed * 1_000_003) ^ (i * 65537 + 12345))
        if len(anns) > 0:
            _rc = ref_rng if ref_rng is not None else random
            unique_cats = list(dict.fromkeys(cats))
            repeated_cats = [c for c in unique_cats if cats.count(c) >= 2]
            if (self.augment and repeated_cats and
                    _rc.random() < self.repeat_reference_prob):
                target_cat = _rc.choice(repeated_cats)
            else:
                target_cat = _rc.choice(unique_cats)
            cand = [j for j, c in enumerate(cats) if c == target_cat]
            ref_idx = _rc.choice(cand)
            bx, by, bw, bh = sample_reference_box(masks[ref_idx], self.min_patch,
                                                  self.max_patch, rng=ref_rng)
            ref_patch = image[by:by + bh, bx:bx + bw].copy()
            ref_match = np.array([1.0 if c == target_cat else 0.0 for c in cats], np.float32)
            # WHERE the user drew is real product input and was being discarded:
            # only the crop survived. The rectangle is sampled inside instance
            # `ref_idx`, and the target union always contains that instance, so
            # these pixels are guaranteed positives -- an anchor the model can
            # grow from instead of having to locate the pattern from scratch.
            ref_box_mask = np.zeros((h0, w0), np.uint8)
            ref_box_mask[by:by + bh, bx:bx + bw] = 1
        else:
            target_cat = None
            ref_patch = image[:min(h0, 224), :min(w0, 224)].copy()
            ref_match = np.zeros((0,), np.float32)
            ref_box_mask = np.zeros((h0, w0), np.uint8)

        # Aspect-preserving resize of image + masks.
        scale = self.image_max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
        image_r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if masks:
            masks_arr = np.stack([cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
                                  for m in masks], 0)
        else:
            masks_arr = np.zeros((0, nh, nw), np.uint8)
        # Same interpolation and geometry as the instance masks, so the anchor
        # stays pixel-aligned with the targets it is supposed to sit inside.
        ref_box_r = cv2.resize(ref_box_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # Reference patch resize.
        #
        # DO NOT "FIX" THIS TO MATCH THE IMAGE SCALE. The plan is resized by
        # image_max_size/max(h,w) (a shrink, typically ~0.34x) while the reference
        # is resized to a flat ref_size square (usually a magnification), so the
        # same hatch appears a MEDIAN 5.6x larger in the reference than in the
        # plan, and only 3% of selections land within 1.5x. That looks like an
        # obvious scale bug. It is not: `--scale-matched-ref` implements the
        # "correction" and it is the single worst change measured (validation
        # 0.7672 -> 0.6146, 2026-08-06).
        #
        # Magnifying the crop to a full ref_size square hands the encoder a large,
        # detailed view of the hatch; matching the scales instead shrinks it to
        # ~40-150px of content adrift on white padding and throws away far more
        # than scale alignment recovers. The model learns a scale-INVARIANT
        # texture embedding, not literal template matching, so reference detail
        # beats geometric correspondence. (Same reason the reference is collapsed
        # to a global-average vector in ref_unet.py: spatial layout is exactly
        # what differs between two instances of one hatch. Dense correlation over
        # a reference token grid, `--corr-grid`, is also a loss: -0.037 at g=4,
        # -0.049 at g=8.)
        if self.scale_matched_ref:
            ref_r = scale_matched_reference(ref_patch, scale, self.ref_size)
        else:
            ref_r = cv2.resize(ref_patch, (self.ref_size, self.ref_size),
                               interpolation=cv2.INTER_LINEAR)

        if self.augment:
            # One flip/rot90 transform shared by image, masks, AND the reference
            # patch. The reference is cropped from the un-augmented image, so
            # applying the same transform keeps it orientation-aligned with the
            # instances it must match — which mirrors real plans, where the same
            # pattern always appears at one consistent orientation. (Independent
            # reference rotation would train for an invariance that doesn't exist
            # and would discard orientation, a real discriminator between patterns.)
            hflip, vflip, k = _sample_fliprot()
            image_r = _apply_fliprot_image(image_r, hflip, vflip, k)
            masks_arr = _apply_fliprot_masks(masks_arr, hflip, vflip, k)
            ref_r = _apply_fliprot_image(ref_r, hflip, vflip, k)
            # The anchor is a spatial map, so it takes the SAME transform as the
            # image and masks. Leaving it out would point the model at a mirrored
            # location and teach it the anchor is noise.
            ref_box_r = _apply_fliprot_masks(ref_box_r[None], hflip, vflip, k)[0]

            if self.realism_aug:
                # Train-time only: push synth toward real PDF-export appearance.
                # Independent draws so scene and reference aren't identically degraded.
                image_r = _realism_degrade(image_r, random)
                ref_r = _realism_degrade(ref_r, random)

            if self.domain_random:
                # Domain randomization (sim2real): random DOWNSCALE so the model sees
                # synth regions across the scale range real exhibits (real instances
                # are larger / wider-spread than synth's fixed generator scale), plus
                # broad photometric jitter. Forces scale/appearance invariance so the
                # mask head generalizes to real instead of overfitting synth's fixed
                # look — the one lever that can raise real_iou WITHOUT lowering synth.
                s = random.uniform(0.4, 1.0)
                if s < 0.99:
                    new_w, new_h = max(1, int(nw * s)), max(1, int(nh * s))
                    image_r = cv2.resize(image_r, (new_w, new_h),
                                         interpolation=cv2.INTER_LINEAR)
                    if masks_arr.shape[0]:
                        masks_arr = np.stack(
                            [cv2.resize(m, (new_w, new_h),
                                        interpolation=cv2.INTER_NEAREST)
                             for m in masks_arr], 0)
                    else:
                        masks_arr = np.zeros((0, new_h, new_w), np.uint8)
                    ref_box_r = cv2.resize(ref_box_r, (new_w, new_h),
                                           interpolation=cv2.INTER_NEAREST)
                    nh, nw = new_h, new_w
                # (Random crop of the scene was tested here — div@ep10 +0.114, within
                # noise of DR-alone +0.101, lowered real too — so not kept.)
                # Photometric: brightness/contrast jitter. (Wider scale 0.4 and
                # added gamma were tested and overshot — real@10 0.21, div +0.148 —
                # so kept at the 0.4-1.0 + brightness/contrast sweet spot: div +0.101.)
                a = random.uniform(0.75, 1.25)
                b = random.uniform(-18, 18)
                image_r = np.clip(image_r.astype(np.float32) * a + b,
                                  0, 255).astype(np.uint8)
                # (Line-weight jitter via morphological erode/dilate was tested here
                # — div@ep10 +0.153, worse: it disrupts the hatch texture the
                # reference-matching head depends on, ref_match_auc fell to ~0.48.)

        if self.grayscale:
            # Strip colour from both the scene and the reference patch so the
            # synth/real comparison runs on identical (luminance-only) appearance.
            image_r = _to_gray3(image_r)
            ref_r = _to_gray3(ref_r)

        return {
            "image": _normalize_chw(image_r),                       # [3, nh, nw]
            "masks": torch.from_numpy(masks_arr).to(torch.uint8),   # [G, nh, nw]
            "ref_match": torch.from_numpy(ref_match),               # [G]
            "reference": _normalize_chw(ref_r),                     # [3, R, R]
            "ref_box": torch.from_numpy(ref_box_r).float()[None],   # [1, nh, nw]
        }


# --------------------------------------------------------------------------- #
# Collation (dynamic padding + pixel mask)
# --------------------------------------------------------------------------- #
def collate_fn(batch, size_divisible=32, union_only=False):
    """Pad a batch to a common size.

    `union_only` folds the reference-conditioned union that `train_refunet.py`
    needs into the worker instead of leaving it to the main process. Two wins:
    the `.any(0)` reduction over [G, H, W] runs across all dataloader workers
    rather than serially on the critical path, and only one [H, W] plane per
    sample crosses the worker boundary instead of G of them. The result is
    identical -- padding is zero, so unioning after padding and padding after
    unioning agree. Per-instance masks are still emitted when something needs
    them (`ranking_loss` does, whenever --rank-weight > 0)."""
    heights = [b["image"].shape[1] for b in batch]
    widths = [b["image"].shape[2] for b in batch]

    def _round_up(x):
        return int(np.ceil(x / size_divisible) * size_divisible)

    maxH, maxW = _round_up(max(heights)), _round_up(max(widths))
    B = len(batch)

    images = torch.zeros(B, 3, maxH, maxW)
    ref_boxes = torch.zeros(B, 1, maxH, maxW)
    unions = torch.zeros(B, 1, maxH, maxW, dtype=torch.bool) if union_only else None
    pixel_mask = torch.zeros(B, maxH, maxW, dtype=torch.bool)
    references = torch.stack([b["reference"] for b in batch], 0)

    targets = []
    for b, sample in enumerate(batch):
        _, h, w = sample["image"].shape
        images[b, :, :h, :w] = sample["image"]
        ref_boxes[b, :, :h, :w] = sample["ref_box"]
        pixel_mask[b, :h, :w] = True

        g = sample["masks"].shape[0]
        entry = {
            "labels": torch.ones(g, dtype=torch.long),        # all foreground
            "ref_match": sample["ref_match"],                 # [G]
        }
        if union_only:
            positive = sample["ref_match"] > 0.5
            if g > 0 and bool(positive.any()):
                unions[b, 0, :h, :w] = sample["masks"][positive].any(0)
        else:
            padded = torch.zeros(g, maxH, maxW, dtype=torch.uint8)
            if g > 0:
                padded[:, :h, :w] = sample["masks"]
            entry["masks"] = padded                           # [G, maxH, maxW]
        targets.append(entry)

    out = {"images": images, "pixel_mask": pixel_mask, "ref_boxes": ref_boxes,
           "references": references, "targets": targets}
    if union_only:
        out["union"] = unions
    return out


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def load_local_records(root):
    """Load records from a local generate_synthetic_v5.py output folder
    (`<root>/images/*.png` + `<root>/annotations/*.json`). Returns a list of
    dicts shaped like the parquet rows: {"image": <png bytes>, "annotations":
    <list of dicts>}, which `InstanceSegDataset._load` already handles. Lets us
    iterate on the generator and train on a small batch without building a
    parquet each round."""
    import glob
    import os

    recs = []
    ann_files = sorted(glob.glob(os.path.join(root, "annotations", "*.json")))
    for jf in ann_files:
        with open(jf) as f:
            ann = json.load(f)
        img_path = os.path.join(root, "images", ann["image"]["file_name"])
        # Store the PATH, not the bytes: lazy decode in _load keeps RAM flat for
        # large local datasets (loading all bytes OOMs at ~thousands of images).
        recs.append({"image_path": img_path, "annotations": ann["annotations"]})
    return recs


def load_parquet_records(repo_id="abshetty/floz-synth-v5", cache_dir=None,
                         config=None, split="train"):
    """Load a parquet split (image bytes + annotation strings).

    The synthetic training data lives in the default config (`split="train"`,
    20k rows). The hand-annotated real-world evaluation set is a separate config
    reached with `config="real-world-test", split="test"` (28 rows,
    schema-identical so it flows through the same dataset pipeline).
    """
    from datasets import load_dataset
    ds = load_dataset(repo_id, config, split=split, cache_dir=cache_dir)
    return ds


def build_datasets(records, image_max_size=1024, ref_size=224, train_split=0.9,
                   seed=42, grayscale=False, realism_aug=False,
                   domain_random=False, repeat_reference_prob=0.0,
                   scale_matched_ref=False):
    n = len(records)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(n * train_split)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_ds = InstanceSegDataset(records, train_idx, image_max_size, ref_size,
                                  augment=True, grayscale=grayscale,
                                  realism_aug=realism_aug,
                                  domain_random=domain_random,
                                  scale_matched_ref=scale_matched_ref,
                                  repeat_reference_prob=repeat_reference_prob)
    # Val stays clean (augment=False) so synth-val measures the data, not the aug.
    val_ds = InstanceSegDataset(records, val_idx, image_max_size, ref_size,
                                augment=False, grayscale=grayscale,
                                scale_matched_ref=scale_matched_ref)
    return train_ds, val_ds
