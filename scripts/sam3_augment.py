#!/usr/bin/env python3
"""Geometric and photometric augmentation for the box-prompted region model.

The labelling-assist model sees 165 training sheets, so the decoder can memorise
them long before it learns the boundary cue. This module perturbs the *sheet*,
not just the prompt box -- `train_sam3_boxseg.py` already jittered boxes, which
varies the prompt but shows the encoder the same 165 pictures every epoch.

Everything here transforms the OUTER POLYGON alongside the pixels and rebuilds
the box from the warped polygon, so the prompt stays consistent with the target.
Holes are not carried: the training target is the filled outer ring by design
(see `build_sam3_finetune_data.py`).

**The full dihedral group (hflip, vflip, rot90 x4) is enabled** in `default`.
The standing argument against vflip and rot90 is that gravity is a real cue on
these sheets -- siding runs level, rakes rise to a ridge, and an elevation upside
down is not a drawing the labeller will ever prompt on -- so they may cost
accuracy on elevations even while they multiply the 165-sheet pool by 8. That is
a hypothesis, not a verdict: the `nodihedral` preset holds hflip only, so the
pair measures it directly. Note the task here is *box -> region boundary*, which
is far less orientation-dependent than the product task, so the argument is
weaker on this track than it would be on RefUNet.

What is deliberately NOT here, and why:

- **Large free rotations.** Sheets are drawn on axis; a scan or a phone photo is
  off by a couple of degrees, not by thirty. (Multiples of 90 are exact and
  lossless, which is why they are treated separately from `rot_deg`.)
- **Hue shifts.** Review colour is meaningful in this pipeline (it marks whole
  pattern regions), so recolouring the ink is not a free perturbation.

The zoom-crop is the one that matters most. SAM 3's vision encoder is fixed at
1008 square, so a 3168px Gemini sheet is squashed by 3x before the decoder ever
sees it, and that is the documented accuracy ceiling of this track. Cropping to
a window around the prompt raises the effective resolution of the region under
the box, so the decoder trains on boundaries it can actually resolve.
"""
import numpy as np
import cv2


def _xy(poly):
    """Flat COCO polygon -> (N,2) float array."""
    return np.asarray(poly, np.float32).reshape(-1, 2)


def _flat(a):
    return [round(float(v), 2) for v in np.asarray(a, np.float32).reshape(-1)]


def _bbox(a, W, H):
    """(N,2) polygon -> integer [x0,y0,x1,y1] clipped to the image."""
    x0, y0 = a.min(0)
    x1, y1 = a.max(0)
    return [int(max(0, np.floor(x0))), int(max(0, np.floor(y0))),
            int(min(W - 1, np.ceil(x1))), int(min(H - 1, np.ceil(y1)))]


def _keep(a, W, H, min_area, min_inside=0.6):
    """Drop a polygon that fell (mostly) outside the frame or collapsed."""
    if len(a) < 3:
        return False
    inside = ((a[:, 0] >= 0) & (a[:, 0] < W) & (a[:, 1] >= 0) & (a[:, 1] < H)).mean()
    if inside < min_inside:
        return False
    c = np.clip(a, [0, 0], [W - 1, H - 1])
    return abs(cv2.contourArea(c.astype(np.float32))) >= min_area


DEFAULT = dict(
    hflip=0.5,          # mirrored elevations are legal drawings
    vflip=0.5,          # dihedral: see the note above, measured by `nodihedral`
    rot90=0.75,         # probability of a quarter turn; k drawn from {1,2,3}
    rot_deg=4.0,        # scan/photo skew, not a real rotation
    scale=0.15,         # +/- 15% about the sheet centre
    crop=0.5,           # probability of the zoom-crop (the resolution lever)
    crop_zoom=(1.3, 3.5),   # window size as a multiple of the anchor box
    photo=0.8,          # probability of any photometric change
    ink=0.25,           # line-weight change (scan/plot quality)
    blur=0.2,
    noise=0.15,
)


# Named arms. `nocrop` vs `default` is the one that isolates the resolution
# lever; `geom` and `photo` split the rest so a win can be attributed.
PRESETS = {
    "none": None,
    "light": dict(DEFAULT, rot_deg=2.0, scale=0.08, crop=0.0,
                  photo=0.5, ink=0.15, blur=0.1, noise=0.1),
    "default": DEFAULT,
    # hflip only -- pairs with `default` to price vflip + rot90
    "nodihedral": dict(DEFAULT, vflip=0.0, rot90=0.0),
    "dihedralonly": dict(DEFAULT, rot_deg=0.0, scale=0.0, crop=0.0,
                         photo=0.0, ink=0.0, blur=0.0, noise=0.0),
    "nocrop": dict(DEFAULT, crop=0.0),
    "geom": dict(DEFAULT, photo=0.0, ink=0.0, blur=0.0, noise=0.0),
    "photo": dict(DEFAULT, hflip=0.0, vflip=0.0, rot90=0.0,
                  rot_deg=0.0, scale=0.0, crop=0.0),
    "croponly": dict(DEFAULT, hflip=0.0, vflip=0.0, rot90=0.0, rot_deg=0.0,
                     scale=0.0, photo=0.0, ink=0.0, blur=0.0, noise=0.0, crop=1.0),
    "strong": dict(DEFAULT, rot_deg=6.0, scale=0.20, crop=0.7,
                   crop_zoom=(1.15, 4.5), photo=0.9, ink=0.35,
                   blur=0.3, noise=0.25),
}


def augment(img, instances, rng, cfg=None, min_area=100):
    """Augment one sheet.

    img        : HxWx3 uint8 RGB
    instances  : list of dicts carrying `outer_polygon` (flat COCO list)
    returns    : (img, instances) with polygons and `bbox_xyxy` rebuilt.
                 Instances that leave the frame are dropped, so the caller must
                 handle an empty list.
    """
    c = dict(DEFAULT if cfg is None else cfg)
    H, W = img.shape[:2]
    polys = [_xy(i["outer_polygon"]) for i in instances]
    meta = [dict(i) for i in instances]

    # ---- horizontal flip -------------------------------------------------
    if rng.random() < c["hflip"]:
        img = np.ascontiguousarray(img[:, ::-1])
        for p in polys:
            # W - x, not W - 1 - x: `render_instance_mask` truncates vertices to
            # int32, so a vertex at u lands in pixel floor(u) and pixel c covers
            # [c, c+1). Under that corner-origin convention the mirror of u is
            # W - u; W - 1 - u shifts every flipped mask one pixel left
            # (measured: 0.947 vs 0.999 IoU against the flipped raster).
            p[:, 0] = W - p[:, 0]

    # ---- vertical flip ---------------------------------------------------
    if rng.random() < c.get("vflip", 0.0):
        img = np.ascontiguousarray(img[::-1])
        for p in polys:
            p[:, 1] = H - p[:, 1]

    # ---- quarter turns (exact and lossless; they swap the frame) ---------
    if c.get("rot90", 0.0) > 0 and rng.random() < c["rot90"]:
        for _ in range(int(rng.integers(1, 4))):
            img = np.ascontiguousarray(np.rot90(img))     # counter-clockwise
            for p in polys:
                x = p[:, 0].copy()
                p[:, 0] = p[:, 1]        # new_x = y
                p[:, 1] = W - x          # new_y = W - x  (corner-origin)
            H, W = W, H                  # frame transposes

    # ---- rotate + scale about the centre ---------------------------------
    if c["rot_deg"] > 0 or c["scale"] > 0:
        ang = rng.uniform(-c["rot_deg"], c["rot_deg"]) if c["rot_deg"] > 0 else 0.0
        sc = 1.0 + (rng.uniform(-c["scale"], c["scale"]) if c["scale"] > 0 else 0.0)
        if abs(ang) > 1e-3 or abs(sc - 1.0) > 1e-3:
            M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, sc)
            # paper, not mirrored ink, outside the sheet
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
            for k, p in enumerate(polys):
                polys[k] = (p @ M[:, :2].T) + M[:, 2]

    # ---- zoom-crop around one instance -----------------------------------
    # The encoder is fixed at 1008 square, so this is what buys real resolution
    # on a 3168px sheet rather than just re-showing the same squashed picture.
    if c["crop"] > 0 and rng.random() < c["crop"] and polys:
        a = polys[int(rng.integers(len(polys)))]
        x0, y0, x1, y1 = a.min(0)[0], a.min(0)[1], a.max(0)[0], a.max(0)[1]
        bw, bh = max(8.0, x1 - x0), max(8.0, y1 - y0)
        z = rng.uniform(*c["crop_zoom"])
        cw, ch = min(W, bw * z), min(H, bh * z)
        cx = (x0 + x1) / 2.0 + rng.uniform(-0.15, 0.15) * cw
        cy = (y0 + y1) / 2.0 + rng.uniform(-0.15, 0.15) * ch
        cx0 = int(np.clip(cx - cw / 2.0, 0, W - cw))
        cy0 = int(np.clip(cy - ch / 2.0, 0, H - ch))
        cx1, cy1 = int(cx0 + cw), int(cy0 + ch)
        if cx1 - cx0 >= 32 and cy1 - cy0 >= 32:
            sub = img[cy0:cy1, cx0:cx1]
            shifted = [p - [cx0, cy0] for p in polys]
            sh, sw = sub.shape[:2]
            keep = [k for k, p in enumerate(shifted) if _keep(p, sw, sh, min_area)]
            if keep:                      # a crop that loses everything is no crop
                img = np.ascontiguousarray(sub)
                polys = [shifted[k] for k in keep]
                meta = [meta[k] for k in keep]
                H, W = sh, sw

    # ---- photometric: paper and plot quality -----------------------------
    if c["photo"] > 0 and rng.random() < c["photo"]:
        f = img.astype(np.float32)
        f = f * rng.uniform(0.85, 1.15) + rng.uniform(-18, 18)      # contrast/brightness
        g = rng.uniform(0.8, 1.25)                                   # gamma
        f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, g)
        img = np.clip(f, 0, 255).astype(np.uint8)

    if c["ink"] > 0 and rng.random() < c["ink"]:
        # erode darkens/thickens ink, dilate thins it -- plotter and scan weight
        k = np.ones((3, 3), np.uint8)
        img = (cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k))

    if c["blur"] > 0 and rng.random() < c["blur"]:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.2))

    if c["noise"] > 0 and rng.random() < c["noise"]:
        n = rng.normal(0, rng.uniform(2, 7), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    # ---- rebuild polygons and boxes --------------------------------------
    out = []
    for p, m in zip(polys, meta):
        if not _keep(p, W, H, min_area):
            continue
        q = np.clip(p, [0, 0], [W, H])       # corner-origin: the frame is [0, W]
        m["outer_polygon"] = _flat(q)
        m["bbox_xyxy"] = _bbox(q, W, H)
        x0, y0, x1, y1 = m["bbox_xyxy"]
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        out.append(m)
    return img, out
