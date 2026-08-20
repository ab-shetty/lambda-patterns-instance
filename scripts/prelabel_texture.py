#!/usr/bin/env python3
"""Draft the polygons for a generated plan, so labelling is correction not drawing.

Hand-labelling is the binding cost of a generation round -- the API is now ~$0.10
an image in batch, while 300 images is days of polygon work. But the thing that
makes these images hard for the model is exactly what makes them tractable here:
a material family is a periodic fill, and every region of one family shares a
spatial frequency and an angle. So cluster the texture and the clusters ARE the
image-local families, which is the grouping the annotation contract asks for.

    python3 scripts/prelabel_texture.py --image IMG.png --out-dir DIR [--overlay]

Writes a COCO instance-segmentation file with `pattern1..N` and a review overlay.
Feed it to Roboflow as a pre-annotation and correct it there.

Verdict as it stands (2026-08-19): NOT yet worth using. On a colourised
elevation the clusters do group the same wall material across two separate
drawings, which is the useful half; on a floor plan they track furniture, text
and stair treads as readily as the fills, and correcting that costs more than
drawing 6 polygons would. Two things would likely fix it, in this order:

1. seed it instead of clustering blind -- the human clicks one point per family
   and `scripts/hatch_matcher.py` (which beats the Gabor probe by a wide margin,
   see synth_progress.md 2026-08-07) matches that patch across the image. One
   click per family beats one polygon per occurrence, and it mirrors the
   product's own interaction;
2. propose axis-aligned rectangles rather than free contours. The median
   annotation in both labelled pools has FOUR vertices -- these are boxes, not
   traced outlines -- so the tile-grid boundaries this produces are the wrong
   shape as well as the wrong precision.

What it will never be is a labeller: it cannot cut `remove` holes, and a family
drawn at two scales splits in two.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

Image.MAX_IMAGE_PIXELS = None


def tile_features(gray, page, window, stride):
    """Per-tile texture descriptor: ink, tone, and the dominant periodic peak."""
    h, w = gray.shape
    ys = range(0, h - window + 1, stride)
    xs = range(0, w - window + 1, stride)
    win2d = np.outer(np.hanning(window), np.hanning(window))
    centre = window // 2
    yy, xx = np.mgrid[0:window, 0:window]
    radius = np.hypot(yy - centre, xx - centre)
    band = (radius > 2) & (radius < window / 2)
    feats, coords = [], []
    for y in ys:
        for x in xs:
            tile = gray[y:y + window, x:x + window]
            ink = float((tile < 0.85 * page).mean())
            power = np.abs(np.fft.fftshift(np.fft.fft2((tile - tile.mean()) * win2d))) ** 2
            peak_idx = np.argmax(np.where(band, power, 0))
            py, px = divmod(int(peak_idx), window)
            sharp = float(power[py, px] / (np.median(power[band]) + 1e-12))
            angle = np.arctan2(py - centre, px - centre)      # mod pi
            freq = float(radius[py, px] / window)
            feats.append([ink * 4.0, float(tile.mean()),
                          np.cos(2 * angle) * min(sharp / 500.0, 1.0),
                          np.sin(2 * angle) * min(sharp / 500.0, 1.0),
                          freq * 3.0, min(sharp / 2000.0, 1.5)])
            coords.append((y, x))
    return np.array(feats, dtype=np.float32), coords, len(list(ys)), len(list(xs))


def cluster_map(feats, rows, cols, k, seed=0):
    model = KMeans(n_clusters=k, n_init=8, random_state=seed)
    labels = model.fit_predict(feats)
    return labels.reshape(rows, cols), model


def polygons_from_mask(mask, min_area, epsilon_frac=0.004):
    """Outer contours only; `remove` holes stay a job for the human."""
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_frac * cv2.arcLength(contour, True),
                                  True)
        if len(approx) >= 3:
            out.append((approx.reshape(-1, 2), float(area)))
    return out


def prelabel(path, k=5, window=64, stride=16, max_side=1600, min_area_frac=0.0012):
    im = Image.open(path).convert("L")
    full_w, full_h = im.size
    scale = min(1.0, max_side / max(im.size))
    if scale < 1.0:
        im = im.resize((int(full_w * scale), int(full_h * scale)), Image.LANCZOS)
    gray = np.asarray(im, dtype=np.float32) / 255.0
    page = float(np.percentile(gray, 95))

    feats, coords, rows, cols = tile_features(gray, page, window, stride)
    labels, _ = cluster_map(feats, rows, cols, k)

    # The cluster holding the paper is whichever has the least ink; drop it.
    ink_by_cluster = {c: feats[labels.reshape(-1) == c][:, 0].mean() for c in range(k)}
    background = min(ink_by_cluster, key=ink_by_cluster.get)

    height, width = gray.shape
    min_area = min_area_frac * height * width
    annotations, families = [], 0
    label_image = np.zeros((height, width), dtype=np.int32) - 1
    for index, (y, x) in enumerate(coords):
        label_image[y:y + window, x:x + window] = labels.reshape(-1)[index]

    for cluster in range(k):
        if cluster == background:
            continue
        polys = polygons_from_mask(label_image == cluster, min_area)
        if not polys:
            continue
        families += 1
        for points, area in polys:
            points = (points / scale).astype(np.float32)     # back to full size
            xs, ys = points[:, 0], points[:, 1]
            annotations.append({
                "category_name": f"pattern{families}",
                "segmentation": [points.reshape(-1).tolist()],
                "area": area / (scale * scale),
                "bbox": [float(xs.min()), float(ys.min()),
                         float(xs.max() - xs.min()), float(ys.max() - ys.min())],
                "iscrowd": 0})
    return annotations, families, (full_w, full_h), label_image, background, scale


def write_coco(path, annotations, size, out_path):
    names = sorted({a["category_name"] for a in annotations})
    categories = [{"id": i + 1, "name": n, "supercategory": "pattern"}
                  for i, n in enumerate(names)]
    ids = {c["name"]: c["id"] for c in categories}
    coco = {"info": {"description": "texture pre-labels, for correction"},
            "images": [{"id": 1, "file_name": Path(path).name,
                        "width": size[0], "height": size[1]}],
            "categories": categories,
            "annotations": [{"id": i + 1, "image_id": 1,
                             "category_id": ids[a["category_name"]],
                             "segmentation": a["segmentation"], "area": a["area"],
                             "bbox": a["bbox"], "iscrowd": 0}
                            for i, a in enumerate(annotations)]}
    Path(out_path).write_text(json.dumps(coco))
    return out_path


def write_overlay(path, annotations, out_path, scale_to=1500):
    colours = [(220, 50, 50), (40, 140, 220), (60, 170, 60), (200, 130, 30),
               (150, 60, 200), (0, 160, 160)]
    im = cv2.imread(str(path))
    overlay = im.copy()
    names = sorted({a["category_name"] for a in annotations})
    for annotation in annotations:
        colour = colours[names.index(annotation["category_name"]) % len(colours)]
        pts = np.array(annotation["segmentation"][0], np.int32).reshape(-1, 2)
        cv2.fillPoly(overlay, [pts], colour)
        cv2.polylines(im, [pts], True, colour, 3)
    blended = cv2.addWeighted(overlay, 0.35, im, 0.65, 0)
    height, width = blended.shape[:2]
    if max(height, width) > scale_to:
        factor = scale_to / max(height, width)
        blended = cv2.resize(blended, (int(width * factor), int(height * factor)))
    cv2.imwrite(str(out_path), blended)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", default="data/image_generation/prelabels")
    ap.add_argument("--families", type=int, default=5,
                    help="clusters including the paper; 3-4 families means 4-5")
    ap.add_argument("--overlay", action="store_true", help="also write a review PNG")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    annotations, families, size, _, _, _ = prelabel(args.image, k=args.families)
    coco = write_coco(args.image, annotations, size, out_dir / f"{stem}.coco.json")
    print(f"{stem}: {families} families, {len(annotations)} regions -> {coco}")
    if args.overlay:
        print(f"overlay: {write_overlay(args.image, annotations, out_dir / f'{stem}.overlay.png')}")


if __name__ == "__main__":
    main()
