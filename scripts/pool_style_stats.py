#!/usr/bin/env python3
"""Compare a generated pool against the real evaluation set on measurable style.

Two rounds of generation were aimed by eye and both missed the target in ways
nobody could name until afterwards: v1 produced whole sheets, v2/v3 produce
crisp black-on-white CAD hatch. The evaluation set is mostly colourised,
rendered elevations at 3:1 and wider. This script puts numbers on that gap so
the next round can be aimed at the distribution rather than at an adjective.

Per image it reports:

  aspect         width / height. The eval set runs to 5:1; gpt-image-2 refuses
                 anything past 3:1, so a generated pool physically cannot cover
                 the tail without post-hoc cropping.
  saturation     mean HSV saturation over non-white pixels. Separates colourised
                 elevations from monochrome CAD.
  ink            fraction of pixels darker than 0.75 of the page white. Drawing
                 density: dense plans sit high, faint elevations near zero.
  contrast       median darkness of those ink pixels. Faint scans read low, and
                 the model's worst images are the faint ones.
  regularity     how pure the repeating fills are, as the ratio of the dominant
                 FFT peak to the median spectrum over tiles that hold a periodic
                 fill. A ruled hatch or a run of siding lines concentrates its
                 energy in one frequency and reads in the tens of thousands; a
                 fill whose lines wander and change spacing smears that peak and
                 reads in the low thousands. This is the measurement of "the AI
                 cannot keep a pattern consistent", and it is scale-controlled:
                 every image is capped to the same long side first.

    python3 scripts/pool_style_stats.py --images DIR [DIR ...] --labels NAME ...

Nothing here is a quality gate on its own. It measures distance to the real
pool, and the real pool is the target because it is what the metric is scored on.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def tile_regularity(tile):
    """Dominant-peak-to-median ratio of the tile's power spectrum.

    A field of parallel lines is a single spatial frequency, so its spectrum is
    one sharp pair of peaks. Anything that makes the lines wander -- variable
    spacing, curvature, a generator that draws hatch by eye -- spreads that
    energy out. The DC neighbourhood is excluded so page tone does not count.
    """
    tile = tile - tile.mean()
    window = np.outer(np.hanning(tile.shape[0]), np.hanning(tile.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(tile * window))) ** 2
    n = power.shape[0]
    centre = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    radius = np.hypot(yy - centre, xx - centre)
    band = (radius > 3) & (radius < n / 2)
    if not band.any():
        return None
    return float(power[band].max() / (np.median(power[band]) + 1e-12))


def image_stats(path, tile=128, max_side=1280):
    im = Image.open(path).convert("RGB")
    width, height = im.size
    if max(im.size) > max_side:                     # keep hatch resolvable, bound cost
        im = im.copy()
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    page = np.percentile(gray, 95)                  # the paper, not the ink
    ink_mask = gray < 0.75 * max(page, 1e-6)
    ink = float(ink_mask.mean())
    contrast = float(1.0 - np.median(gray[ink_mask])) if ink_mask.any() else 0.0
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    coloured = gray < 0.98 * max(page, 1e-6)
    saturation = float(sat[coloured].mean()) if coloured.any() else 0.0

    peaks = []
    h, w = gray.shape
    for y in range(0, h - tile + 1, tile // 2):     # half-tile stride, fills are small
        for x in range(0, w - tile + 1, tile // 2):
            block = gray[y:y + tile, x:x + tile]
            block_ink = (block < 0.75 * max(page, 1e-6)).mean()
            if not 0.05 < block_ink < 0.75:         # blank, or solid poche/text
                continue
            peak = tile_regularity(block)
            if peak is not None and peak >= 30:     # below this there is no fill
                peaks.append(peak)
    return {"file": path.name, "w": width, "h": height,
            "aspect": round(width / height, 2),
            "mpix": round(width * height / 1e6, 2),
            "saturation": round(saturation, 4), "ink": round(ink, 4),
            "contrast": round(contrast, 3),
            "regularity": round(float(np.median(peaks))) if peaks else None,
            "fill_tiles": len(peaks)}


def summarise(label, rows):
    def col(key):
        return [r[key] for r in rows if r.get(key) is not None]
    out = {"pool": label, "n": len(rows)}
    for key in ("aspect", "mpix", "saturation", "ink", "contrast", "regularity"):
        values = col(key)
        out[key] = round(float(np.median(values)), 4) if values else None
    out["aspect_over_3"] = sum(r["aspect"] > 3 for r in rows)
    out["colourised"] = sum(r["saturation"] > 0.05 for r in rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="one directory per pool")
    ap.add_argument("--labels", nargs="*", help="names for those pools")
    ap.add_argument("--json-out")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--max-side", type=int, default=1280,
                    help="cap every image here first; 1280 is the training size")
    args = ap.parse_args()
    labels = args.labels or [Path(d).name for d in args.images]

    report = {}
    for label, directory in zip(labels, args.images):
        paths = sorted(p for p in Path(directory).iterdir()
                       if p.suffix.lower() in SUFFIXES)
        rows = [image_stats(p, tile=args.tile, max_side=args.max_side)
                for p in paths]
        report[label] = {"summary": summarise(label, rows), "images": rows}
        s = report[label]["summary"]
        print(f"{label:<12} n={s['n']:<4} aspect={s['aspect']:<6} "
              f"(>3:1 in {s['aspect_over_3']})  mpix={s['mpix']:<6} "
              f"sat={s['saturation']:<7} (coloured {s['colourised']})  "
              f"ink={s['ink']:<7} contrast={s['contrast']:<6} "
              f"regularity={s['regularity']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
