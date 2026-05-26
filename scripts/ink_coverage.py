#!/usr/bin/env python3
"""Measure per-instance interior ink coverage for a generated dataset."""

import argparse
import glob
import json

import numpy as np
from PIL import Image, ImageDraw


def render_instance_mask(segmentation, h, w):
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    outer = list(zip(segmentation[0][0::2], segmentation[0][1::2]))
    draw.polygon(outer, fill=1, outline=1)
    for hole in segmentation[1:]:
        pts = list(zip(hole[0::2], hole[1::2]))
        draw.polygon(pts, fill=0, outline=0)
    return np.array(mask, dtype=np.uint8)


def summarize(values):
    arr = np.array(values, dtype=np.float32)
    return {
        'count': int(arr.size),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)),
        'p90': float(np.percentile(arr, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='dataset root with images/ and annotations/')
    ap.add_argument('--limit', type=int, default=200, help='number of images to scan')
    ap.add_argument('--threshold', type=int, default=200,
                    help='gray threshold below which a pixel counts as ink')
    ap.add_argument('--by-role', action='store_true',
                    help='also print summaries split by annotation role')
    args = ap.parse_args()

    files = sorted(glob.glob(f'{args.root}/annotations/*.json'))[:args.limit]
    values = []
    by_role = {}

    for jf in files:
        with open(jf) as f:
            ann = json.load(f)
        gray = np.array(
            Image.open(f"{args.root}/images/{ann['image']['file_name']}").convert('L')
        )
        for obj in ann['annotations']:
            mask = render_instance_mask(obj['segmentation'], *gray.shape)
            inside = gray[mask > 0]
            if inside.size == 0:
                continue
            ink = float((inside < args.threshold).mean())
            values.append(ink)
            if args.by_role:
                by_role.setdefault(obj.get('role', 'unknown'), []).append(ink)

    if not values:
        raise SystemExit('no instances found')

    total = summarize(values)
    print(
        f"overall: images={len(files)} instances={total['count']} "
        f"mean={total['mean']:.4f} median={total['median']:.4f} "
        f"p10={total['p10']:.4f} p90={total['p90']:.4f}"
    )
    if args.by_role:
        for role in sorted(by_role):
            stats = summarize(by_role[role])
            print(
                f"{role}: instances={stats['count']} "
                f"mean={stats['mean']:.4f} median={stats['median']:.4f} "
                f"p10={stats['p10']:.4f} p90={stats['p90']:.4f}"
            )


if __name__ == '__main__':
    main()
