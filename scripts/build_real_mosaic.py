#!/usr/bin/env python3
"""Build train-only real-panel mosaics as local-data synthetic coverage.

This script never uses the held-out HF real indices. It composes only the
deterministic train side of real-world-test plus optional local extra real pools
into white construction-sheet pages, transforming all annotation polygons into
the mosaic coordinate system.
"""

import argparse
import glob
import io
import json
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former import load_parquet_records


def _norm_anns(anns):
    if isinstance(anns, str):
        anns = json.loads(anns)
    return json.loads(json.dumps(
        anns, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


def _poly_area(poly):
    pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _ann_area(segmentation):
    if not segmentation:
        return 0.0
    return max(0.0, _poly_area(segmentation[0]) -
               sum(_poly_area(p) for p in segmentation[1:]))


def _bbox(segmentation):
    xs = []
    ys = []
    for poly in segmentation:
        xs.extend(poly[0::2])
        ys.extend(poly[1::2])
    if not xs:
        return [0, 0, 0, 0]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _scale_anns(anns, scale):
    out = []
    for ann in anns:
        seg = [[float(c) * scale for c in poly] for poly in ann["segmentation"]]
        cur = dict(ann)
        cur["segmentation"] = seg
        cur["bbox"] = _bbox(seg)
        cur["area"] = _ann_area(seg)
        cur["num_holes"] = max(0, len(seg) - 1)
        out.append(cur)
    return out


def _transform_anns(anns, scale, dx, dy, start_id):
    out = []
    next_id = start_id
    for ann in anns:
        seg = []
        for poly in ann["segmentation"]:
            dst = []
            for i in range(0, len(poly), 2):
                dst.append(float(poly[i]) * scale + dx)
                dst.append(float(poly[i + 1]) * scale + dy)
            seg.append(dst)
        cur = dict(ann)
        cur["id"] = next_id
        cur.setdefault("category_id", 1)
        cur.setdefault("category_name", "pattern")
        cur["segmentation"] = seg
        cur["bbox"] = _bbox(seg)
        cur["area"] = _ann_area(seg)
        cur["num_holes"] = max(0, len(seg) - 1)
        out.append(cur)
        next_id += 1
    return out, next_id


def _copy_base_dirs(base_dirs, out_root):
    n = 0
    for base in base_dirs:
        for jf in sorted(glob.glob(os.path.join(base, "annotations", "*.json"))):
            with open(jf) as f:
                ann = json.load(f)
            old = ann["image"]["file_name"]
            src_img = os.path.join(base, "images", old)
            fn = f"item_{n:06d}.png"
            shutil.copy(src_img, os.path.join(out_root, "images", fn))
            ann["image"]["file_name"] = fn
            with open(os.path.join(out_root, "annotations", f"item_{n:06d}.json"), "w") as f:
                json.dump(ann, f)
            n += 1
    return n


def _load_hf_scenes(args):
    recs = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir,
                                config="real-world-test", split="test")
    perm = list(range(len(recs)))
    random.Random(args.split_seed).shuffle(perm)
    train_idx, held_idx = perm[:14], perm[14:]
    scenes = []
    for idx in train_idx:
        rec = recs[idx]
        img = Image.open(io.BytesIO(rec["image"])).convert("RGB")
        anns = _norm_anns(rec["annotations"])
        if anns:
            scenes.append({"image": img, "anns": anns, "name": f"hf{idx}"})
    print(f"HF real train={sorted(train_idx)} held-out={sorted(held_idx)} scenes={len(scenes)}")
    return scenes, held_idx


def _load_extra_scenes(extra_dirs, target_long):
    scenes = []
    for extra_dir in extra_dirs:
        before = len(scenes)
        for jf in sorted(glob.glob(os.path.join(extra_dir, "annotations", "*.json"))):
            with open(jf) as f:
                ann = json.load(f)
            img_path = os.path.join(extra_dir, "images", ann["image"]["file_name"])
            img = Image.open(img_path).convert("RGB")
            anns = _norm_anns(ann["annotations"])
            if not anns:
                continue
            w, h = img.size
            if target_long and max(w, h) != target_long:
                scale = target_long / max(w, h)
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                 Image.LANCZOS)
                anns = _scale_anns(anns, scale)
            stem = os.path.splitext(os.path.basename(jf))[0]
            scenes.append({"image": img, "anns": anns, "name": stem})
        print(f"extra real [{extra_dir}] scenes={len(scenes) - before}")
    return scenes


def _page_size(rng):
    layouts = [(2200, 1500), (1700, 2200), (1900, 1900), (2600, 1800), (1800, 2600)]
    w, h = rng.choice(layouts)
    scale = rng.uniform(0.88, 1.12)
    return int(w * scale), int(h * scale)


def _cells_for_page(rng, w, h, n_panels):
    margin = rng.randint(55, 135)
    gutter = rng.randint(35, 95)
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    if n_panels == 1:
        return [(x0, y0, x1, y1)]
    if n_panels == 2:
        if w >= h:
            mid = (x0 + x1 - gutter) // 2
            return [(x0, y0, mid, y1), (mid + gutter, y0, x1, y1)]
        mid = (y0 + y1 - gutter) // 2
        return [(x0, y0, x1, mid), (x0, mid + gutter, x1, y1)]
    if rng.random() < 0.5:
        midx = (x0 + x1 - gutter) // 2
        midy = (y0 + y1 - gutter) // 2
        return [(x0, y0, midx, midy), (midx + gutter, y0, x1, midy),
                (x0, midy + gutter, midx, y1)]
    if w >= h:
        midx = int(x0 + (x1 - x0 - gutter) * 0.58)
        midy = (y0 + y1 - gutter) // 2
        return [(x0, y0, midx, y1), (midx + gutter, y0, x1, midy),
                (midx + gutter, midy + gutter, x1, y1)]
    midy = int(y0 + (y1 - y0 - gutter) * 0.58)
    midx = (x0 + x1 - gutter) // 2
    return [(x0, y0, x1, midy), (x0, midy + gutter, midx, y1),
            (midx + gutter, midy + gutter, x1, y1)]


def _mild_panel_aug(img, rng):
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.92, 1.08))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.92, 1.10))
    if rng.random() < 0.18:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.45)))
    arr = np.asarray(img).astype(np.float32)
    if rng.random() < 0.35:
        nrng = np.random.RandomState(rng.randint(0, 2**31 - 1))
        arr += nrng.normal(0, rng.uniform(1.0, 3.5), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _draw_sheet_marks(canvas, cells, rng):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    gray = rng.randint(186, 225)
    for x0, y0, x1, y1 in cells:
        if rng.random() < 0.85:
            draw.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6],
                           outline=(gray, gray, gray), width=rng.choice([1, 1, 2]))
    if rng.random() < 0.45:
        bx1 = canvas.width - rng.randint(260, 520)
        by1 = canvas.height - rng.randint(105, 190)
        draw.rectangle([bx1, by1, canvas.width - 55, canvas.height - 55],
                       outline=(170, 170, 170), width=1)
        for k in range(rng.randint(2, 4)):
            y = by1 + (k + 1) * (canvas.height - 55 - by1) / rng.randint(4, 6)
            draw.line([bx1, y, canvas.width - 55, y], fill=(190, 190, 190), width=1)


def _make_mosaic(scenes, rng):
    w, h = _page_size(rng)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    n_panels = rng.choices([1, 2, 3], weights=[0.55, 0.30, 0.15], k=1)[0]
    cells = _cells_for_page(rng, w, h, n_panels)
    _draw_sheet_marks(canvas, cells, rng)

    anns = []
    next_id = 1
    for cell, scene in zip(cells, rng.sample(scenes, k=min(n_panels, len(scenes)))):
        x0, y0, x1, y1 = cell
        pad = rng.randint(8, 28)
        cw = max(1, x1 - x0 - 2 * pad)
        ch = max(1, y1 - y0 - 2 * pad)
        sw, sh = scene["image"].size
        scale = min(cw / sw, ch / sh) * rng.uniform(0.88, 1.0)
        nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
        panel = scene["image"].resize((nw, nh), Image.LANCZOS)
        panel = _mild_panel_aug(panel, rng)
        dx = x0 + pad + rng.randint(0, max(0, cw - nw))
        dy = y0 + pad + rng.randint(0, max(0, ch - nh))
        canvas.paste(panel, (dx, dy))
        cur, next_id = _transform_anns(scene["anns"], scale, dx, dy, next_id)
        anns.extend(cur)
    return canvas, anns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="number of mosaics to append")
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--base-dir", action="append", default=[],
                    help="existing local-data dir to copy before appending mosaics")
    ap.add_argument("--real-extra-dir", action="append", default=[])
    ap.add_argument("--real-extra-size", type=int, default=1024,
                    help="resize each extra real so long side == this; 0 keeps native")
    ap.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--split-seed", type=int, default=1234)
    args = ap.parse_args()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(os.path.join(args.out, "images"))
    os.makedirs(os.path.join(args.out, "annotations"))

    n_base = _copy_base_dirs(args.base_dir, args.out)
    hf_scenes, held_idx = _load_hf_scenes(args)
    extra = _load_extra_scenes(args.real_extra_dir, args.real_extra_size)
    scenes = hf_scenes + extra
    if not scenes:
        raise SystemExit("no train-side real scenes loaded")
    print(f"source scenes total={len(scenes)} (HF train={len(hf_scenes)}, extra={len(extra)})")

    rng = random.Random(args.seed)
    for i in range(args.n):
        img, anns = _make_mosaic(scenes, rng)
        item_id = n_base + i
        fn = f"item_{item_id:06d}.png"
        img.save(os.path.join(args.out, "images", fn))
        with open(os.path.join(args.out, "annotations", f"item_{item_id:06d}.json"), "w") as f:
            json.dump({
                "image": {"file_name": fn, "width": img.width, "height": img.height},
                "mode": "real_mosaic",
                "annotations": anns,
            }, f)
    print(f"wrote {n_base + args.n} items to {args.out} "
          f"(base={n_base}, mosaics={args.n})")
    print("HELD-OUT indices for --real-eval-indices:")
    print("  " + ",".join(str(i) for i in held_idx))


if __name__ == "__main__":
    main()
