#!/usr/bin/env python3
"""Upgrade cheap box annotations into polygon annotations with SAM 3.

Box-drawing is the fast gesture in the Roboflow UI; polygon tracing is the slow
one. So label a sheet with rough boxes, export, run this, and re-import: each box
becomes a regularized polygon on the region it encloses.

    python3 scripts/sam3_box_to_polygon.py \
        --coco  data/roboflow/<batch>/train/_annotations.coco.json \
        --img-dir data/roboflow/<batch>/train \
        --out   data/proposals/<batch>_polygons.coco.json \
        --preview data/proposals/<batch>_preview

The mask is not hole-aware -- windows punched out of a wall stay filled and are
cut afterwards as separate `remove` polygons, which is how the Roboflow pipeline
already handles them.

This runs the FINE-TUNED decoder through `RegionModel`, with the zoom-crop on by
default. Until 2026-09-09 it built its own stock `facebook/sam3` and ignored the
fine-tune entirely, so the batch path was labelling zero-shot: `>=0.8` 71.4%
where the tuned decoder plus `--crop-zoom 3` measures 91.8%. Pass
`--checkpoint ""` to get the old zero-shot behaviour back (and note that
cropping is negative there, so it also forces `--crop-zoom 0`).
"""
import argparse, json, os, sys
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.regularize_polygon import regularize
from scripts.sam3_region_model import RegionModel, DEFAULT_CHECKPOINT


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--coco", help="COCO json whose annotations carry bbox [x,y,w,h]")
    src.add_argument("--boxes", help='json: {"img.png": [[x0,y0,x1,y1], ...], ...}')
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", help="directory for overlay PNGs (optional)")
    ap.add_argument("--eps-frac", type=float, default=0.010,
                    help="Douglas-Peucker tolerance as a fraction of perimeter")
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--checkpoint", default=None,
                    help="local .pth/.safetensors or Hub repo id; default is the "
                         'published decoder. Pass "" for stock zero-shot SAM 3.')
    ap.add_argument("--crop-zoom", type=float, default=2.0,
                    help="run the encoder on a window this many times the box's "
                         "long side (0 = whole sheet). Costs one forward per box "
                         "instead of per sheet; worth it on a GPU.")
    args = ap.parse_args()

    zoom = args.crop_zoom
    if args.checkpoint == "":                 # explicit zero-shot
        ck, zoom = None, 0.0
    else:
        ck = args.checkpoint if args.checkpoint else DEFAULT_CHECKPOINT
    rm = RegionModel(checkpoint=ck, eps_frac=args.eps_frac,
                     crop_zoom=zoom if zoom else None)
    print(f"decoder: {ck or 'stock zero-shot'}   crop_zoom: {zoom or 'off'}   "
          f"device: {rm.device}")

    # ---- gather {file_name: [box_xyxy, ...]} -------------------------------
    jobs, categories, coco = {}, [], None
    if args.coco:
        coco = json.load(open(args.coco))
        by_id = {im["id"]: im for im in coco["images"]}
        categories = coco.get("categories", [])
        for a in coco["annotations"]:
            im = by_id.get(a["image_id"])
            if im is None:
                continue
            x, y, w, h = a["bbox"]
            jobs.setdefault(im["file_name"], []).append(
                ([x, y, x + w, y + h], a.get("category_id", 1)))
    else:
        for fn, boxes in json.load(open(args.boxes)).items():
            jobs[fn] = [(b, 1) for b in boxes]
        categories = [{"id": 1, "name": "pattern"}]

    out_images, out_anns, aid, skipped = [], [], 1, 0
    for iid, (fn, boxes) in enumerate(sorted(jobs.items()), start=1):
        p = os.path.join(args.img_dir, fn)
        if not os.path.exists(p):
            print(f"  missing image, skipped: {fn}"); skipped += len(boxes); continue
        image = Image.open(p).convert("RGB")
        W, H = image.size
        out_images.append({"id": iid, "file_name": fn, "width": W, "height": H})
        overlay = np.asarray(image).copy() if args.preview else None
        for box, cat in boxes:
            m = rm.masks(image, [box])[0]
            if m.sum() < args.min_area:
                skipped += 1; continue
            poly = regularize(m, eps_frac=args.eps_frac)
            if poly is None or len(poly) < 3:
                skipped += 1; continue
            poly = np.clip(poly, [0, 0], [W - 1, H - 1])
            xs, ys = poly[:, 0], poly[:, 1]
            out_anns.append({
                "id": aid, "image_id": iid, "category_id": cat,
                "segmentation": [[round(float(v), 2) for v in poly.reshape(-1)]],
                "bbox": [float(xs.min()), float(ys.min()),
                         float(xs.max() - xs.min()), float(ys.max() - ys.min())],
                "area": float(cv2.contourArea(poly.astype(np.float32))),
                "iscrowd": 0})
            aid += 1
            if overlay is not None:
                cv2.polylines(overlay, [np.round(poly).astype(np.int32)], True, (255, 0, 0), 3)
        if overlay is not None:
            os.makedirs(args.preview, exist_ok=True)
            Image.fromarray(overlay).save(os.path.join(args.preview, fn.rsplit(".", 1)[0] + ".png"))
        print(f"  {fn}: {len(boxes)} boxes -> {sum(1 for a in out_anns if a['image_id']==iid)} polygons",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"images": out_images, "annotations": out_anns,
               "categories": categories or [{"id": 1, "name": "pattern"}]},
              open(args.out, "w"), indent=1)
    v = [len(a["segmentation"][0]) // 2 for a in out_anns]
    print(f"\nwrote {len(out_anns)} polygons over {len(out_images)} images -> {args.out}")
    if v:
        print(f"vertices: median {np.median(v):.0f}  mean {np.mean(v):.1f}  max {max(v)}")
    if skipped:
        print(f"skipped {skipped} boxes (no mask / degenerate)")


if __name__ == "__main__":
    main()
