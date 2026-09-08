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

Measured on 321 held-out instances (see scripts/regularize_polygon.py): median 7
vertices at median IoU 0.831 against human polygons, from a hand-drawn box. The
mask is not hole-aware -- windows punched out of a wall stay filled and are cut
afterwards as separate `remove` polygons, which is how the Roboflow pipeline
already handles them.
"""
import argparse, json, os, sys
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.regularize_polygon import regularize


def load_sam3(device):
    from transformers import Sam3TrackerProcessor, Sam3TrackerModel
    tok = os.environ.get("HF_TOKEN")
    proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3", token=tok)
    model = Sam3TrackerModel.from_pretrained("facebook/sam3", token=tok).to(device).eval()
    return proc, model


def mask_for_box(proc, model, image, box_xyxy, device):
    """Best-scoring SAM3 mask for one box prompt. Returns HxW bool."""
    inputs = proc(images=image, input_boxes=[[list(map(float, box_xyxy))]],
                  return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.post_process_masks(out.pred_masks.cpu(),
                                    inputs["original_sizes"].cpu())[0][0].numpy()
    masks = masks if masks.ndim == 3 else masks[None]
    scores = out.iou_scores.detach().cpu().numpy().reshape(-1)
    return masks[int(np.argmax(scores[:len(masks)]))].astype(bool)


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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load_sam3(device)

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
            m = mask_for_box(proc, model, image, box, device)
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
