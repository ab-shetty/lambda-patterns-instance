"""CascadePSP (Cheng et al., CVPR 2020; pip segmentation-refinement, pretrained on natural images, no fine-tuning) on the default
mask. Per question: lin_fill mask and the sheet (RGB), cropped to the mask's bounding box + 25% margin (+32 px), refined, output
> 127, holes filled, polygons. Variants: cpsp (global + local steps, L=900), cpsp_fast (global step only), cpsp_band8 (cpsp's
changes kept only within 8 native px of lin_fill's edge). Pick on Gemini30, read val / HF14 once."""
import sys, json, os
import cv2, numpy as np, torch
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load
from methods import fill_holes, polygons, iou
import segmentation_refinement as refine
T = 0.35
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
R = refine.Refiner(device="cpu")

def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]; rgb = img[:, :, ::-1].copy(); rows = []
    for ref, cat, tgt, p in qs:
        s = p.shape[1] / w0; ha = 4096 / s / s
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lf = fill_holes(P > T, ha)
        r = {"i": idx, "q": ref, "ref": iou(polygons(lf)[0], tgt)}
        ys, xs = np.nonzero(lf)
        if not len(ys):
            for k in ("cpsp", "cpsp_fast", "cpsp_band8"): r[k] = r["ref"]
            rows.append(r); continue
        my, mx = int(0.25 * (ys.ptp() + 1)) + 32, int(0.25 * (xs.ptp() + 1)) + 32
        y1, y2, x1, x2 = max(ys.min() - my, 0), min(ys.max() + my + 1, h0), max(xs.min() - mx, 0), min(xs.max() + mx + 1, w0)
        crop, mc = rgb[y1:y2, x1:x2], lf[y1:y2, x1:x2].astype(np.uint8) * 255
        din = cv2.distanceTransform(lf.astype(np.uint8), cv2.DIST_L2, 5); dout = cv2.distanceTransform((~lf).astype(np.uint8), cv2.DIST_L2, 5)
        band = (din <= 8) & (dout <= 8)
        for k, fast in (("cpsp", False), ("cpsp_fast", True)):
            o = R.refine(crop, mc, fast=fast, L=900) > 127
            m = lf.copy(); m[y1:y2, x1:x2] = o
            r[k] = iou(polygons(fill_holes(m, ha))[0], tgt)
            if k == "cpsp": r["cpsp_band8"] = iou(polygons(fill_holes(np.where(band, m, lf), ha))[0], tgt)
        rows.append(r)
    print(idx, {k: round(float(np.mean([x[k] - x["ref"] for x in rows])), 4) for k in rows[0] if k.startswith("cpsp")}, flush=True)
    return rows

if __name__ == "__main__":
    split = sys.argv[1]; only = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    rows = [r for j in jobs(split) if only is None or j[0] in only for r in sheet(j)]
    json.dump(rows, open(f"data/diag/rows_cpsp_{split}.json", "w"))
    b = np.array([r["ref"] for r in rows])
    for k in ("cpsp", "cpsp_fast", "cpsp_band8"):
        a = np.array([r[k] for r in rows]); d = a - b
        print(f"{split} {k:10s} {a.mean():.4f} ({d.mean():+.4f} vs default; up {(d > .005).sum()} / down {(d < -.005).sum()}; worst {d.min():+.3f})", flush=True)
