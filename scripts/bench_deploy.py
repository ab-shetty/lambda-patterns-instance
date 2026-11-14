#!/usr/bin/env python3
"""Deployment latency of the product, one command for a candidate card (2026-11-14; scripts/bench_latency.py is the GH200
research benchmark of post-processing variants): run it on the candidate card (A4000 / A4500 / RTX 4000 / RTX 2000 class).

Times one query end to end the way the product runs it -- the model at each scale (backbone + decoder; the backbone result is
reused for later questions on the same sheet), then scripts/product.py's post-processing -- and reports peak GPU memory, which
decides whether 5120 px fits a 16 GB card. First question on a sheet and later questions are timed separately (the backbone
and the per-sheet post-processing caches are reused for later ones).

    python3 scripts/bench_deploy.py --checkpoint ck_fgavg78.pth                       # an HF14 sheet, 3 scales
    python3 scripts/bench_deploy.py --checkpoint ck.pth --image plan.png --box 800,1200,150,150 --scales 4096,3072
Needs: torch with CUDA, opencv-python, shapely, scipy; the repo's refmask2former/ and scripts/ on PYTHONPATH (PYTHONPATH=.:scripts).
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_refunet_selection import load_refunet, _normalize_on  # noqa: E402
from refmask2former.dataset import _normalize_chw  # noqa: E402
import product  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def plane(box, h0, w0, nh, nw, device):
    x, y, w, h = box
    rows = np.zeros((h0, 1), np.uint8); rows[y:y + h] = 1
    cols = np.zeros((1, w0), np.uint8); cols[0, x:x + w] = 1
    r = torch.from_numpy(cv2.resize(rows, (1, nh), interpolation=cv2.INTER_NEAREST)[:, 0]).to(device).float()
    c = torch.from_numpy(cv2.resize(cols, (nw, 1), interpolation=cv2.INTER_NEAREST)[0]).to(device).float()
    return (r[:, None] * c[None, :])[None, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", help="a plan image (default: HF real-world-test sheet --sheet, needs the HF dataset)")
    ap.add_argument("--box", help="x,y,w,h reference box in native px (default: the sheet's first eval box)")
    ap.add_argument("--sheet", type=int, default=12)
    ap.add_argument("--scales", default=",".join(map(str, product.SCALES)))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", help="write the numbers as JSON")
    a = ap.parse_args()
    scales = [int(s) for s in a.scales.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.image:
        img = cv2.imread(a.image, cv2.IMREAD_COLOR)
        box = tuple(int(v) for v in a.box.split(",")) if a.box else (img.shape[1] // 2, img.shape[0] // 2, 150, 150)
    else:
        from refmask2former.dataset import load_parquet_records
        recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
        img = cv2.imdecode(np.frombuffer(recs[a.sheet]["image"], np.uint8), cv2.IMREAD_COLOR)
        box = tuple(json.load(open("eval_boxes/hand_v1.json"))[str(a.sheet)]["0"]) if a.box is None else \
            tuple(int(v) for v in a.box.split(","))
    h0, w0 = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model, _ = load_refunet(a.checkpoint, dev)
    roi = bool(getattr(model, "roi_ref", False))
    x, y, bw, bh = box
    ref = _normalize_chw(cv2.resize(rgb[y:y + bh, x:x + bw], (224, 224), interpolation=cv2.INTER_LINEAR))[None].to(dev)

    def run(feats_cache):
        t = {}
        maps = {}
        for z in scales:
            sync(); t0 = time.perf_counter()
            s = z / max(h0, w0); nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                im = _normalize_on(cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR), dev)[None]
                if z not in feats_cache:
                    feats_cache[z] = model.features(im)
                logit = model.forward_cached(feats_cache[z], im.shape[-2:], ref, plane(box, h0, w0, nh, nw, dev) if roi else None,
                                             image=im)
                maps[z] = (logit.sigmoid().float()[0, 0] * 255).round().to(torch.uint8).cpu().numpy()
            sync(); t[f"model_{z}"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        polys = product.product_polygons(maps, img, box)
        t["post"] = time.perf_counter() - t0
        t["total"] = sum(t.values())
        return t, polys

    run({})                                   # warm-up (cuDNN autotune, allocator)
    product._SHEET.clear()
    import polygonize
    polygonize._CELLS.clear()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    firsts, laters = [], []
    for _ in range(a.repeats):
        product._SHEET.clear(); polygonize._CELLS.clear()
        cache = {}
        t1, polys = run(cache)                # first question on a new sheet
        t2, _ = run(cache)                    # a later question on the same sheet
        firsts.append(t1); laters.append(t2)
    med = lambda L, k: float(np.median([d[k] for d in L]))
    keys = list(firsts[0])
    res = {"device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "cpu_threads": os.cpu_count(), "blur_threads": product.THREADS, "sheet_px": [w0, h0], "scales": scales,
           "first_question_s": {k: round(med(firsts, k), 3) for k in keys},
           "later_question_s": {k: round(med(laters, k), 3) for k in keys},
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == "cuda" else None,
           "polygons": len(polys)}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
