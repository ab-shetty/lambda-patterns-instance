#!/usr/bin/env python3
"""Counterfactual pools of the worst real sheets: the same sheets, labels and reference boxes, with ONE change to the
image (or to the box) per variant. Scoring each variant with the same model says which change recovers the regions the
model misses / merges -- a causal test of "what makes real sheets hard", where comparing statistics failed (2026-10-14).

  python3 scripts/probes/make_counterfactual_pool.py --out data/eval_pools/cf_v1
  for V in orig gray autocontrast flatbg soften sharpen; do
    PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint CKPT --local-pool data/eval_pools/cf_v1/$V \
      --boxes data/eval_pools/cf_v1/boxes_orig.json --image-max-size 4096 --mask-thresh 0.35 \
      --metrics-out data/evaluations/cf_v1/TAG_$V.json
  done
  for B in boxlarge box2x; do   # box variants on the unchanged images
    ... --local-pool data/eval_pools/cf_v1/orig --boxes data/eval_pools/cf_v1/boxes_$B.json \
      --metrics-out data/evaluations/cf_v1/TAG_$B.json
  done
  python3 scripts/probes/counterfactual_summary.py data/evaluations/cf_v1/TAG

Sheets: the worst by summed loss of single18k-blend50 @4096 (HF14 12 14 18 27 2, val 17 13 19 9 10, Gemini30 26 19 28 1
16 7 17). Boxes: every question's box as that model's metrics recorded it (hand boxes where they exist), written
explicitly, because automatic boxes are keyed on the sheet index, which changes in a new pool. HF14 18 carries label
fix v2 (right roof its own family). Image variants keep the size, so boxes stay valid:
  gray          luminance only (3 equal channels)
  autocontrast  per-channel 1-99 percentile stretch
  flatbg        divide by a large blur (removes paper tone, vignettes, photographed gradients), rescaled to white paper
  soften        downscale 0.5x (area) and back up (linear): softer lines, as synthetic sheets at their scale
  sharpen       unsharp mask (sigma 1.5, amount 1)
Box variants (orig images): boxlarge = a box on the family's LARGEST piece (the training sampler, seeded); box2x = the
box doubled about its centre, clipped to the sheet.
"""
import argparse
import io
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from refmask2former.dataset import render_instance_mask, sample_reference_box  # noqa: E402

SHEETS = [("hf14", i) for i in (12, 14, 18, 27, 2)] + [("val", i) for i in (17, 13, 19, 9, 10)] + \
         [("gemini30", i) for i in (26, 19, 28, 1, 16, 7, 17)]
METRICS = "eval_baselines/single18k-blend50_{}_4096.json"
GEM = "data/eval_pools/gemini30_s20261006"


def variants(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(img.reshape(-1, 3), 1, axis=0), np.percentile(img.reshape(-1, 3), 99, axis=0)
    ac = np.clip((img.astype(np.float32) - lo) * 255 / np.maximum(hi - lo, 1), 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(img.astype(np.float32), (0, 0), max(img.shape[:2]) / 30)
    fb = np.clip(img.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    so = cv2.resize(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    sh = cv2.addWeighted(img, 2.0, cv2.GaussianBlur(img, (0, 0), 1.5), -1.0, 0)
    return {"orig": img, "gray": cv2.merge([g, g, g]), "autocontrast": ac, "flatbg": fb, "soften": so, "sharpen": sh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval_pools/cf_v1")
    args = ap.parse_args()
    from refmask2former import load_parquet_records          # the evaluator's loader (cached under ./data)
    real = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
    gem_files = sorted(os.listdir(f"{GEM}/annotations"))
    fixes = json.load(open("eval_labels/hf14_fixes_v2.json"))
    sel = {sp: {} for sp in ("hf14", "val", "gemini30")}
    for sp in sel:
        for s in json.load(open(METRICS.format(sp)))["selections"]:
            sel[sp].setdefault(s["image_index"], {})[s["reference_instance"]] = s["reference_box_native"]
    boxes = {b: {"_meta": {"pool": None, "what": f"counterfactual boxes ({b})"}} for b in ("orig", "boxlarge", "box2x")}
    index = []
    for k, (sp, i) in enumerate(SHEETS):
        if sp == "gemini30":
            a = json.load(open(f"{GEM}/annotations/{gem_files[i]}"))
            img = cv2.imread(f"{GEM}/images/{a['image']['file_name']}")
            anns = a["annotations"]
        else:
            r = real[i]
            img = cv2.imdecode(np.frombuffer(r["image"], np.uint8), cv2.IMREAD_COLOR)
            anns = r["annotations"]
            anns = json.loads(anns) if isinstance(anns, str) else [dict(x) for x in anns]
            if sp == "hf14":
                for fx in fixes.get(str(i), []):
                    if "reassign" in fx:
                        anns[fx["reassign"]] = dict(anns[fx["reassign"]], category_name=fx["category"])
        H, W = img.shape[:2]
        name = f"{k:02d}_{sp}_{i:02d}"
        for v, im in variants(img).items():
            os.makedirs(f"{args.out}/{v}/images", exist_ok=True)
            os.makedirs(f"{args.out}/{v}/annotations", exist_ok=True)
            cv2.imwrite(f"{args.out}/{v}/images/{name}.png", im)
            json.dump({"image": {"file_name": f"{name}.png", "height": H, "width": W}, "annotations": anns,
                       "source": {"split": sp, "index": i}}, open(f"{args.out}/{v}/annotations/{name}.json", "w"))
        masks = [render_instance_mask(x["segmentation"], H, W).astype(np.uint8) for x in anns]
        cats = [x.get("category_name") for x in anns]
        for b in boxes:
            boxes[b][str(k)] = {}
        for q in range(len(anns)):
            box = sel[sp].get(i, {}).get(q)
            if box is None:
                for b in boxes:
                    boxes[b][str(k)][str(q)] = None
                continue
            x, y, bw, bh = [int(round(v)) for v in box]
            boxes["orig"][str(k)][str(q)] = [x, y, bw, bh]
            cx, cy = x + bw / 2, y + bh / 2
            x2, y2 = max(0, int(cx - bw)), max(0, int(cy - bh))
            boxes["box2x"][str(k)][str(q)] = [x2, y2, min(W - x2, 2 * bw), min(H - y2, 2 * bh)]
            fam = [j for j, c in enumerate(cats) if c == cats[q]]
            big = max(fam, key=lambda j: masks[j].sum())
            boxes["boxlarge"][str(k)][str(q)] = [int(v) for v in sample_reference_box(
                masks[big], 128, 512, rng=random.Random(k * 7919 + q))]
        index.append({"pool_index": k, "split": sp, "index": i, "questions": len(sel[sp].get(i, {}))})
    for b, d in boxes.items():
        d["_meta"]["pool"] = f"{args.out}/orig" if b != "orig" else None
        json.dump(d, open(f"{args.out}/boxes_{b}.json", "w"))
    json.dump(index, open(f"{args.out}/index.json", "w"), indent=1)
    print(f"{args.out}: {len(SHEETS)} sheets x 6 image variants, {sum(e['questions'] for e in index)} questions")


if __name__ == "__main__":
    main()
