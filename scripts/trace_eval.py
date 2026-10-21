#!/usr/bin/env python3
"""Score trace twins (facsimile_trace.py, run WITHOUT labels) against the sheets' hand labels (2026-10-21).

Per sheet: every true family is matched to the traced family that overlaps it most (one traced family may serve one
true family only; greedy by overlap). Reports per-family IoU (unmatched = 0), the share of labelled pixels no traced
family claims ("unclaimed": original texture left unlabelled in the twin) and traced pixels outside every true label
("spill").

    python3 scripts/trace_eval.py --trace data/synthetic/trace_r4 --pool data/roboflow/floz-gen-gemini-r4-clean
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask  # noqa: E402


def families(anns, h, w):
    out = {}
    for a in anns:
        m = render_instance_mask(a["segmentation"], h, w).astype(bool)
        out[a["category_name"]] = out.get(a["category_name"], np.zeros((h, w), bool)) | m
    return out


def score(true, traced):
    pairs = sorted(((int((t & r).sum()), tn, rn) for tn, t in true.items() for rn, r in traced.items()), reverse=True)
    used_t, used_r, match = set(), set(), {}
    for ov, tn, rn in pairs:
        if ov and tn not in used_t and rn not in used_r:
            match[tn] = rn; used_t.add(tn); used_r.add(rn)
    ious = {}
    for tn, t in true.items():
        r = traced.get(match.get(tn), np.zeros_like(t))
        ious[tn] = float((t & r).sum() / max(1, (t | r).sum()))
    allt = np.any(list(true.values()), axis=0)
    allr = np.any(list(traced.values()), axis=0) if traced else np.zeros_like(allt)
    return ious, float((allt & ~allr).sum() / max(1, allt.sum())), float((allr & ~allt).sum() / max(1, allr.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    truth = {os.path.basename(f).split("_png")[0]: f for f in glob.glob(f"{a.pool}/annotations/*.json")}
    rows = []
    for f in sorted(glob.glob(f"{a.trace}/annotations/trace_*.json")):
        stem = os.path.basename(f)[len("trace_"):-len(".json")]
        if stem not in truth:
            continue
        T = json.load(open(truth[stem])); R = json.load(open(f))
        h, w = T["image"]["height"], T["image"]["width"]
        ious, unclaimed, spill = score(families(T["annotations"], h, w), families(R["annotations"], h, w))
        rows.append({"name": stem, "fam_iou": ious, "miou": float(np.mean(list(ious.values()))) if ious else 0.0,
                     "unclaimed": unclaimed, "spill": spill})
    for r in rows:
        print(f"{r['name']}: mIoU {r['miou']:.3f}  unclaimed {r['unclaimed']:.2f}  spill {r['spill']:.2f}  "
              + " ".join(f"{k}={v:.2f}" for k, v in r["fam_iou"].items()))
    if rows:
        print(f"== {len(rows)} sheets: mean family IoU {np.mean([r['miou'] for r in rows]):.3f}, "
              f"unclaimed {np.mean([r['unclaimed'] for r in rows]):.3f}, spill {np.mean([r['spill'] for r in rows]):.3f}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
