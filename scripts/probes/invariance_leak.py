#!/usr/bin/env python3
"""Leak table for the invariance probe: how much of the OTHER panel a model selects when the two panels
differ in one attribute (scripts/probes/make_invariance_probe.py). 1.00 = the model treats them as one
pattern; 0.00 = it separates them. Reads the evaluator's metrics JSON and --save-probs folder.

  python3 scripts/probes/make_invariance_probe.py data/eval_pools/probe_invariance_v1
  PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint CKPT \
      --local-pool data/eval_pools/probe_invariance_v1 --boxes data/eval_pools/probe_invariance_v1/boxes.json \
      --image-max-size 4096 --mask-thresh 0.35 --metrics-out M.json --save-probs P/
  python3 scripts/probes/invariance_leak.py --metrics M.json --probs P/ [--json out.json]

Baselines (eval_baselines/probe_*): longswiss-swa37-41 @4096 and revitfailswiss2k-e8 @2048.
"""
import argparse
import glob
import json
import os

import numpy as np

W, H = 2400, 1300                      # make_invariance_probe.py geometry
PANELS = ((180, 420, 1080, 1180), (1320, 420, 2220, 1180))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--pool", default="data/eval_pools/probe_invariance_v1")
    ap.add_argument("--thresh", type=float, default=0.35)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.pool}/annotations/*.json"))
    rows = {(r["image_index"], r["reference_instance"]): r for r in json.load(open(a.metrics))["selections"]}
    out = []
    print(f"{'probe':22s} {'differs in':14s} leak A->B  leak B->A")
    for i, f in enumerate(files):
        attr = json.load(open(f)).get("attr", "?")
        leak = []
        for q in (0, 1):
            p = np.load(os.path.join(a.probs, f"{i:03d}_{q:03d}.npz"))["prob"] > round(a.thresh * 255)
            h, w = p.shape
            x0, y0, x1, y1 = PANELS[1 - q]
            leak.append(float(p[int(y0 * h / H):int(y1 * h / H), int(x0 * w / W):int(x1 * w / W)].mean()))
        name = os.path.basename(f)[3:-5]
        out.append({"probe": name, "attr": attr, "leak": leak,
                    "iou": [rows[(i, q)]["iou"] for q in (0, 1) if (i, q) in rows]})
        print(f"{name:22s} {attr:14s} {leak[0]:9.2f} {leak[1]:9.2f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
