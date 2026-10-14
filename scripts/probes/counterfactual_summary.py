#!/usr/bin/env python3
"""Summary of a counterfactual run (make_counterfactual_pool.py): per variant, mean IoU over the same questions, and the
change against `orig` split by how the model fails on the original (under-selects: pred/target < 0.77; over-selects:
> 1.3; else), plus how many questions gain / lose > 0.1.

  python3 scripts/probes/counterfactual_summary.py data/evaluations/cf_v1/TAG
"""
import glob
import json
import os
import sys

import numpy as np

pre = sys.argv[1]
runs = {os.path.basename(f)[len(os.path.basename(pre)) + 1:-5]: f for f in glob.glob(pre + "_*.json")}
load = lambda f: {(s["image_index"], s["reference_instance"]): s for s in json.load(open(f))["selections"]}
base = load(runs.pop("orig"))
kind = {k: ("under" if s["prediction_pixels"] / max(1, s["target_pixels"]) < 0.77 else
            "over" if s["prediction_pixels"] / max(1, s["target_pixels"]) > 1.3 else "ok") for k, s in base.items()}
n = {t: sum(v == t for v in kind.values()) for t in ("under", "over", "ok")}
print(f"orig: {np.mean([s['iou'] for s in base.values()]):.4f} over {len(base)} questions "
      f"(under-selecting {n['under']}, over {n['over']}, ok {n['ok']})")
for v in sorted(runs):
    d = load(runs[v]); ks = [k for k in base if k in d]
    delta = {k: d[k]["iou"] - base[k]["iou"] for k in ks}
    by = {t: np.mean([delta[k] for k in ks if kind[k] == t]) if n[t] else float("nan") for t in n}
    print(f"{v:13s} {np.mean([d[k]['iou'] for k in ks]):.4f}  ({np.mean(list(delta.values())):+.4f}) | under {by['under']:+.3f} "
          f"over {by['over']:+.3f} ok {by['ok']:+.3f} | gain>0.1: {sum(x > 0.1 for x in delta.values())}  "
          f"loss>0.1: {sum(x < -0.1 for x in delta.values())}")
