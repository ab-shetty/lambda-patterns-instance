#!/usr/bin/env python3
"""Score an evaluate_refunet_selection.py --local-pool run by labelled families per sheet.

    python3 scripts/family_breakdown.py --pool DIR --metrics M.json [--label swin_t_seen]

Prints one row per bucket (1, 2, 3, 4+ families): questions, mean IoU, share below 0.5,
share over-selected (> 1.3x the target) and under-selected (< 0.77x). --indices of the
evaluator index the pool's sorted annotation files, as here.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def breakdown(pool, metrics):
    files = sorted(Path(pool, "annotations").glob("*.json"))
    fams = [min(4, len({a["family"] for a in json.loads(f.read_text())["annotations"]})) for f in files]
    rows = json.load(open(metrics))["selections"]
    g = defaultdict(list)
    for r in rows:
        g[fams[r["image_index"]]].append(r)
    out = {"all": rows, **{k: g[k] for k in (1, 2, 3, 4)}}
    res = {}
    for k, R in out.items():
        if not R:
            continue
        ratio = [r["prediction_pixels"] / max(r["target_pixels"], 1) for r in R]
        res[k] = {"n": len(R), "iou": float(np.mean([r["iou"] for r in R])),
                  "below_05": float(np.mean([r["iou"] < 0.5 for r in R])),
                  "over": float(np.mean([x > 1.3 for x in ratio])),
                  "under": float(np.mean([x < 0.77 for x in ratio]))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    for k, v in breakdown(a.pool, a.metrics).items():
        name = "all" if k == "all" else f"{k}{'+' if k == 4 else ''} fam"
        print(f"{a.label:24s} {name:7s} n={v['n']:4d}  IoU {v['iou']:.3f}  <0.5 {100 * v['below_05']:4.1f}%  "
              f"over {100 * v['over']:4.1f}%  under {100 * v['under']:4.1f}%")


if __name__ == "__main__":
    main()
