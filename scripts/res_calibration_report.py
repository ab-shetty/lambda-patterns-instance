#!/usr/bin/env python3
"""Report for run_res_calibration.sh: does the r8 - v6d delta at 1024 predict it at 2048?

Per seed and resolution, the paired delta over the fixed selections (77 val +
52 HF14). Then agreement across resolutions: seed-mean deltas side by side,
and the correlation of per-selection seed-averaged deltas (1024-trained vs
2048-trained). A high correlation with matching means says the cheap screen
ranks the same way; a low one with matching means says it agrees only on
average; mismatched means say resolution interacts with the change.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_compare import load_rows  # noqa: E402


def deltas(d, res, seed, infer, split):
    a, _ = load_rows(f"{d}/v6d_r{res}_s{seed}_i{infer}_{split}.json")
    b, _ = load_rows(f"{d}/r8_r{res}_s{seed}_i{infer}_{split}.json")
    keys = sorted(set(a) & set(b))
    return keys, np.array([b[k] - a[k] for k in keys]), np.array([a[k] for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    args = ap.parse_args()
    seeds = " ".join(args.seeds).split()
    configs = [("1024", "2048"), ("2048", "2048"), ("2048", "4096")]
    for split in ("val", "hf14"):
        print(f"\n=== {split} ===")
        per = {}
        for res, inf in configs:
            rows = []
            for s in seeds:
                try:
                    keys, dl, base = deltas(args.dir, res, s, inf, split)
                except FileNotFoundError:
                    continue
                rows.append(dl)
                print(f"train {res} infer {inf} seed {s}: v6d {base.mean():.4f}  "
                      f"r8-v6d {dl.mean():+.4f}  (better {int((dl > .01).sum())} / "
                      f"worse {int((dl < -.01).sum())} / tied {int((abs(dl) <= .01).sum())})")
            if rows:
                m = np.mean(rows, 0)
                per[(res, inf)] = m
                seed_means = [r.mean() for r in rows]
                sd = np.std(seed_means, ddof=1) if len(rows) > 1 else float("nan")
                print(f"  -> mean over {len(rows)} seeds {np.mean(seed_means):+.4f} (sd {sd:.4f})")
        ref = per.get(("1024", "2048"))
        for cfg in (("2048", "2048"), ("2048", "4096")):
            if ref is not None and cfg in per:
                r = np.corrcoef(ref, per[cfg])[0, 1]
                print(f"  per-selection corr of seed-mean deltas, 1024 vs train {cfg[0]}/infer {cfg[1]}: {r:+.3f}")


if __name__ == "__main__":
    main()
