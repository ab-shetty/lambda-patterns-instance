"""Visible-error A/B of a post-processing change (human.py's patches, 2026-11-14): product vs fill_gaps pmin 0.1 (regular.py).
Per split: questions with a visible patch, total visible patch area, and the questions whose visible errors changed.
Usage: python3 scripts/pp_probe/human_ab.py SPLIT [MAPS]"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import common, product, human, regular
from polygonize import rasterize
common.WITH_BOX[0] = True

if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/gem2s_x")
    common.MAPS = maps
    S = {"base": [], "pmin.1": []}; ch = []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            mp = {z: (p4 if z == 4096 else np.load(maps / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
                  for z in product.SCALES}
            P = product.blend(mp, (h, w)); ign = common.IGNORE.get((idx, ref))
            r = {}
            for k, pm in (("base", 0.05), ("pmin.1", 0.1)):
                j = human.judge(regular.polys(regular.mask(P, img, box, pm), P, 0), tgt, ign, (h, w))
                r[k] = (sum(j["miss"]) + sum(j["extra"]), len(j["miss"]) + len(j["extra"]), j["iou"])
                S[k].append(r[k])
            if abs(r["base"][0] - r["pmin.1"][0]) > 0.001:
                ch.append((idx, ref, r["base"], r["pmin.1"]))
    for k, v in S.items():
        a = np.array(v)
        print(f"{split} {maps.name} {k:7s} mIoU {a[:, 2].mean():.4f}  questions with a visible patch {(a[:, 1] > 0).sum()}  "
              f"visible patch area {a[:, 0].mean():.4f} of label per question")
    for idx, ref, b, p in sorted(ch, key=lambda x: x[3][0] - x[2][0]):
        print(f"   {idx:3d} q{ref}: visible error {b[0]:.4f} -> {p[0]:.4f}  (IoU {b[2]:.3f} -> {p[2]:.3f})")
