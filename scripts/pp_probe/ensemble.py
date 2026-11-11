"""Map-level model ensemble on the product output (2026-11-11). gem2s-x, gk-x and gke-x passed the gate side by side (within
noise of each other); averaging their maps (each the production blend 0.7 x 4096 + 0.3 x 2048) costs as much compute as extra
scales. Each variant through the eps-6 low-vertex pipeline, with and without the 1 px offset.
Usage: python3 scripts/pp_probe/ensemble.py SPLIT [SHARD NSHARDS];  python3 scripts/pp_probe/ensemble.py --summary SPLIT"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2

D = Path("data/floz-diag")
VARS = {  # name: [(maps dir, {size: weight}), ...] averaged equally
    "gem2s_x":              [("gem2s_x", {4096: .7, 2048: .3})],
    "gem2s_x+gk_x":         [("gem2s_x", {4096: .7, 2048: .3}), ("gk_x", {4096: .7, 2048: .3})],
    "gem2s_x+gke_x":        [("gem2s_x", {4096: .7, 2048: .3}), ("gke_x", {4096: .7, 2048: .3})],
    "all three":            [(m, {4096: .7, 2048: .3}) for m in ("gem2s_x", "gk_x", "gke_x")],
    "gem2s 3-scale":        [("gem2s_x", {5120: 1 / 3, 4096: 1 / 3, 3072: 1 / 3})],
    "gem2s 3-scale+gk_x":   [("gem2s_x", {5120: 1 / 3, 4096: 1 / 3, 3072: 1 / 3}), ("gk_x", {4096: .7, 2048: .3})],
}


def summary(split):
    R = [r for f in sorted(glob.glob(f"scripts/pp_probe/rows_ensemble_{split}*.json")) for r in json.load(open(f))]
    L = [r for r in R if r["label_v"] > 0]
    print(f"{split} {len(R)} q   eps6 (/ +off1), change vs gem2s_x, vertices / label median")
    for k in VARS:
        for o in ("", " off1"):
            d = np.array([r[k + o] - r["gem2s_x" + o] for r in R])
            print(f"  {k + o:24s} {np.mean([r[k + o] for r in R]):.4f}  {d.mean():+.4f} ({(d > 1e-3).sum()} up / "
                  f"{(d < -1e-3).sum()} down)  {np.median([r['v_' + k] / r['label_v'] for r in L]):.2f}x")


if __name__ == "__main__" and sys.argv[1] == "--summary":
    summary(sys.argv[2]); sys.exit()

import common
from stack import colour_cut, iou, CUR, nv
from curve import label_vertices
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, rasterize
common.WITH_BOX[0] = True

if __name__ == "__main__":
    split = sys.argv[1]
    shard, nsh = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    common.MAPS = D / "gem2s_x"
    ms = common.map_split(split)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rows = []
    for job in list(common.jobs(split))[shard::nsh]:
        idx, img, anyann, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            CUR[0] = (idx, ref)
            s = p4.shape[1] / w
            cache = {}

            def get(m, sz):
                if (m, sz) not in cache:
                    a = np.load(D / m / f"{ms}_{sz}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
                    cache[(m, sz)] = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                return cache[(m, sz)]
            r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat)}
            for name, parts in VARS.items():
                P = sum(sum(wt * get(m, sz) for sz, wt in wts.items()) for m, wts in parts) / len(parts)
                mk = colour_cut(img, fill_gaps(_fill_holes(P > 0.35, 4096 / s / s), P), box)
                pl = mask_to_polygons(mk, 0.005, 6, 6.0)
                base = rasterize(pl, (h, w))
                r[name], r["v_" + name] = iou(base, tgt), nv(pl)
                r[name + " off1"] = iou(cv2.dilate(base.astype(np.uint8), k3).astype(bool), tgt)
            rows.append(r)
    suf = f"_{shard}" if nsh > 1 else ""
    json.dump(rows, open(f"scripts/pp_probe/rows_ensemble_{split}{suf}.json", "w"))
    print("done", split, shard, len(rows))
