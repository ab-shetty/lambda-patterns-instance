"""Fragmentation (2026-11-09): per question, our pieces that lie on the target (>= 50% of the piece inside it) vs the label's
polygon count for the target family, and the vertices on each kind. Usage: python3 scripts/pp_probe/frag.py SPLIT MAPS EPS [SHARD N]
  python3 scripts/pp_probe/frag.py --summary MODEL EPS"""
import sys, json, glob
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from pathlib import Path


def label_rings(rec, cat):
    anns = rec["annotations"]; anns = json.loads(anns) if isinstance(anns, str) else anns
    n = 0
    for a in anns:
        if a.get("category_name", a.get("category")) != cat:
            continue
        seg = a["segmentation"]; seg = json.loads(seg) if isinstance(seg, str) else seg
        rings = seg if isinstance(seg, list) else seg.get("polygons", [])
        n += sum(1 for r in rings if r)
    return n


if __name__ == "__main__" and sys.argv[1] == "--summary":
    model, eps = sys.argv[2], sys.argv[3]
    for split in ("gemini30", "val", "hf14"):
        rows = sum((json.load(open(f)) for f in sorted(glob.glob(f"scripts/pp_probe/rows_frag_{model}_{split}_e{eps}*.json"))), [])
        rows = [r for r in rows if r["label_v"] > 0]
        on = np.array([r["n_on"] for r in rows]); lr = np.array([r["label_rings"] for r in rows])
        small = np.array([r["n_on_small"] for r in rows])
        v_on = np.array([r["v_on"] for r in rows]); v_main = np.array([r["v_main"] for r in rows]); v_sp = np.array([r["v_spill"] for r in rows])
        v_sm = np.array([r["v_on_small"] for r in rows]); tot = v_main + v_on + v_sp
        print(f"{split} eps {eps}: label polygons median {np.median(lr):.0f} | our on-target pieces median {np.median(on):.0f} "
              f"(ratio median {np.median(on / np.maximum(lr, 1)):.2f}, p75 {np.percentile(on / np.maximum(lr, 1), 75):.2f}) | "
              f"vertices: main {100 * v_main.sum() / tot.sum():.0f}%, other on-target {100 * v_on.sum() / tot.sum():.0f}% "
              f"(of which pieces < 1% of target: {100 * v_sm.sum() / tot.sum():.0f}%), spill {100 * v_sp.sum() / tot.sum():.0f}% | "
              f"q with more on-target pieces than label polygons: {100 * np.mean(on > lr):.0f}%")
    sys.exit()

import common
from curve import polys_from, label_vertices, B
from polygonize import rasterize
common.WITH_BOX[0] = True
split, maps, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])
shard, nsh = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
common.MAPS = Path(maps)
rows = []
for k, job in enumerate(common.jobs(split)):
    if k % nsh != shard:
        continue
    idx, img, _, qs = common.load(job)
    h, w = img.shape[:2]
    for ref, cat, tgt, p4, box in qs:
        s = p4.shape[1] / w
        P = cv2.resize(p4.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_LINEAR)
        polys = polys_from(P, s, {**B, "eps": eps})
        r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat), "label_rings": label_rings(job[1], cat),
             "n_on": 0, "n_on_small": 0, "v_main": 0, "v_on": 0, "v_on_small": 0, "v_spill": 0}
        ta = max(1, int(tgt.sum()))
        pieces = []
        for p in polys:
            m = rasterize([p], (h, w)).astype(bool); a = int(m.sum())
            pieces.append((a, (m & tgt).sum() / max(1, a), len(p["outer"]) + sum(len(x) for x in p["holes"])))
        if pieces:
            j = int(np.argmax([p[0] for p in pieces]))
            for i, (a, inside, v) in enumerate(pieces):
                if inside >= 0.5:
                    r["n_on"] += 1
                if i == j:
                    r["v_main"] += v
                elif inside >= 0.5:
                    r["v_on"] += v
                    if a < 0.01 * ta:
                        r["v_on_small"] += v; r["n_on_small"] += 1
                else:
                    r["v_spill"] += v
        rows.append(r)
tag = Path(maps).name
json.dump(rows, open(f"scripts/pp_probe/rows_frag_{tag}_{split}_e{sys.argv[3]}{'_%d' % shard if nsh > 1 else ''}.json", "w"))
print("done", split, tag, shard, len(rows))
