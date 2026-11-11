"""Which edges to grow (2026-11-10). lvdecomp: a uniform 1 px outward offset on the low-vertex polygons wins on the real sheets
(valfix3 +0.007, hf14fix2 +0.003) and loses on Gemini30 (-0.006, 158 / 195 down) -- the label sets put the edge on opposite
sides of the boundary stroke. Candidates on the eps-6 low-vertex output (fills, colour cut, eps 6, pieces < 0.5%):
  off1        grow 1 px everywhere
  inkA r      grow into drawn ink within r px (common.ink_map: darker than the 9 px local max by > 60)
  inkB r      same with common.ink_map_v2 (thin dark strokes only: black top-hat, sheet-adaptive threshold)
Per question also the sheet's colourfulness (median HSV saturation of non-paper pixels) so a sheet-style gate can be read off.
Usage: python3 scripts/pp_probe/edgeoff.py SPLIT MAPS [SHARD NSHARDS] -> rows_edgeoff_<tag>_<split>[_<shard>].json
       python3 scripts/pp_probe/edgeoff.py --summary TAG SPLIT"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2

VARS = ["eps6", "off1", "inkA1", "inkA2", "inkB1", "inkB2", "inkA1p"]


def summary(tag, split):
    R = [r for f in sorted(glob.glob(f"scripts/pp_probe/rows_edgeoff_{tag}_{split}*.json")) for r in json.load(open(f))]
    print(f"{split} {tag} {len(R)} q, IoU and change vs eps6 (up / down)")
    for k in VARS:
        d = np.array([r[k] - r["eps6"] for r in R])
        print(f"  {k:6s} {np.mean([r[k] for r in R]):.4f}  {d.mean():+.4f} ({(d > 1e-3).sum()} up / {(d < -1e-3).sum()} down)")
    L = [r for r in R if r.get("label_v", 0) > 0]
    if L and "v_inkA1p" in L[0]:
        for k in ("eps6", "inkA1p"):
            v = np.array([r["v_" + k] / r["label_v"] for r in L])
            print(f"  vertices / label {k}: median {np.median(v):.2f}x (p90 {np.percentile(v, 90):.1f}x)")
    sat = np.array([r["sat"] for r in R])
    print("  sheet colourfulness (median sat of non-paper px): quartiles", np.percentile(sat, [0, 25, 50, 75, 100]).round(1))
    for t in (20, 30, 40, 60):
        g = [r["off1"] if r["sat"] < t else r["eps6"] for r in R]
        print(f"  off1 only if sat < {t:2d}: {np.mean(g):.4f} ({np.mean(g) - np.mean([r['eps6'] for r in R]):+.4f}), "
              f"{sum(r['sat'] < t for r in R)} / {len(R)} q gated in")


if __name__ == "__main__" and sys.argv[1] == "--summary":
    summary(sys.argv[2], sys.argv[3]); sys.exit()

import common
from stack import colour_cut, iou, CUR, nv
from curve import label_vertices
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, rasterize
common.WITH_BOX[0] = True


def grow(m, r, where=None):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    g = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    return g if where is None else m | (g & where)


def colourfulness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    nonpaper = hsv[..., 2] < np.percentile(hsv[..., 2], 90) - 10
    s = hsv[..., 1][nonpaper]
    return float(np.median(s)) if s.size else 0.0


if __name__ == "__main__":
    split, maps = sys.argv[1], sys.argv[2]
    shard, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    common.MAPS = Path(maps)
    rows = []
    for job in list(common.jobs(split))[shard::nsh]:
        idx, img, anyann, qs = common.load(job)
        h, w = img.shape[:2]
        inkA, inkB, sat = common.ink_map(img), common.ink_map_v2(img), colourfulness(img)
        for ref, cat, tgt, p4, box in qs:
            CUR[0] = (idx, ref)
            f2 = Path(maps) / f"{common.map_split(split)}_2048" / "probs" / f"{idx:03d}_{ref:03d}.npz"
            if not f2.exists():
                continue
            s = p4.shape[1] / w
            P = (0.7 * cv2.resize(p4, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                 + 0.3 * cv2.resize(np.load(f2)["prob"], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255)
            m = colour_cut(img, fill_gaps(_fill_holes(P > 0.35, 4096 / s / s), P), box)
            base = rasterize(mask_to_polygons(m, 0.005, 6, 6.0), (h, w))
            r = {"sheet": idx, "q": ref, "sat": sat, "eps6": iou(base, tgt), "off1": iou(grow(base, 1), tgt)}
            for name, ink in (("inkA", inkA), ("inkB", inkB)):
                for rr in (1, 2):
                    r[f"{name}{rr}"] = iou(grow(base, rr, ink), tgt)
            # inkA1p: grow the mask into ink BEFORE tracing (what a product would ship: one polygon pass), eps 6
            pp = mask_to_polygons(grow(m, 1, inkA), 0.005, 6, 6.0)
            r["inkA1p"] = iou(rasterize(pp, (h, w)), tgt)
            r["v_eps6"], r["v_inkA1p"] = nv(mask_to_polygons(m, 0.005, 6, 6.0)), nv(pp)
            r["label_v"] = label_vertices(job[1], cat)
            rows.append(r)
    tag = Path(maps).name
    suf = f"_{shard}" if nsh > 1 else ""
    json.dump(rows, open(f"scripts/pp_probe/rows_edgeoff_{tag}_{split}{suf}.json", "w"))
    print("done", split, tag, shard, len(rows))
