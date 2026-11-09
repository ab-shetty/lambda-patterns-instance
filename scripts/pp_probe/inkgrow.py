"""Grow the product mask into the edge strokes it touches (2026-11-09). edgeink.py: labels take 56-61% of the stroke pixels on
their own edge, the prediction 48-52%, and 60% of the error within 2 px of the edge is the prediction falling short. So add
drawn-ink pixels within r px of the mask (never paper or fill), then re-polygonise. Rows: product (stack ms+cc) vs + ink grow
r = 1 / 2 / 3, IoU with the label-fix ignore pixels out.   Usage: python3 scripts/pp_probe/inkgrow.py SPLIT [MAPS_DIR]"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common
from stack import to_mask
from polygonize import probs_to_polygons, rasterize


def ink_grow(m, ink, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return m | (cv2.dilate(m.astype(np.uint8), k).astype(bool) & ink)


if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/gem2s_x")
    common.MAPS = maps
    R = {k: [] for k in ("product", "ink1", "ink2", "ink3")}
    for job in common.jobs(split):
        idx, img, anyann, qs = common.load(job)
        h, w = img.shape[:2]
        ink = common.ink_map(img)
        for ref, cat, tgt, p4, box in qs:
            p2 = np.load(maps / f"{common.map_split(split)}_2048" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
            P = 0.7 * cv2.resize(p4, (w, h)).astype(np.float32) / 255 + 0.3 * cv2.resize(p2, (w, h)).astype(np.float32) / 255
            m = to_mask(P, p4.shape[1] / w, img, box)
            key = (idx, ref)
            R["product"].append(common.masked_iou(m, tgt, key))
            for r in (1, 2, 3):
                g = rasterize(probs_to_polygons(np.where(ink_grow(m, ink, r), 255, 0).astype(np.uint8), (h, w)), (h, w)).astype(bool)
                R[f"ink{r}"].append(common.masked_iou(g, tgt, key))
    b = np.array(R["product"])
    print(split, len(b), "q | " + "  ".join(f"{k} {np.mean(v):.4f}" + ("" if k == "product" else
          f" ({np.mean(v) - b.mean():+.4f}, up {int((np.array(v) > b + 1e-3).sum())} / down {int((np.array(v) < b - 1e-3).sum())})")
          for k, v in R.items()))
