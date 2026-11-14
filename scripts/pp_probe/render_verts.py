"""Product v3 polygons with their vertices, two map sets side by side (2026-11-14, fgab vertices). Per question: crop around
label + prediction; label green fill, polygons red with a dot per vertex; title = IoU and vertices.
Usage: python3 scripts/pp_probe/render_verts.py SPLIT OUT_DIR MAPS_A MAPS_B sheet:q [sheet:q ...]"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from polygonize import rasterize
common.WITH_BOX[0] = True

split, out, ma, mb = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
want = {tuple(int(x) for x in a.split(":")) for a in sys.argv[5:]}
out.mkdir(parents=True, exist_ok=True)


def maps_for(md, idx, ref):
    return {z: np.load(md / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"] for z in product.SCALES}


common.MAPS = ma
for job in common.jobs(split):
    if job[0] not in {s for s, _ in want}:
        continue
    idx, img, _, qs = common.load(job)
    h, w = img.shape[:2]
    for ref, cat, tgt, _, box in qs:
        if (idx, ref) not in want:
            continue
        panels, ms = [], []
        for md in (ma, mb):
            polys = product.product_polygons(maps_for(md, idx, ref), img, box)
            ms.append((polys, rasterize(polys, (h, w))))
        ys, xs = np.where(tgt | ms[0][1] | ms[1][1])
        pad = 40
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad), max(0, xs.min() - pad), min(w, xs.max() + pad)
        for md, (polys, m) in zip((ma, mb), ms):
            v = img.copy()
            v[tgt] = (0.6 * v[tgt] + 0.4 * np.array([60, 200, 60])).astype(np.uint8)
            lw = max(2, int(max(x1 - x0, y1 - y0) / 700))
            for p in polys:
                for r in [p["outer"]] + p["holes"]:
                    a = np.array(r, np.int32)
                    cv2.polylines(v, [a], True, (0, 0, 255), lw)
                    for x, y in a: cv2.circle(v, (int(x), int(y)), lw * 2, (255, 0, 0), -1)
            bx, by, bw, bh = box
            cv2.rectangle(v, (bx, by), (bx + bw, by + bh), (0, 220, 255), lw)
            c = v[y0:y1, x0:x1]; s = 1100 / max(c.shape[:2]); c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            nv = product.nverts(polys); iou = common.masked_iou(m, tgt, (idx, ref))
            c = cv2.copyMakeBorder(c, 50, 0, 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(c, f"{md.name}  {idx:02d} q{ref}  IoU {iou:.4f}  {nv} vertices", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            panels.append(c)
        hh = max(p.shape[0] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, hh - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in panels]
        cv2.imwrite(str(out / f"{split}_{idx:02d}_q{ref}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", out / f"{split}_{idx:02d}_q{ref}.jpg", flush=True)
