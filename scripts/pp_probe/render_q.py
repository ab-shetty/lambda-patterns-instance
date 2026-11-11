"""Render chosen questions on the product output (eps-6 low-vertex + 1 px offset), 2026-11-10: where the remaining real-sheet
gap sits. Panel per question: [image + reference box (yellow) | label (green fill) with ignore pixels (magenta) |
error view: overlap grey, excess red, missed blue], plus the same family's other-colour context.
Usage: python3 scripts/pp_probe/render_q.py SPLIT MAPS OUT_DIR sheet:q [sheet:q ...]"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common
from stack import colour_cut, iou, CUR
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, rasterize
common.WITH_BOX[0] = True

split, maps, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
want = {tuple(int(x) for x in a.split(":")) for a in sys.argv[4:]}
out.mkdir(parents=True, exist_ok=True)
common.MAPS = maps
for job in common.jobs(split):
    if job[0] not in {s for s, _ in want}:
        continue
    idx, img, anyann, qs = common.load(job)
    h, w = img.shape[:2]
    for ref, cat, tgt, p4, box in qs:
        if (idx, ref) not in want:
            continue
        CUR[0] = (idx, ref)
        s = p4.shape[1] / w
        p2 = np.load(maps / f"{common.map_split(split)}_2048" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
        P = (0.7 * cv2.resize(p4, (w, h)).astype(np.float32) + 0.3 * cv2.resize(p2, (w, h)).astype(np.float32)) / 255
        m = colour_cut(img, fill_gaps(_fill_holes(P > 0.35, 4096 / s / s), P), box)
        pred = cv2.dilate(rasterize(mask_to_polygons(m, 0.005, 6, 6.0), (h, w)).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
        v = iou(pred, tgt)
        ign = common.IGNORE.get(CUR[0])
        a = img.copy(); x, y, bw, bh = box
        cv2.rectangle(a, (x, y), (x + bw, y + bh), (0, 220, 255), max(3, w // 400))
        b = img.copy(); b[tgt] = (0.5 * b[tgt] + [0, 110, 0]).astype(np.uint8)
        if ign is not None and ign.any():
            b[ign] = (0.5 * b[ign] + [110, 0, 110]).astype(np.uint8)
        c = (img * 0.35 + 165).astype(np.uint8)
        c[pred & tgt] = (110, 110, 110); c[pred & ~tgt] = (0, 0, 230); c[~pred & tgt] = (230, 90, 0)
        if ign is not None:
            c[ign] = (200, 0, 200)
        panel = np.concatenate([a, b, c], 1)
        sc = 2400 / panel.shape[1]
        panel = cv2.resize(panel, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        fp, fn = (pred & ~tgt).sum(), (~pred & tgt).sum()
        cv2.putText(panel, f"{split} sheet {idx} q{ref}  IoU {v:.3f}  excess {fp} px  missed {fn} px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        f = out / f"{split}_{idx:03d}_q{ref}_{v:.3f}.png"
        cv2.imwrite(str(f), panel)
        print(f, f"excess {fp} missed {fn} target {tgt.sum()}")
