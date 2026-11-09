"""Where the near-edge error sits (2026-11-09): error pixels within 2 px of the true edge, split by whether they are drawn ink
(a line stroke) or paper / fill, and by side (prediction too wide = FP, too short = FN); plus how the LABEL treats strokes on
its own edge (share of edge-stroke pixels inside the label: ~0.5 = the edge runs through stroke centres) and how the
PREDICTION does. Also by sheet size: the model predicts at stride 4 of the 4096 input, so a 6000 px sheet gets ~6 px cells.
Usage: python3 scripts/pp_probe/edgeink.py SPLIT [MAPS_DIR]"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common
from stack import to_mask

split = sys.argv[1]
maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/gem2s_x")
common.MAPS = maps
tot = dict(fp_ink=0, fp_paper=0, fn_ink=0, fn_paper=0)
lab_in, pred_in, n_edge_ink = 0, 0, 0
by_size = {}
for job in common.jobs(split):
    idx, img, anyann, qs = common.load(job)
    h, w = img.shape[:2]
    ink = common.ink_map(img)
    for ref, cat, tgt, p4, box in qs:
        p2 = np.load(maps / f"{common.map_split(split)}_2048" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
        P = 0.7 * cv2.resize(p4, (w, h)).astype(np.float32) / 255 + 0.3 * cv2.resize(p2, (w, h)).astype(np.float32) / 255
        m = to_mask(P, p4.shape[1] / w, img, box)
        ign = common.IGNORE.get((idx, ref))
        if ign is not None:
            m = m & ~ign
        t8 = tgt.astype(np.uint8)
        edge = cv2.morphologyEx(t8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        near = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5) <= 2
        fp, fn = m & ~tgt & near, ~m & tgt & near
        tot["fp_ink"] += int((fp & ink).sum()); tot["fp_paper"] += int((fp & ~ink).sum())
        tot["fn_ink"] += int((fn & ink).sum()); tot["fn_paper"] += int((fn & ~ink).sum())
        e_ink = near & ink
        lab_in += int((e_ink & tgt).sum()); pred_in += int((e_ink & m).sum()); n_edge_ink += int(e_ink.sum())
        k = "large (> 4096 px)" if max(h, w) > 4096 else "small (<= 4096 px)"
        u = (m | tgt).sum()
        d = by_size.setdefault(k, [0, 0, 0])
        d[0] += 1; d[1] += float((m & tgt).sum() / max(1, u)); d[2] += float(((m ^ tgt) & near).sum() / max(1, u))
err = sum(tot.values())
print(f"{split}: error px within 2 px of the true edge: " + "  ".join(f"{k} {100 * v / err:.0f}%" for k, v in tot.items()))
print(f"  edge-stroke pixels inside: label {lab_in / max(1, n_edge_ink):.2f}, prediction {pred_in / max(1, n_edge_ink):.2f}")
for k, (n, s_iou, s_edge) in sorted(by_size.items()):
    print(f"  {k}: {n} q, IoU {s_iou / n:.4f}, IoU lost to the 2 px band {s_edge / n:.4f}")
