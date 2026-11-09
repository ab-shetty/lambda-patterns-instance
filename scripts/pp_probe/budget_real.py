"""Real-sheet error budget (2026-11-09; target: mIoU >= 0.95 on val and HF14 fix v2). Per question, on the product output
(stack.py ms+cc mask -> polygonize.py polygons), every error pixel goes in one bucket, and each bucket's mIoU cost is the gain if
only its pixels were fixed:
  edge        error within 4 px (native) of the true boundary (jitter; the edge-weight test did not move it)
  spill       false positive outside the band, connected to a correct region, on paper / unlabelled area
  other_fam   false positive outside the band on ANOTHER labelled family (look-alike / merge: the contrastive-loss case)
  island      false positive outside the band, not connected to any correct region, on unlabelled area
  miss_piece  false negative: a target piece (8-connected, >= 0.2% of the target) recalled < 25% (a whole region missed)
  miss_part   other false negatives outside the band (a correct region cut short / holes in it)
Usage: python3 scripts/pp_probe/budget_real.py SPLIT [MAPS_DIR]  (MAPS_DIR holds {split}_4096 and _2048 probs; default gem2s_x)
Rows -> scripts/pp_probe/rows_budget_real_{split}.json; prints the per-bucket mIoU cost and the 12 costliest questions."""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common
from stack import to_mask
from polygonize import T

BUCKETS = ("edge", "spill", "other_fam", "island", "miss_piece", "miss_part")


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def budget(m, tgt, anyann):
    t8 = tgt.astype(np.uint8)
    edge = cv2.morphologyEx(t8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    near = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5) <= 4
    fp, fn = m & ~tgt, ~m & tgt
    b = {k: np.zeros_like(m) for k in BUCKETS}
    b["edge"] = (fp | fn) & near
    far_fp = fp & ~near
    b["other_fam"] = far_fp & anyann
    n, cc = cv2.connectedComponents((far_fp & ~anyann).astype(np.uint8), connectivity=8)
    # a spill crosses the 4 px edge band before it gets here: connected = within the band + 2 px of a correct region, through
    # the predicted mask
    touch = cv2.dilate((m & tgt).astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool) & m
    if n > 1:
        hit = np.zeros(n, bool); hit[np.unique(cc[touch & (cc > 0)])] = True; hit[0] = False
        b["spill"] = hit[cc]
        b["island"] = (cc > 0) & ~hit[cc]
    n, cc, st, _ = cv2.connectedComponentsWithStats(t8, connectivity=8)
    whole = np.zeros(n, bool)
    if n > 1:
        rec = np.bincount(cc[m & tgt], minlength=n) / np.maximum(st[:, 4], 1)
        whole = (rec < 0.25) & (st[:, 4] >= 0.002 * tgt.sum()); whole[0] = False
    far_fn = fn & ~near
    b["miss_piece"] = far_fn & whole[cc]
    b["miss_part"] = far_fn & ~whole[cc]
    base = iou(m, tgt)
    gain = {}
    for k, px in b.items():
        fixed = (m & ~(px & fp)) | (px & fn)
        gain[k] = iou(fixed, tgt) - base
    return base, gain


if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/gem2s_x")
    common.MAPS = maps
    rows = []
    for job in common.jobs(split):
        idx, img, anyann, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            sp = common.map_split(split)
            p2 = np.load(maps / f"{sp}_2048" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
            P = 0.7 * cv2.resize(p4, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 + \
                0.3 * cv2.resize(p2, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            m = to_mask(P, p4.shape[1] / w, img, box)
            if common.IGNORE.get((idx, ref)) is not None:       # label fixes v3: either-way pixels count for neither side
                m = m & ~common.IGNORE[(idx, ref)]
            ign = common.IGNORE.get((idx, ref))
            base, gain = budget(m, tgt, anyann & ~tgt & (True if ign is None else ~ign))
            rows.append({"sheet": idx, "q": ref, "cat": cat, "iou": base, **gain})
    json.dump(rows, open(f"scripts/pp_probe/rows_budget_real_{split}.json", "w"))
    mi = np.mean([r["iou"] for r in rows])
    print(f"{split} {maps.name}: {len(rows)} q, product mIoU {mi:.4f}, gap to 0.95 {0.95 - mi:+.4f}")
    print("  mIoU gain if only this bucket were fixed: " + "  ".join(f"{k} +{np.mean([r[k] for r in rows]):.4f}" for k in BUCKETS))
    for r in sorted(rows, key=lambda r: r["iou"])[:12]:
        top = max(BUCKETS, key=lambda k: r[k])
        print(f"  sheet {r['sheet']:3d} q {r['q']:2d} iou {r['iou']:.3f}  worst: {top} +{r[top]:.3f}  "
              + " ".join(f"{k[:5]} {r[k]:.3f}" for k in BUCKETS))
