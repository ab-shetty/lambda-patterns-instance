"""Smoke check for --foreground: python3 scripts/probes/fg_check.py POOL_FG_ON POOL_FG_OFF (same ids / seed).
Per sheet: labels unchanged, new ref_exclude rings, changed pixels outside them (should be darker = cast shadow),
fraction of the footprints over labels, footprint bottom minus the bottom of the labels in its column (0 = on grade;
large values = another elevation row or an unlabelled wall)."""
import json, sys, glob, os
import numpy as np, cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union
A, B = sys.argv[1], sys.argv[2]          # fg on, fg off
def rings(flat): return Polygon(np.array(flat).reshape(-1, 2)).buffer(0)
def lab_union(ann):
    ps = []
    for a in ann["annotations"]:
        segs = a["segmentation"]; outer = rings(segs[0])
        for h in segs[1:]: outer = outer.difference(rings(h))
        ps.append(outer)
    return unary_union(ps)
tot = []
for f in sorted(glob.glob(f"{A}/annotations/*.json")):
    n = os.path.basename(f); a1 = json.load(open(f)); a0 = json.load(open(f"{B}/annotations/{n}"))
    same_labels = a1["annotations"] == a0["annotations"]
    r0 = {json.dumps(r) for r in a0.get("ref_exclude", [])}
    new = [r for r in a1.get("ref_exclude", []) if json.dumps(r) not in r0]
    i1 = cv2.imread(f"{A}/images/{a1['image']['file_name']}"); i0 = cv2.imread(f"{B}/images/{n[:-5]}.png")
    H, W = i1.shape[:2]
    D = (np.abs(i1.astype(int) - i0.astype(int)).sum(2) > 6)
    M = np.zeros((H, W), np.uint8)
    for r in new: cv2.fillPoly(M, [np.round(np.array(r).reshape(-1, 2)).astype(np.int32)], 1)
    Md = cv2.dilate(M, np.ones((7, 7), np.uint8))
    out = D & (Md == 0)
    darker = (i1.astype(int).sum(2) < i0.astype(int).sum(2))
    shadow_frac = (out & darker).sum() / max(out.sum(), 1)
    L = lab_union(a1)
    newu = unary_union([rings(r) for r in new]) if new else Polygon()
    cov = newu.intersection(L).area / newu.area if new else float("nan")
    # bottom of each footprint vs the bottom of the labelled house directly above it
    gaps = []
    for r in new:
        p = rings(r); x0, y0, x1, y1 = p.bounds
        col = L.intersection(Polygon([(x0, 0), (x1, 0), (x1, H), (x0, H)]))
        if not col.is_empty: gaps.append(round(y1 - col.bounds[3], 1))
    tot.append((n, same_labels, len(new), D.sum(), out.sum(), shadow_frac, cov, gaps))
    print(f"{n} labels_same={same_labels} new_rx={len(new)} changed_px={D.sum()} outside_rx={out.sum()} "
          f"(darker {shadow_frac:.2f}) rx_on_labels={cov:.2f} bottom-minus-label-bottom_px={gaps}")
print("sheets with objects:", sum(t[2] > 0 for t in tot), "/", len(tot), " all labels unchanged:", all(t[1] for t in tot))
