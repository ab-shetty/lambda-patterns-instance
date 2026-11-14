"""Would a person accept this answer? (2026-11-14; the user: "the actual target is getting what a human would consider right").
A label-based proxy for looking at the product polygons on screen, per question:
  tol_iou     IoU with disagreement within TOL native px of the label's edge ignored (nobody sees a 1-2 px shift)
  miss / extra  visible error patches: pieces of (label - prediction) / (prediction - label) outside that band, opened by a
              3 px disc (slivers along an edge are not patches), each at least VIS screen px across when the sheet is
              shown fit-to-screen (SCREEN px on its long side) -- i.e. area >= (VIS x long side / SCREEN)^2 native px
  jogs        outline vertices whose two edges are both shorter than VIS screen px: stair-steps and wiggles that read as a
              ragged edge rather than as the drawing's corners
A question is "accepted" when it has no miss / extra patch and no more than JOGS jogs.
Prints per split: product mIoU, tol mIoU, % accepted, and the questions with patches (worst first) -> rows JSON.
Usage: python3 scripts/pp_probe/human.py SPLIT [MAPS]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from polygonize import rasterize

TOL, SCREEN, VIS, JOGS = 2, 2000, 4, 2
common.WITH_BOX[0] = True


def judge(polys, tgt, ign, hw):
    h, w = hw
    pred = rasterize(polys, hw)
    if ign is not None:
        pred = pred & ~ign
    u8 = tgt.astype(np.uint8)
    band = (cv2.dilate(u8, np.ones((2 * TOL + 1,) * 2, np.uint8)) != cv2.erode(u8, np.ones((2 * TOL + 1,) * 2, np.uint8)))
    keep = ~band if ign is None else ~band & ~ign
    tol = float((pred & tgt & keep).sum() / max(1, ((pred | tgt) & keep).sum()))
    px = max(h, w) / SCREEN
    amin = (VIS * px) ** 2
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out = {}
    for name, e in (("miss", tgt & ~pred & keep), ("extra", pred & ~tgt & keep)):
        e = cv2.morphologyEx(e.astype(np.uint8), cv2.MORPH_OPEN, disc)
        n, _, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
        a = st[1:, 4] if n > 1 else np.zeros(0)
        out[name] = sorted((a[a >= amin] / max(1, tgt.sum())).round(4).tolist(), reverse=True)
    jogs = 0
    for p in polys:
        for r in [p["outer"]] + p["holes"]:
            r = np.asarray(r, float)
            if len(r) < 4:
                continue
            L = np.linalg.norm(np.roll(r, -1, 0) - r, axis=1)       # edge i: vertex i -> i+1
            jogs += int(((L < VIS * px) & (np.roll(L, 1) < VIS * px)).sum())
    return dict(iou=common.masked_iou(pred, tgt, None), tol_iou=tol, jogs=jogs, **out)


if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/gem2s_x")
    common.MAPS = maps
    rows = []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        for ref, cat, tgt, p4, box in qs:
            mp = {z: (p4 if z == 4096 else np.load(maps / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
                  for z in product.SCALES}
            polys = product.product_polygons(mp, img, box)
            r = judge(polys, tgt, common.IGNORE.get((idx, ref)), img.shape[:2])
            r.update(sheet=idx, q=ref, accepted=not r["miss"] and not r["extra"] and r["jogs"] <= JOGS)
            rows.append(r)
    out = Path(f"scripts/pp_probe/rows_human_{split}_{maps.name}.json")
    json.dump(rows, open(out, "w"))
    A = lambda k: np.mean([r[k] for r in rows])
    bad = [r for r in rows if r["miss"] or r["extra"]]
    print(f"{split} {maps.name}: {len(rows)} q  mIoU {A('iou'):.4f}  tol{TOL} mIoU {A('tol_iou'):.4f}  accepted "
          f"{100 * A('accepted'):.0f}%  | with a visible patch {len(bad)}, ragged (> {JOGS} jogs) "
          f"{sum(r['jogs'] > JOGS for r in rows)}")
    for r in sorted(bad, key=lambda r: -(sum(r["miss"]) + sum(r["extra"])))[:12]:
        print(f"   {r['sheet']:3d} q{r['q']}  IoU {r['iou']:.3f}  miss {r['miss'][:3]}  extra {r['extra'][:3]}  jogs {r['jogs']}")
