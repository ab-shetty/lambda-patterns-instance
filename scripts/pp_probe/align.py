"""How far are label edges / prediction edges from drawn ink? Per question: distance (native px) from each edge pixel to the
nearest ink pixel. Also: of the boundary-band error pixels (pred xor tgt within 4 px of the target edge), how many lie on ink."""
import sys, json
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load, ink_map, edge
def sheet(job):
    idx, img, anyann, qs = load(job)
    ink = ink_map(img)
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
    out = []
    for ref, cat, tgt, p in qs:
        h0, w0 = tgt.shape
        pred = cv2.resize(p, (w0, h0), interpolation=cv2.INTER_LINEAR) > 0.35 * 255
        et, ep = edge(tgt), edge(pred)
        dt = cv2.distanceTransform((~tgt).astype(np.uint8), cv2.DIST_L2, 5) + cv2.distanceTransform(tgt.astype(np.uint8), cv2.DIST_L2, 5)
        band = dt <= 4
        err = (pred ^ tgt) & band
        out.append({"i": idx, "lab_d": np.percentile(dist[et], [50, 90]).tolist(), "lab_on": float((dist[et] <= 2).mean()),
                    "pred_d": np.percentile(dist[ep], [50, 90]).tolist() if ep.any() else [0, 0], "pred_on": float((dist[ep] <= 2).mean()) if ep.any() else 0,
                    "err_px": int(err.sum()), "err_on_ink": float(ink[err].mean()) if err.any() else 0,
                    "fp_band_on_ink": float(ink[err & pred].mean()) if (err & pred).any() else 0,
                    "fn_band_on_ink": float(ink[err & tgt].mean()) if (err & tgt).any() else 0,
                    "tgt_ink_share": float(ink[tgt].mean()), "ink_inside_tgt_edge_band": float(ink[et].mean())})
    return out
if __name__ == "__main__":
    for split in sys.argv[1:]:
        with Pool(3) as pool: rows = [r for rs in pool.map(sheet, jobs(split)) for r in rs]
        m = lambda k, j=None: np.median([r[k][j] if j is not None else r[k] for r in rows])
        print(f"{split}: n={len(rows)}  label edge -> ink: median {m('lab_d',0):.1f} px, p90 {m('lab_d',1):.1f}; on ink (<=2px) {m('lab_on'):.2f}"
              f" | prediction edge -> ink: median {m('pred_d',0):.1f}, p90 {m('pred_d',1):.1f}; on ink {m('pred_on'):.2f}"
              f" | band errors on ink: all {m('err_on_ink'):.2f}, FP {m('fp_band_on_ink'):.2f}, FN {m('fn_band_on_ink'):.2f}; ink share inside target {m('tgt_ink_share'):.2f}")
