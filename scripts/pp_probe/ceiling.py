"""How much of the edge loss is the label's own edge noise? (2026-11-19.) No model involved: variants of the LABEL itself, each
a perfectly consistent rule a predictor could at best follow, scored against the label:
  shiftK      the label moved K px (worst of 4 directions -> best of 4 reported): sensitivity of the metric to a 1-2 px slip
  dilK/eroK   the label grown / shrunk K px everywhere (a predictor with a constant edge bias)
  stroke_in   the label plus every drawn-stroke pixel touching its edge (strokes always included)
  stroke_out  the label minus every drawn-stroke pixel along its edge (strokes always excluded)
The best of stroke_in / stroke_out is an upper bound for a predictor that puts its edge on a fixed side of the drawn line;
the label owns ~56-61% of edge-stroke pixels (2026-11-12), so neither matches it exactly. Usage: ceiling.py SPLIT"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common
common.WITH_BOX[0] = True

if __name__ == "__main__":
    split = sys.argv[1]
    common.MAPS = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/fgavg78")
    R = {k: [] for k in ("shift1", "shift2", "dil1", "ero1", "dil2", "ero2", "stroke_in", "stroke_out", "best_stroke")}
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        ink = common.ink_map(img, 40)
        for ref, cat, tgt, _, box in qs:
            key = (idx, ref); t8 = tgt.astype(np.uint8)
            iou = lambda m: common.masked_iou(m, tgt, key)
            for k in (1, 2):
                R[f"shift{k}"].append(max(iou(np.roll(tgt, s, ax)) for s in (k, -k) for ax in (0, 1)))
                ker = np.ones((2 * k + 1,) * 2, np.uint8)
                R[f"dil{k}"].append(iou(cv2.dilate(t8, ker).astype(bool)))
                R[f"ero{k}"].append(iou(cv2.erode(t8, ker).astype(bool)))
            band = cv2.dilate(t8, np.ones((7, 7), np.uint8)).astype(bool) & ~cv2.erode(t8, np.ones((7, 7), np.uint8)).astype(bool)
            # stroke pixels connected to the edge band (one stroke width ~ up to 6 px)
            si = iou(tgt | (ink & band)); so = iou(tgt & ~(ink & band))
            R["stroke_in"].append(si); R["stroke_out"].append(so); R["best_stroke"].append(max(si, so))
    print(f"{split}: {len(R['shift1'])} q -- the LABEL vs consistent rules applied to itself (mean IoU)")
    for k, v in R.items():
        print(f"  {k:12s} {np.mean(v):.4f}")
