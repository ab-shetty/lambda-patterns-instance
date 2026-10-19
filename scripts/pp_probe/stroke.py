"""Do labels include the boundary stroke? Ink pixels within 3 px of the target edge (either side): share inside the target.
And for the prediction (linear >0.35)."""
import sys
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load, ink_map
def sheet(job):
    idx, img, anyann, qs = load(job); ink = ink_map(img); h0, w0 = img.shape[:2]; out = []
    for ref, cat, tgt, p in qs:
        d = cv2.distanceTransform((~tgt).astype(np.uint8), cv2.DIST_L2, 5) + cv2.distanceTransform(tgt.astype(np.uint8), cv2.DIST_L2, 5)
        b = ink & (d <= 3)
        P = cv2.resize(p, (w0, h0), interpolation=cv2.INTER_LINEAR) > 89
        if b.sum() > 100: out.append((tgt[b].mean(), P[b].mean()))
    return out
if __name__ == "__main__":
    for split in sys.argv[1:]:
        with Pool(2) as pool: r = np.array([x for xs in pool.map(sheet, jobs(split)) for x in xs])
        print(f"{split}: boundary-stroke pixels inside label: median {np.median(r[:,0]):.2f} (p25 {np.percentile(r[:,0],25):.2f}, p75 {np.percentile(r[:,0],75):.2f}); inside prediction: median {np.median(r[:,1]):.2f}")
