import sys, json, glob, random
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from common import ink_map
from refmask2former.dataset import render_instance_mask
P = "data/synthetic/single18kq_s600000"
def sheet(f):
    d = json.load(open(f)); img = cv2.imread(f"{P}/images/{d['image']['file_name']}"); h, w = img.shape[:2]
    ink = ink_map(img); fams = {}
    for a in d["annotations"]:
        fams.setdefault(a["category_name"], np.zeros((h, w), bool))
        fams[a["category_name"]] |= render_instance_mask(a["segmentation"], h, w).astype(bool)
    out = []
    for tgt in fams.values():
        dd = cv2.distanceTransform((~tgt).astype(np.uint8), cv2.DIST_L2, 5) + cv2.distanceTransform(tgt.astype(np.uint8), cv2.DIST_L2, 5)
        b = ink & (dd <= 3)
        if b.sum() > 100: out.append((f"{d['mode']}_{d['appearance']}", float(tgt[b].mean())))
    return out
if __name__ == "__main__":
    fs = sorted(glob.glob(f"{P}/annotations/*.json")); random.Random(0).shuffle(fs)
    with Pool(32) as pool: r = [x for xs in pool.map(sheet, fs[:200]) for x in xs]
    v = np.array([x[1] for x in r])
    print(f"synthetic all ({len(v)} family targets): median {np.median(v):.2f} (p25 {np.percentile(v,25):.2f}, p75 {np.percentile(v,75):.2f})")
    for k in sorted({x[0] for x in r}):
        u = np.array([x[1] for x in r if x[0] == k]); print(f"  {k:24} n={len(u):3}  median {np.median(u):.2f} (p25 {np.percentile(u,25):.2f}, p75 {np.percentile(u,75):.2f})")
