"""Does the B&W copy (--degrade gray) merge differently coloured regions? Per question: the target category vs the other
labelled categories on the sheet -- colour distance (CIE Lab dE, median colour of each region) vs grey-level distance
(the same BGR2GRAY the stress test uses) -- next to the question's IoU drop none -> gray (stress_best JSONs)."""
import io, json, sys
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, "scripts")
from evaluate_refunet_selection import load_parquet_records, render_instance_mask

recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
fixes = json.load(open("eval_labels/real_fixes_v4.json"))
rows = []
for split in ("valfix4", "hf14fix4"):
    a = json.load(open(f"data/evaluations/stress_best/{split}_none.json"))["selections"]
    b = json.load(open(f"data/evaluations/stress_best/{split}_gray.json"))["selections"]
    drop = {(s["image_index"], s["reference_instance"]): (s["iou"], t["iou"]) for s, t in zip(a, b)}
    cats_q = {(s["image_index"], s["reference_instance"]): s["category"] for s in a}
    for i in sorted({k[0] for k in drop}):
        r = recs[i]; img = np.asarray(Image.open(io.BytesIO(r["image"])).convert("RGB"))
        anns = r["annotations"]; anns = json.loads(anns) if isinstance(anns, str) else anns
        h, w = img.shape[:2]
        fx = fixes.get(str(i), []) if isinstance(fixes, dict) else []
        cats = [x.get("category_name", "pattern") for x in anns]
        for f in fx:
            if "reassign" in f: cats[f["reassign"]] = f["category"]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
        lab[:, 0] *= 100 / 255; lab[:, 1:] -= 128
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(-1).astype(float)
        cm = {}
        for x, c in zip(anns, cats):
            m = render_instance_mask(x["segmentation"], h, w).astype(bool)
            cm[c] = cm.get(c, np.zeros((h, w), bool)) | m
        stat = {c: (np.median(lab[m.reshape(-1)], 0), np.median(gray[m.reshape(-1)]), lab[m.reshape(-1)].std(0)[0])
                for c, m in cm.items()}
        for (ii, ref), (n, g) in drop.items():
            if ii != i: continue
            c = cats_q[(ii, ref)]; L, G, _ = stat[c]
            others = [(np.linalg.norm(L - Lo), abs(G - Go), co) for co, (Lo, Go, _) in stat.items() if co != c]
            if not others: dE, dG, co = np.nan, np.nan, "-"
            else: dE, dG, co = min(others, key=lambda t: t[1])   # the neighbour closest in GREY
            rows.append((split, i, ref, c, n, g, g - n, dE, dG, co))
rows.sort(key=lambda t: t[6])
print(f"{'split':9} sheet ref cat        none   gray   drop   | closest-in-grey other: dE(colour) dGrey")
for t in rows:
    print(f"{t[0]:9} {t[1]:5} {t[2]:3} {t[3]:10} {t[4]:.3f}  {t[5]:.3f}  {t[6]:+.3f}  | {t[9]:10} {t[7]:6.1f} {t[8]:6.1f}")
R = np.array([(t[6], t[7], t[8]) for t in rows if not np.isnan(t[7])])
collapse = (R[:, 1] > 15) & (R[:, 2] < 10)          # clearly different colours, near-identical grey
print(f"\nquestions with a colour-distinct (dE>15) but grey-alike (dGrey<10) neighbour: {collapse.sum()}/{len(R)}")
print(f"  mean IoU drop, collapsed: {R[collapse,0].mean():+.3f}   not collapsed: {R[~collapse,0].mean():+.3f}")
