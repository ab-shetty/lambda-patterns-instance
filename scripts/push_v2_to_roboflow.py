#!/usr/bin/env python3
"""Push label fixes v2 (eval_labels/hf14_fixes_v2.json: HF14 sheets 14 and 18) into Roboflow floz-eval28-reference, so Roboflow
is the single source of the eval labels again (2026-11-11, the user). Each sheet's full annotation is rebuilt as COCO from the
image's current Roboflow boxes (every polygon kept, coordinates unchanged) plus the edit, uploaded with overwrite, re-read and
compared polygon by polygon. Backups of the pre-push annotations: eval_labels/roboflow_backup_2026-11-11/.
  18: the right-roof polygon (over HF annotation 9) pattern3 -> pattern3b   (v2 "reassign")
  14: + pattern1 rectangle [[4895,926],[5042,926],[5042,985],[4895,985]] (v2: panel under window 09)
  14b: drop the old pattern2 rectangle on that same panel (Roboflow had it pattern2, v2 pattern1; the user, 2026-11-11: keep
       the one that matches the other horizontal lap-siding panels, i.e. pattern1). Built from eval_14_after_v2push.json.
Usage: python3 scripts/push_v2_to_roboflow.py SHEET|14b [--dry-run]"""
import json, os, sys, glob, subprocess, tempfile
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, cv2

PROJ = "perceive-ai/floz-eval28-reference"
KEY = os.environ["ROBOFLOW_API_KEY"]


def pts_of(bx):
    if "points" in bx:
        return [[float(p["x"]), float(p["y"])] if isinstance(p, dict) else [float(p[0]), float(p[1])] for p in bx["points"]]
    cx, cy, w, h = (float(bx[k]) for k in ("x", "y", "width", "height"))
    return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


def get(img_id):
    out = subprocess.run(["curl", "-sS", "-m", "60", f"https://api.roboflow.com/{PROJ}/images/{img_id}?api_key={KEY}"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["image"]


def sig(boxes):
    return sorted((b_label, tuple(tuple(np.round(p, 1)) for p in pts)) for b_label, pts in boxes)


step, dry = sys.argv[1], "--dry-run" in sys.argv
sheet = int(step.rstrip("b"))
files = {14: "WULmP5UDWIg4LMKW2MoQ", 18: "xGTk41KmyVJVBoaeYgSg"}
img_id = files[sheet]
src = "eval_14_after_v2push.json" if step == "14b" else f"eval_{sheet}.json"
im = json.load(open(f"eval_labels/roboflow_backup_2026-11-11/{src}"))["image"]
an = im["annotation"]; h, w = int(an["height"]), int(an["width"])
boxes = [(b["label"], pts_of(b)) for b in an["boxes"]]
if sheet == 18:
    sys.path.insert(0, "scripts")
    from refmask2former import load_parquet_records
    from refmask2former.dataset import render_instance_mask
    recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
    anns = recs[18]["annotations"]; anns = json.loads(anns) if isinstance(anns, str) else anns
    m9 = render_instance_mask(anns[9]["segmentation"], h, w).astype(bool)
    hits = []
    for j, (lab, p) in enumerate(boxes):
        mm = np.zeros((h, w), np.uint8); cv2.fillPoly(mm, [np.round(np.array(p)).astype(np.int32)], 1)
        a = mm.astype(bool)
        if lab == "pattern3" and (a & m9).sum() / m9.sum() > 0.9 and (a & ~m9).sum() / a.sum() < 0.2:
            hits.append(j)
    assert len(hits) == 1, f"expected one right-roof polygon, found {hits}"
    j = hits[0]; boxes[j] = ("pattern3b", boxes[j][1])
    want = boxes
elif step == "14b":
    drop = [j for j, (lab, p) in enumerate(boxes) if lab == "pattern2" and 4890 <= np.array(p)[:, 0].min() <= 4900
            and 920 <= np.array(p)[:, 1].min() <= 930 and np.array(p)[:, 0].max() < 5050]
    assert len(drop) == 1, drop
    want = [b for j, b in enumerate(boxes) if j != drop[0]]
elif sheet == 14:
    want = boxes + [("pattern1", [[4895., 926.], [5042., 926.], [5042., 985.], [4895., 985.]])]
cats = sorted({l for l, _ in want})
coco = {"images": [{"id": 0, "file_name": im["name"], "width": w, "height": h}],
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(cats)],
        "annotations": []}
for k, (lab, p) in enumerate(want):
    a = np.array(p)
    x0, y0 = a.min(0); x1, y1 = a.max(0)
    coco["annotations"].append({"id": k + 1, "image_id": 0, "category_id": cats.index(lab) + 1, "iscrowd": 0,
                                "segmentation": [a.flatten().tolist()], "bbox": [x0, y0, x1 - x0, y1 - y0],
                                "area": float(cv2.contourArea(a.astype(np.float32)))})
print(f"sheet {sheet}: {len(an['boxes'])} boxes now -> {len(want)} after; labels {sorted(l for l, _ in want)}")
if dry:
    sys.exit()
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(coco, f); path = f.name
r = subprocess.run(["curl", "-sS", "-m", "120", "-X", "POST",
                    f"https://api.roboflow.com/dataset/{PROJ.split('/')[1]}/annotate/{img_id}?api_key={KEY}"
                    f"&name=eval_{sheet:02d}_v2.json&overwrite=true",
                    "-H", "Content-Type: text/plain", "--data-binary", f"@{path}"], capture_output=True, text=True)
print("upload:", r.stdout[:300], r.stderr[:200])
back = get(img_id)["annotation"]["boxes"]
got = [(b["label"], pts_of(b)) for b in back]
print(f"read back: {len(got)} boxes; identical to intended: {sig(got) == sig(want)}")
if sig(got) != sig(want):
    a, b = set(sig(got)), set(sig(want))
    print("  only in Roboflow:", [x[0] for x in a - b], " only intended:", [x[0] for x in b - a])
