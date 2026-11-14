"""Per-arm summary from evaluate_refunet_selection.py --product metrics (no maps needed): product mIoU, vertices / label
median (label vertices as product.py counts them: vertcount.label_vertices on the split's records), and chosen questions.
Usage: python3 scripts/pp_probe/fgab_summary.py EVAL_DIR [EVAL_DIR ...]   (each holds product_{valfix4,hf14fix4,gemini30}.json)"""
import sys, json
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import common
from vertcount import label_vertices

WATCH = {"hf14fix4": [(2, 1)]}
recs = {}
for d in sys.argv[1:]:
    for split in ("valfix4", "hf14fix4", "gemini30"):
        f = Path(d) / f"product_{split}.json"
        if not f.exists():
            continue
        if split not in recs:
            recs[split] = {j[0]: j[1] for j in common.jobs(split)}
        m = json.load(open(f)); sel = m["selections"]
        r = np.array([s["vertices"] / lv for s in sel if (lv := label_vertices(recs[split][s["image_index"]], s["category"])) > 0])
        line = (f"{Path(d).name} {split}: {len(sel)} q, product mIoU {m['mean_iou']:.4f}, vertices / label median "
                f"{np.median(r):.2f}x (p90 {np.percentile(r, 90):.1f}x)")
        for sh, q in WATCH.get(split, []):
            s = [x for x in sel if x["image_index"] == sh and x["reference_instance"] == q][0]
            line += f" | {sh:02d} q{q}: IoU {s['iou']:.4f}, {s['vertices']} vertices (label {label_vertices(recs[split][sh], s['category'])})"
        print(line)
