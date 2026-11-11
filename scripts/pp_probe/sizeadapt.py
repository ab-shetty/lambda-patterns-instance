"""Size-adaptive multi-scale (2026-11-11): msweep rows say real sheets (mostly > 4096 px) gain from 5120+4096+3072 while Gemini
sheets (~2.7k px) gain from 4096+3072+2048 (equal). Choose the blend from the sheet's native long side; reads msweep rows.
Usage: python3 scripts/pp_probe/sizeadapt.py"""
import sys, json, glob
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common

SPL = [("valfix3", "gem2s_x"), ("hf14fix2", "gem2s_x"), ("gemini30", "gem2s_x"), ("gemdev", "xblend-kq30-kms30")]
size = {}
for split, _ in SPL:
    for job in common.jobs(split):
        idx, rec = job[0], job[1]
        if "image_path" in rec:
            im = cv2.imread(rec["image_path"], cv2.IMREAD_REDUCED_GRAYSCALE_2); s = 2 * max(im.shape[:2])
        else:
            b = rec["image"]; b = b["bytes"] if isinstance(b, dict) else b
            s = max(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE).shape[:2])
        size[(split, idx)] = s
SMALL, LARGE, PROD = "4096+3072+2048 eq", "5120+4096+3072", "4096+2048 (prod)"
for split, tag in SPL:
    R = [r for f in sorted(glob.glob(f"scripts/pp_probe/rows_msweep_{tag}_{split}*.json")) for r in json.load(open(f))]
    ss = np.array([size[(split, r["sheet"])] for r in R])
    print(f"{split}: native long side quartiles {np.percentile(ss, [0, 25, 50, 75, 100]).astype(int)}")
    for o in ("", " off1"):
        base = np.mean([r[PROD + o] for r in R])
        line = []
        for cut in (3000, 3500, 4096, 5000):
            v = [r[SMALL + o] if size[(split, r["sheet"])] <= cut else r[LARGE + o] for r in R]
            line.append(f"cut {cut}: {np.mean(v):.4f} ({np.mean(v) - base:+.4f})")
        print(f"  prod{o} {base:.4f} | " + " | ".join(line))
