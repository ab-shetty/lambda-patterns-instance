#!/usr/bin/env python3
"""Quantify instance size (px) and count distributions: real vs synthetic,
including sizes AFTER the 2048 longest-side resize the model trains at."""
import io, json
import numpy as np
from PIL import Image
from refmask2former.dataset import render_instance_mask, load_parquet_records

MAXSIZE = 2048


def load(rec):
    image = np.array(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
    anns = rec["annotations"]
    if isinstance(anns, str):
        anns = json.loads(anns)
    return image, anns


def collect(ds, k):
    minsides_orig, minsides_resized, counts = [], [], []
    for i in range(min(k, len(ds))):
        img, anns = load(ds[i])
        h, w = img.shape[:2]
        scale = MAXSIZE / max(h, w)
        counts.append(len(anns))
        for a in anns:
            m = render_instance_mask(a["segmentation"], h, w)
            ys, xs = np.where(m > 0)
            if len(ys) == 0:
                continue
            bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            ms = min(bw, bh)
            minsides_orig.append(ms)
            minsides_resized.append(ms * scale)
    return (np.array(minsides_orig), np.array(minsides_resized), np.array(counts))


def report(name, mo, mr, cnt):
    print(f"\n[{name}]  {len(mo)} instances over {len(cnt)} images")
    print(f"  instances/img : mean {cnt.mean():.1f}  median {np.median(cnt):.0f}  max {cnt.max()}")
    print(f"  bbox min-side (orig px) : p10 {np.percentile(mo,10):.0f}  median {np.median(mo):.0f}  p90 {np.percentile(mo,90):.0f}  min {mo.min():.0f}")
    print(f"  bbox min-side @2048 px  : p10 {np.percentile(mr,10):.0f}  median {np.median(mr):.0f}  p90 {np.percentile(mr,90):.0f}  min {mr.min():.0f}")
    for thr in (16, 32, 48):
        frac = (mr < thr).mean()
        print(f"    frac instances with @2048 min-side < {thr}px : {frac:.2%}")


real = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                            config="real-world-test", split="test")
synth = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data")
report("REAL", *collect(real, 28))
report("SYNTH", *collect(synth, 150))
