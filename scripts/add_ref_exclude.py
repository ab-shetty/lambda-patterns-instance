#!/usr/bin/env python3
"""Write text regions into a hand-labelled local pool's annotation files as "ref_exclude" rings.

Real / Gemini labels carry no ref_exclude, so training's reference sampler (and the --local-pool
evaluator) put boxes on room tags, material callouts and slope labels. Tesseract word boxes
(scripts/text_mask.py, upright + rotated 90 degrees), dilated, become flat outer rings in the same
format the synthetic generator writes; `reference_region` then keeps reference boxes off them (a
piece wholly covered is never a reference). Targets are unchanged. Run it on the CLEAN sources: the
offline augmentation (build_mix.augment_scenes) carries the rings through its rotations and flips.

    PYTHONPATH=. python3 scripts/add_ref_exclude.py --pool data/roboflow/floz-gen-gemini-r4-clean
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from scripts.text_mask import text_mask


def rings_for(path):
    m = text_mask(cv2.imread(str(path))).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(float).ravel().tolist() for c in cs if len(c) >= 3], float(m.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, action="append", help="local pool dir (repeatable)")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    for pool in map(Path, a.pool):
        files = sorted((pool / "annotations").glob("*.json"))
        anns = [json.loads(f.read_text()) for f in files]
        with Pool(a.workers) as p:
            res = p.map(rings_for, [pool / "images" / x["image"]["file_name"] for x in anns])
        for f, ann, (rings, frac) in zip(files, anns, res):
            keep = [r for r in (ann.get("ref_exclude") or []) if r]
            ann["ref_exclude"] = keep + rings
            f.write_text(json.dumps(ann))
        print(f"{pool}: {len(files)} sheets, text {np.mean([r[1] for r in res]):.1%} of pixels, "
              f"{sum(len(r[0]) for r in res)} rings")


if __name__ == "__main__":
    main()
