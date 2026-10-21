#!/usr/bin/env python3
"""Label-shape twins of Gemini30 sheets (2026-10-20, the user's call after the --spec twins came out too different).

A twin keeps the Gemini sheet's own geometry -- every labelled family's exact mask, the windows / doors cut out of it,
and everything outside the labels -- and repaints only the INSIDE of each labelled family with our generator's
texture (generate_synthetic_v6.draw_fill) for that family's material: spacing, line weight and line colour are the
generator's, the fill colour is the family's own median colour (colour sheets) or white paper (line-only sheets).
Labels are copied unchanged and the reference boxes are the same, so each twin question pairs with its original:
good on the twin but bad on the original = the generator draws that material differently (appearance);
bad on both = content / layout. Twins are diagnostic only, never in a training pool.

    python3 scripts/facsimile_relabel.py --map image_generation/facsimile/gemini30_family_kinds.json \
        --out data/synthetic/facsimile_g30_relabel
writes images/, annotations/, and boxes.json (eval_boxes/gemini30_v1.json remapped to the twin pool's indices).
Scale: S (px per foot) = width of the union of the labels / 45 ft (a one-house elevation), as the generator sizes views.
"""
import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_synthetic_v6 as G  # noqa: E402
from refmask2former.dataset import render_instance_mask  # noqa: E402

POOL = "data/eval_pools/gemini30_s20261006"
ROOF = {"asphalt", "seam", "tile_roof", "flat_roof"}


def family_style(kind, img, mask, shaded, S, seed, name):
    rng = random.Random(seed)
    ink = 25
    app = {"colour": shaded, "level": "normal", "ink": (ink,) * 3, "outline": (ink,) * 3,
           "ink_fill": (rng.randint(110, 175),) * 3, "paper": (255, 255, 255), "trim": (255, 255, 255), "outline_lw": 1.0}
    base = None
    if shaded:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        px = img[mask & (g > np.percentile(g[mask], 30))]          # the fill, not its own lines
        base = tuple(int(v) for v in np.median(px, 0)[::-1])         # BGR -> RGB
    mk = {"stipple": "stucco", "flat": "stucco", "panel": "lap"}.get(kind, kind)
    st = G.make_style(rng, mk, app, seed, S, name, roof=kind in ROOF, base_override=base)
    if kind == "panel":                         # fibre-cement panels (generate_synthetic_fc --panel-siding's params)
        pw = rng.choice([2.0, 3.0, 4.0, 4.0])
        st.kind, st.params = "panel", {"pw": pw, "ph": rng.choice([4.0, 6.0, 8.0, 9.0, 10.0]), "stagger": rng.random() < 0.35,
                                       "sp": pw}
    for t in range(40):                                             # stucco: stipple or flat as asked
        if kind not in ("stipple", "flat") or st.kind == kind:
            break
        st = G.make_style(random.Random(seed * 7919 + t), "stucco", app, seed, S, name, base_override=base)
    if shaded:
        lum = 0.299 * st.base[0] + 0.587 * st.base[1] + 0.114 * st.base[2]
        st.line = G.mix(st.base, (0, 0, 0), 0.35) if lum > 90 else G.mix(st.base, (255, 255, 255), 0.3)
    else:                                       # line-only: the sheet's own paper (scans / photos are not white)
        paper = np.percentile(img.reshape(-1, 3), 90, axis=0)
        st.base, st.line = tuple(int(v) for v in paper[::-1]), app["ink_fill"]
    st.params.pop("grad", None)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help='{"<sheet index>": {"shaded": bool, "kinds": {"pattern1": "lap", ...}}}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", default="eval_boxes/gemini30_v1.json", help="eval boxes to remap ('' = none)")
    ap.add_argument("--pool", default=POOL, help="images/ + annotations/ of the Gemini sheets")
    a = ap.parse_args()
    fmap = json.load(open(a.map))
    anns = sorted(glob.glob(f"{a.pool}/annotations/*.json"))
    boxes = json.load(open(a.boxes)) if a.boxes else {}
    os.makedirs(f"{a.out}/images", exist_ok=True)
    os.makedirs(f"{a.out}/annotations", exist_ok=True)
    new_boxes, order = {}, []
    for k, (si, spec) in enumerate(sorted(((k_, v) for k_, v in fmap.items() if k_.isdigit()), key=lambda kv: int(kv[0]))):
        si = int(si)
        A = json.load(open(anns[si]))
        img = cv2.imread(f"{a.pool}/images/{A['image']['file_name']}")
        h, w = img.shape[:2]
        fams = {}
        for an in A["annotations"]:
            m = render_instance_mask(an["segmentation"], h, w).astype(bool)
            fams[an["category_name"]] = fams.get(an["category_name"], np.zeros((h, w), bool)) | m
        allm = np.any(list(fams.values()), axis=0)
        xs = np.nonzero(allm.any(0))[0]
        S = max(8.0, (xs.max() - xs.min() + 1) / 45.0)
        out = img.copy()
        for j, (cat, m) in enumerate(sorted(fams.items())):
            kind = spec["kinds"].get(cat)
            if kind is None:
                continue
            st = family_style(kind, img, m, spec["shaded"], S, 1000 * si + j, cat)
            ys, xs2 = np.nonzero(m)
            tmp = out.copy()
            G.draw_fill(tmp, box(xs2.min(), ys.min(), xs2.max() + 1, ys.max() + 1), st, S, w, h)
            out[m] = tmp[m]
        name = f"twin_{si:02d}_{A['image']['file_name']}"
        cv2.imwrite(f"{a.out}/images/{name}", out)
        A["image"]["file_name"] = name
        json.dump(A, open(f"{a.out}/annotations/{os.path.splitext(name)[0]}.json", "w"))
        order.append((name, si))
    # the local pool indexes sheets by sorted annotation file name: remap the boxes to those indices
    names = sorted(os.path.splitext(n)[0] for n, _ in order)
    for n, si in order:
        new_boxes[str(names.index(os.path.splitext(n)[0]))] = boxes.get(str(si), {})
    new_boxes["_meta"] = {"from": a.boxes, "pool_index_to_gemini30": {str(names.index(os.path.splitext(n)[0])): si
                                                                       for n, si in order}}
    json.dump(new_boxes, open(f"{a.out}/boxes.json", "w"), indent=1)
    print(f"{len(order)} twins -> {a.out}")


if __name__ == "__main__":
    main()
