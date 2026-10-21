#!/usr/bin/env python3
"""Family -> material-kind maps for facsimile_relabel.py, from Gemini instead of by hand (2026-10-21).

Each sheet is sent with every labelled family outlined in its own colour (legend in the prompt); Gemini names the
generator kind of each family and whether the sheet is colour-filled. Output: the --map JSON facsimile_relabel.py
reads ({"<sheet index>": {"shaded": bool, "kinds": {"pattern1": "lap", ...}}}, index = sorted annotation file order).

    GEMINI_API_KEY=... python3 scripts/facsimile_kinds.py --pool data/roboflow/floz-gen-gemini-r4-clean \
        --out image_generation/facsimile/r4_family_kinds.json
"""
import argparse
import glob
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask  # noqa: E402

KINDS = {"lap": "horizontal lap siding", "vertical": "vertical siding / standing lines", "bb": "board and batten",
         "shingle": "shingle / shake siding", "brick": "brick", "stone": "random stone", "ashlar": "cut / ashlar stone",
         "stipple": "textured stucco (dots)", "flat": "plain flat fill", "panel": "fibre-cement panels with a reveal grid",
         "concrete": "concrete (speckle)", "asphalt": "asphalt shingle roof", "seam": "standing-seam metal roof",
         "tile_roof": "tile roof", "flat_roof": "flat / membrane roof", "plank": "wood plank floor", "grid": "square tile grid",
         "hatch": "diagonal hatch", "cross": "cross hatch", "solid": "solid dark fill"}
COLS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (128, 128, 0)]
NAMES = ["red", "green", "blue", "orange", "purple", "cyan", "magenta", "olive"]


def overlay(img, fams):
    out = img.copy()
    legend = []
    for j, (cat, m) in enumerate(sorted(fams.items())):
        c = COLS[j % len(COLS)]
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, c[::-1], max(4, img.shape[1] // 300))
        legend.append(f'"{cat}" = {NAMES[j % len(NAMES)]} outline')
    return out, legend


def ask(client, types, img, fams):
    from PIL import Image
    ov, legend = overlay(img, fams)
    parts = []
    for im in (img, ov):
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); pil.thumbnail((2048, 2048))
        buf = io.BytesIO(); pil.save(buf, "JPEG", quality=88)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    prompt = (f"The first image is an architectural drawing; the second is the same drawing with labelled pattern "
              f"regions outlined: {'; '.join(legend)}. For each outlined region name the material pattern INSIDE it, "
              f"choosing from {json.dumps(KINDS)}. Answer with ONE JSON object: "
              '{"shaded": true if surfaces are colour-filled else false, "kinds": {"<label name>": "<kind key>"}}')
    r = client.models.generate_content(model="gemini-2.5-pro", contents=parts + [prompt],
                                       config=types.GenerateContentConfig(response_mime_type="application/json",
                                                                          temperature=0))
    return json.loads(r.text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma list of image stems (before _png) to map")
    a = ap.parse_args()
    from google import genai
    from google.genai import types
    client = genai.Client()
    anns = sorted(glob.glob(f"{a.pool}/annotations/*.json"))
    done = json.load(open(a.out)) if os.path.exists(a.out) else {}
    only = set(a.only.split(",")) if a.only else None
    jobs = {}
    with ThreadPoolExecutor(4) as pool:
        for si, f in enumerate(anns):
            stem = os.path.basename(f).split("_png")[0]
            if str(si) in done or (only and stem not in only):
                continue
            A = json.load(open(f))
            img = cv2.imread(f"{a.pool}/images/{A['image']['file_name']}")
            h, w = img.shape[:2]
            fams = {}
            for an in A["annotations"]:
                m = render_instance_mask(an["segmentation"], h, w).astype(bool)
                fams[an["category_name"]] = fams.get(an["category_name"], np.zeros((h, w), bool)) | m
            if fams:
                jobs[pool.submit(ask, client, types, img, fams)] = (si, stem)
        for fut in as_completed(jobs):
            si, stem = jobs[fut]
            try:
                d = fut.result()
                d["kinds"] = {k: v for k, v in d.get("kinds", {}).items() if v in KINDS}
                d["name"] = stem
                done[str(si)] = d
            except Exception as e:                       # noqa: BLE001 - rerun to retry
                print(stem, "error", str(e)[:120], flush=True)
            json.dump(done, open(a.out, "w"), indent=1)
    print(f"{len(done)} sheets mapped -> {a.out}")


if __name__ == "__main__":
    main()
