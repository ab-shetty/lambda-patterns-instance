#!/usr/bin/env python3
"""Self-training stage B' (2026-11-19): a Gemini vision model reviews the teacher's pseudo-label families before they become
training records. Per sheet ONE call shows Gemini the drawing, and per eligible family (pseudolabel.family_layers, `pl`) the
sheet with the family's draft tinted blue + its reference box in red, and a zoomed crop of the reference box. Gemini answers
per family: does the box show a material fill, is the tinted area that material (good / contaminated / wrong), boxes around
tinted areas that are another material, boxes around untinted areas drawn with the same material, and which other families
are the same material. The verdicts are applied by pseudolabel.py --refine DIR (apply_refine there):
  drop       reference_valid false, verdict wrong, or contaminated without any wrong_regions box;
  wrong      confident target inside a wrong_regions box -> ignored;
  missed     confident background inside a missed_regions box -> ignored (the teacher's "own material as background");
  merge      families Gemini calls the same material are merged into one family (union of targets).
Validated on Gemini30 / val labels first (pseudolabel.py --report and --refine), never on HF14.

    PYTHONPATH=.:scripts python3 scripts/selftrain/gemini_refine.py --disc data/selftrain/disc_gemini30 \
        --local-pool data/eval_pools/gemini30_s20261006 --out data/selftrain/gref_gemini30
    ... --disc D --real --out ...;   ... --disc D --images DIR --out ...
Resumable: a sheet whose verdict file exists is skipped. Needs GEMINI_API_KEY.
"""
import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pseudolabel as PL  # noqa: E402

PROMPT = """You are checking automatically produced material labels on an architectural drawing (an elevation or a plan).
In these drawings each building material (lap siding, brick, stone, shingles, stucco, board-and-batten, concrete, roofing,
flooring, ...) is drawn as a fill: a hatch, stipple, line pattern or flat colour. A MATERIAL FAMILY is every region of the
drawing filled with the same material. Rules:
- Grouping follows the material, not the direction its fill runs (roof courses that turn with each roof plane are one family).
- Regions drawn with a visibly different fill (different spacing, angle, line weight, texture or colour) are different families.
- Windows, doors, glazing, text, dimension lines, symbols, furniture, plants, cars, outlines and blank paper are NOT material;
  a window or door inside a wall is a hole in the wall, not part of it.

Image 1 is the drawing. Then, for each candidate family (ids F1, F2, ...), two images follow: an OVERLAY of the drawing
where the family's proposed regions are tinted BLUE and its reference box is outlined in RED, and a ZOOMED crop of the red
reference box. The family is meant to be: every region drawn with the same material as inside the red box.

For each family answer:
- material: a few words naming what the red box shows.
- reference_valid: true if the red box shows a material fill; false if it shows blank paper, a window, text, line work only, etc.
- verdict: "good" if the blue tint covers that material and essentially nothing else (small edge slop is fine);
  "contaminated" if the blue tint also covers a clear area of a DIFFERENT material or of non-material;
  "wrong" if most of the blue tint is not the red box's material.
- wrong_regions: boxes around blue-tinted areas that are NOT the red box's material (empty if none).
- missed_regions: boxes around areas drawn with the SAME material as the red box that are NOT tinted blue (empty if none;
  ignore slivers along edges).
- same_as: ids of the other families whose red box shows the same material (empty if none).
Boxes are [ymin, xmin, ymax, xmax] normalised to 0-1000 on Image 1. Be precise and conservative: only report a region
you are confident about."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {"families": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"}, "material": {"type": "STRING"}, "reference_valid": {"type": "BOOLEAN"},
            "verdict": {"type": "STRING", "enum": ["good", "contaminated", "wrong"]},
            "wrong_regions": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
            "missed_regions": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
            "same_as": {"type": "ARRAY", "items": {"type": "STRING"}}},
        "required": ["id", "material", "reference_valid", "verdict", "wrong_regions", "missed_regions", "same_as"]}}},
    "required": ["families"]}


def png(img_bgr, long_side):
    h, w = img_bgr.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    b = io.BytesIO(); Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(b, "PNG")
    return b.getvalue()


def overlay(img, draft, box):
    o = img.copy()
    tint = np.array([255, 90, 0], np.float32)              # BGR blue
    o[draft] = (0.6 * o[draft] + 0.4 * tint).astype(np.uint8)
    cs, _ = cv2.findContours(draft.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    t = max(2, max(img.shape[:2]) // 800)
    cv2.drawContours(o, cs, -1, (200, 60, 0), t)
    x, y, w, h = box
    cv2.rectangle(o, (x, y), (x + w, y + h), (0, 0, 255), 2 * t)
    return o


def crop(img, box):
    x, y, w, h = box
    H, W = img.shape[:2]
    p = max(w, h) // 2
    c = img[max(0, y - p):y + h + p, max(0, x - p):x + w + p].copy()
    cv2.rectangle(c, (x - max(0, x - p), y - max(0, y - p)), (x - max(0, x - p) + w, y - max(0, y - p) + h), (0, 0, 255), 2)
    s = max(1.0, 512 / max(c.shape[:2]))
    return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) if s > 1 else c


def review(d, a, client, types):
    meta = json.loads((d / "families.json").read_text())
    sid = meta["sheet"]
    out = Path(a.out) / f"{sid}.json"
    if out.exists():
        return sid, "cached"
    img = PL.load_img(sid, a)
    fams = [f for f in PL.family_layers(d, img, meta) if f["pl"]]
    if not fams:
        out.write_text(json.dumps({"sheet": sid, "ids": {}, "families": []})); return sid, "no families"
    ids = {f"F{i + 1}": f["k"] for i, f in enumerate(fams)}
    parts = [types.Part.from_text(text=PROMPT), types.Part.from_text(text="Image 1 (the drawing):"),
             types.Part.from_bytes(data=png(img, 3072), mime_type="image/png")]
    for (fid, _), f in zip(ids.items(), fams):
        parts += [types.Part.from_text(text=f"{fid} overlay:"),
                  types.Part.from_bytes(data=png(overlay(img, f["draft"], f["box"]), 2048), mime_type="image/png"),
                  types.Part.from_text(text=f"{fid} zoomed reference box:"),
                  types.Part.from_bytes(data=png(crop(img, f["box"]), 768), mime_type="image/png")]
    kw = {}
    if a.media_res:
        kw["media_resolution"] = getattr(types.MediaResolution, f"MEDIA_RESOLUTION_{a.media_res.upper()}")
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, **kw)
    for attempt in range(6):
        try:
            r = client.models.generate_content(model=a.model, contents=parts, config=cfg)
            v = json.loads(r.text)
            break
        except Exception as e:  # noqa: BLE001 -- rate limits / transient errors: back off and retry
            if attempt == 5:
                return sid, f"FAILED {type(e).__name__}: {str(e)[:200]}"
            time.sleep(10 * 2 ** attempt)
    u = getattr(r, "usage_metadata", None)
    out.write_text(json.dumps({"sheet": sid, "model": a.model, "ids": ids, "families": v["families"],
                               "usage": u.model_dump() if hasattr(u, "model_dump") else None}, indent=1))
    return sid, f"{len(fams)} families"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disc", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-pool"); src.add_argument("--real", action="store_true"); src.add_argument("--images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-pro-latest")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--media-res", default="high", help="low / medium / high ('' = the model default)")
    ap.add_argument("--only", help="file of sheet stems")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    PL.CFG.update(hi=PL.HI, lo=PL.LO, s_min=PL.S_MIN, min_area=PL.MIN_AREA, drop_fragments=True)
    if a.real:
        from evaluate_refunet_selection import load_parquet_records
        PL.CFG["recs"] = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
    dirs = sorted(p for p in Path(a.disc).iterdir() if (p / "families.json").exists())
    if a.only:
        keep = set(Path(a.only).read_text().split()); dirs = [p for p in dirs if p.name in keep]
    if a.limit:
        dirs = dirs[:a.limit]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.threads) as ex:
        for sid, msg in ex.map(lambda d: review(d, a, client, types), dirs):
            print(sid, msg, flush=True)


if __name__ == "__main__":
    main()
