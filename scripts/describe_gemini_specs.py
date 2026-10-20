#!/usr/bin/env python3
"""Describe generated (or real) drawings in the facsimile spec vocabulary (generate_synthetic_fc.py --spec) and tally
what the vocabulary cannot express -- the generator's gap list (codex_doc 2026-10-20, "synthetic facsimiles").

One Gemini call per image (image + the vocabulary below), JSON out: a spec the generator can try, plus `gap_tags` from a
fixed list and free-text `gaps`. Resumable: images already in --out are skipped.

    python3 scripts/describe_gemini_specs.py --images data/image_generation/gemini_r5 --out data/image_generation/gemini_r5/specs_described.jsonl
    python3 scripts/describe_gemini_specs.py --tally data/image_generation/gemini_r5/specs_described.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GAP_TAGS = [
    "cross_gable_or_steep_tudor", "multiple_front_gables", "dormer", "hip_and_gable_mix", "flat_or_parapet_roof",
    "mansard_or_gambrel", "apartment_or_multistorey_block", "balconies", "detached_second_building",
    "three_plus_stacked_wall_bands", "material_change_mid_wall", "bay_or_bump_out", "porch_or_columns_detail",
    "turret_or_curved_wall", "arched_openings", "non_rectangular_footprint", "split_level_or_stepped_storeys",
    "section_or_cutaway", "site_or_landscape_context",
    "perspective_or_3d", "material_not_in_vocabulary", "pattern_style_not_in_vocabulary", "colour_markup_overlay",
    "floor_plan_furniture_or_fixtures", "floor_plan_room_shapes", "roof_plan_complex_hips", "other"]

PROMPT = f"""You are describing an architectural drawing so a procedural generator can draw a close copy of it.
Answer with ONE JSON object and nothing else:
{{"mode": "elevation" | "roof_plan" | "freeform" (floor plans, sections and anything else),
 "shaded": true if surfaces are filled with colour, false if it is line-only,
 "views": 1 | 2 | 4 (number of separate building views on the sheet; use the nearest),
 "view_list": the views shown, in order, from "front", "rear", "left", "right" (the street-facing one is "front";
           a sheet with ONE view of an unknown side is ["front"]),
 "house": {{"stories": 1|2|3, "roof": "gable"|"hip"|"flat"|"shed",
           "ridge": "side" if the main ridge runs left-right in the first view (eaves face the viewer) or "front" if the
           main gable END faces the viewer,
           "facade_aspect": overall building width / height from ground to the top of the roof, in the first view (a number),
           "cross_gables": 0|1|2 (gabled bays projecting from the front wall with a ridge perpendicular to the main ridge),
           "split": true if part of the main wall (an end or entry bay, full height) is a different material, "units": null or 2..5 (identical side-by-side townhouse units),
           "wing": true if an attached garage or side wing, "ell": true if a rear ell, "porch": bool, "chimney": bool,
           "dormers": bool, "accent": null|"gable"|"wainscot"|"upper"|"block" (where a second wall material sits:
           gable triangles, a base band, the upper storey, or a whole block/wing), "accent2": same values or null,
           "label_trim": true if trim boards are a prominent separate material}},
 "styles": {{role: {{"kind": k, "base": [r,g,b]}}}} for the roles that appear, roles from main (the main wall),
           accent, accent2, roof, foundation, chimney; wall kinds: lap, vertical, bb (board and batten), shingle,
           brick, stone, stipple (textured stucco), flat (plain painted/stucco), panel (fibre-cement panels with a reveal grid); roof kinds: asphalt, seam (standing seam
           metal), tile_roof, shingle, flat_roof; "base" only when shaded (the fill colour, 0-255 RGB),
 "gap_tags": every tag from this list that describes something in the drawing the vocabulary above CANNOT express:
           {json.dumps(GAP_TAGS)},
 "gaps": short phrases for each gap, most important first}}
For plans fill mode/views/gap_tags/gaps and leave house/styles empty. Be literal about what is drawn."""


def describe(client, types, path, model):
    from PIL import Image
    import io
    im = Image.open(path).convert("RGB")
    im.thumbnail((2048, 2048))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2))
    spec = json.loads(resp.text)
    spec["name"] = Path(path).stem
    return spec


def tally(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "error" not in r]
    modes = Counter(r.get("mode") for r in rows)
    tags = Counter(t for r in rows for t in set(r.get("gap_tags", [])))
    elev = [r for r in rows if r.get("mode") == "elevation"]
    etags = Counter(t for r in elev for t in set(r.get("gap_tags", [])))
    print(f"{len(rows)} described; modes {dict(modes)}; with no gap: {sum(not r.get('gap_tags') for r in rows)}")
    print(f"{'gap tag':40s} {'all':>5s} {'elev':>5s}")
    for t, n in tags.most_common():
        print(f"{t:40s} {n:5d} {etags.get(t, 0):5d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images")
    ap.add_argument("--out")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tally")
    a = ap.parse_args()
    if a.tally:
        return tally(a.tally)
    from google import genai
    from google.genai import types
    client = genai.Client()
    done = set()
    if os.path.exists(a.out):
        done = {r["name"] for r in map(json.loads, filter(str.strip, open(a.out))) if "error" not in r}
    imgs = [p for p in sorted(Path(a.images).glob("*.png")) if p.stem not in done]
    print(f"{len(imgs)} to describe ({len(done)} done)", flush=True)
    with ThreadPoolExecutor(a.workers) as pool, open(a.out, "a") as fo:
        futs = {pool.submit(describe, client, types, p, a.model): p for p in imgs}
        for n, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            try:
                row = f.result()
            except Exception as e:                       # noqa: BLE001 - recorded, resumable
                row = {"name": p.stem, "error": f"{type(e).__name__}: {e}"[:300]}
            fo.write(json.dumps(row) + "\n"); fo.flush()
            if n % 20 == 0:
                print(f"{n}/{len(imgs)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
