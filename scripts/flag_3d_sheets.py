#!/usr/bin/env python3
"""Ask Gemini, one image at a time, whether a sheet is a flat orthographic CAD elevation or a 3-D / perspective /
rendered-depth drawing (the latter is not what clients send; excluded from facsimile work, user 2026-10-21).

    GEMINI_API_KEY=... python3 scripts/flag_3d_sheets.py IMAGE_DIR OUT.jsonl
"""
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT = """Is this architectural drawing a FLAT ORTHOGRAPHIC elevation, as exported from CAD / Revit (all walls drawn
parallel to the sheet, no vanishing points, depth only suggested by simple flat shadows), or is it 3-D: a perspective or
axonometric view, a rendering with realistic depth / lighting, or a sketch showing receding sides of the building?
Answer with ONE JSON object: {"flat_orthographic": true|false, "why": "few words"}"""


def ask(client, types, p):
    from PIL import Image
    im = Image.open(p).convert("RGB"); im.thumbnail((1536, 1536))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    r = client.models.generate_content(model="gemini-2.5-pro", contents=[types.Part.from_bytes(data=buf.getvalue(),
                                       mime_type="image/jpeg"), PROMPT],
                                       config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
    d = json.loads(r.text); d["name"] = Path(p).stem
    return d


def main(img_dir, out):
    from google import genai
    from google.genai import types
    client = genai.Client()
    done = {json.loads(l)["name"] for l in open(out) if "error" not in l} if os.path.exists(out) else set()
    imgs = [p for p in sorted(Path(img_dir).glob("*.png")) if p.stem not in done]
    with ThreadPoolExecutor(4) as pool, open(out, "a") as fo:
        futs = {pool.submit(ask, client, types, p): p for p in imgs}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as e:                       # noqa: BLE001 - recorded, resumable
                row = {"name": futs[f].stem, "error": str(e)[:200]}
            fo.write(json.dumps(row) + "\n"); fo.flush()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
