"""Runpod Serverless worker for the pattern-segmentation product (2026-11-23).

Request  (`/runsync` or `/run`):  {"input": {"image_b64": "<base64 PNG/JPEG bytes>", "box": [x, y, w, h]}}
                                   box in native px of that image; optional "scales": [5120, 4096, 3072]
Response: {"polygons": [{"outer": [[x, y], ...], "holes": [...]}, ...], "vertices": n, "size": [w, h], "box": [...],
           "timing_s": {...}, "device": "..."}   or {"error": "..."}

The checkpoint comes from the private Hub repo at start-up (HF_TOKEN in the endpoint's environment, ideally a Runpod secret), or
from CHECKPOINT=/path if it is baked in. A warm worker keeps the model and the last image's backbone features."""
import base64
import os
import traceback

import runpod

from segment import Segmenter

CK = os.environ.get("CHECKPOINT")
SEG = Segmenter(CK) if CK else Segmenter.from_hub()


def handler(job):
    try:
        inp = job["input"]
        data = base64.b64decode(inp["image_b64"])
        box = inp["box"]
        if len(box) != 4:
            return {"error": "box must be [x, y, w, h]"}
        kw = {"scales": tuple(int(s) for s in inp["scales"])} if inp.get("scales") else {}
        return SEG.run(data, box, **kw)
    except Exception as e:                                   # report, do not crash the worker
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
