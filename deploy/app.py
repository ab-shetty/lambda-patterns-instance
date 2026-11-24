#!/usr/bin/env python3
"""Box-and-see app for the 28 real sheets (HF real-world-test), 2026-11-23. Pick a sheet, drag a box on a pattern, get the
product's polygons (fgavg78 + scripts/product.py) drawn over the sheet, with the human label of the family under the box for
comparison (IoU, vertex counts) and a JSON download.

    # GPU worker on Runpod Serverless (deploy/handler.py); the Runpod key is injected by the session proxy or read from RUNPOD_API_KEY
    python3 deploy/app.py --endpoint <ENDPOINT_ID> [--port 7860]
    # model in this process (a GPU VM; on CPU each box takes 20-80 s)
    python3 deploy/app.py --local data/runs/ck_best/fgavg78.pth

Standard library HTTP server + one HTML page (deploy/app.html); the sheets come from the private HF dataset abshetty/floz-synth-v5
(real-world-test), so HF access is needed where this runs. The label comparison uses the raw Roboflow annotations (no eval label
fixes / ignore regions), so its IoU can differ slightly from the reported eval numbers."""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

VAL = [4, 5, 6, 8, 9, 10, 13, 15, 17, 19, 20, 21, 22, 26]
HF14 = [12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24]
DISPLAY = 2400                     # long side of the image sent to the browser


def render_instance_mask(seg, h, w):
    """COCO polygons -> mask; the first ring is the outer, the rest holes (as refmask2former.dataset)."""
    m = np.zeros((h, w), np.uint8)
    for j, r in enumerate(seg or []):
        cv2.fillPoly(m, [np.asarray(r, np.float32).reshape(-1, 2).astype(np.int32)], 0 if j else 1)
    return m


def rasterize(polys, hw):
    """Product polygons -> bool mask (as scripts/polygonize.rasterize)."""
    m = np.zeros(hw, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p["outer"], np.int32)], 1)
        for hole in p["holes"]:
            cv2.fillPoly(m, [np.array(hole, np.int32)], 0)
    return m.astype(bool)


class Sheets:
    def __init__(self):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        f = hf_hub_download("abshetty/floz-synth-v5", "real-world-test/test-00000.parquet", repo_type="dataset")
        self.recs = pq.read_table(f).to_pylist()
        self._img, self._disp = {}, {}

    def __len__(self):
        return len(self.recs)

    def raw(self, i):
        b = self.recs[i]["image"]
        return b["bytes"] if isinstance(b, dict) else b

    def image(self, i):
        if i not in self._img:
            self._img = {i: cv2.imdecode(np.frombuffer(self.raw(i), np.uint8), cv2.IMREAD_COLOR)}
        return self._img[i]

    def display(self, i):
        if i not in self._disp:
            img = self.image(i); s = min(1.0, DISPLAY / max(img.shape[:2]))
            small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
            self._disp[i] = (cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), s)
        return self._disp[i]

    def families(self, i):
        anns = self.recs[i]["annotations"]
        anns = json.loads(anns) if isinstance(anns, str) else anns
        out = {}
        for a in anns:
            out.setdefault(a.get("category_name", "pattern"), []).append(a["segmentation"])
        return out

    def label_for_box(self, i, box):
        """The labelled family covering most of the box: (name, mask, polygons, vertices) or None."""
        img = self.image(i); h, w = img.shape[:2]; x, y, bw, bh = box
        best = None
        for name, segs in self.families(i).items():
            m = np.zeros((h, w), bool)
            for s in segs:
                m |= render_instance_mask(s, h, w).astype(bool)
            cover = m[y:y + bh, x:x + bw].mean() if bw > 0 and bh > 0 else 0
            if best is None or cover > best[0]:
                best = (cover, name, m, segs)
        if best is None or best[0] < 0.5:
            return None
        _, name, m, segs = best
        polys = [[np.asarray(r, float).reshape(-1, 2).round().astype(int).tolist() for r in s] for s in segs]
        return name, m, polys, sum(len(r) for p in polys for r in p)


class RunpodBackend:
    def __init__(self, endpoint, timeout=300):
        self.base = f"https://api.runpod.ai/v2/{endpoint}"
        self.key = os.environ.get("RUNPOD_API_KEY", "proxy-injected")   # the session proxy replaces the header
        self.timeout = timeout

    def _call(self, url, body=None):
        req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                                     method="POST" if body is not None else "GET")
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    def run(self, data, box):
        t0 = time.time()
        r = self._call(self.base + "/runsync", {"input": {"image_b64": base64.b64encode(data).decode(), "box": list(box)}})
        while r.get("status") in ("IN_QUEUE", "IN_PROGRESS") and time.time() - t0 < self.timeout:   # cold start / long job
            time.sleep(2)
            r = self._call(f"{self.base}/status/{r['id']}")
        if r.get("status") != "COMPLETED":
            raise RuntimeError(f"Runpod job {r.get('status')}: {str(r.get('error') or r)[:500]}")
        out = r["output"]
        if "error" in out:
            raise RuntimeError(out["error"])
        out["round_trip_s"] = round(time.time() - t0, 2)
        out["runpod_delay_s"] = round(r.get("delayTime", 0) / 1000, 2)          # queue + cold start
        return out


class LocalBackend:
    def __init__(self, checkpoint):
        sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT), str(ROOT / "scripts")]
        from segment import Segmenter
        self.seg = Segmenter(checkpoint)

    def run(self, data, box):
        t0 = time.time(); out = self.seg.run(data, box); out["round_trip_s"] = round(time.time() - t0, 2)
        return out


def make_handler(sheets, backend):
    page = (Path(__file__).resolve().parent / "app.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write("[app] " + fmt % a + "\n")

        def _send(self, code, body, ctype="application/json"):
            body = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/api/sheets":
                out = []
                for i in range(len(sheets)):
                    r = sheets.recs[i]
                    out.append({"i": i, "name": r["filename"], "w": r["width"], "h": r["height"],
                                "split": "val" if i in VAL else "HF14" if i in HF14 else "-"})
                return self._send(200, out)
            if self.path.startswith("/api/image/"):
                i = int(self.path.rsplit("/", 1)[1].split(".")[0])
                jpg, s = sheets.display(i)
                self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("X-Scale", str(s))
                self.send_header("Content-Length", str(len(jpg))); self.end_headers(); self.wfile.write(jpg)
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/segment":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                i = int(req["sheet"]); box = [int(round(v)) for v in req["box"]]
                out = backend.run(sheets.raw(i), box)
                img = sheets.image(i)
                lab = sheets.label_for_box(i, out["box"])
                if lab:
                    name, m, lpolys, lv = lab
                    pred = rasterize(out["polygons"], img.shape[:2])
                    out["label"] = {"family": name, "polygons": lpolys, "vertices": lv,
                                    "iou": round(float((pred & m).sum() / max(1, (pred | m).sum())), 4)}
                return self._send(200, out)
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--endpoint", help="Runpod Serverless endpoint id")
    g.add_argument("--local", help="checkpoint .pth: run the model in this process")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    sheets = Sheets()
    backend = RunpodBackend(a.endpoint) if a.endpoint else LocalBackend(a.local)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(sheets, backend))
    print(f"serving {len(sheets)} sheets on http://{a.host}:{a.port} ({'runpod ' + a.endpoint if a.endpoint else 'local'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
