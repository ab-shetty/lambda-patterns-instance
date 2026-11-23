"""One query of the shipped product (fgavg78 + scripts/product.py), reusable by the Runpod worker and the local app (2026-11-23).

    seg = Segmenter("data/runs/ck_best/fgavg78.pth")          # or Segmenter.from_hub() (needs HF_TOKEN for the private repo)
    out = seg.run(image_bytes, (x, y, w, h))                   # box in native px of the image
    out = {"polygons": [{"outer": [[x, y], ...], "holes": [[[x, y], ...], ...]}, ...], "vertices": int, "size": [w, h],
           "timing_s": {...}, "device": str}

The model runs at product.SCALES (5120 / 4096 / 3072 long side) exactly as the evaluator does: the reference is the native box
crop resized to 224 px, the box plane goes in for the roi_ref checkpoint, bf16 autocast on CUDA, sigmoid -> uint8 maps, then
product.product_polygons. The image backbone result is cached per image (by content hash) and scale, so later boxes on the same
image skip it."""
import hashlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_refunet_selection import load_refunet, _normalize_on  # noqa: E402
from refmask2former.dataset import _normalize_chw  # noqa: E402
import product  # noqa: E402

HUB_REPO = "abshetty/floz-refunet-swint-fgavg78"


def _plane(box, h0, w0, nh, nw, device):
    x, y, w, h = box
    rows = np.zeros((h0, 1), np.uint8); rows[y:y + h] = 1
    cols = np.zeros((1, w0), np.uint8); cols[0, x:x + w] = 1
    r = torch.from_numpy(cv2.resize(rows, (1, nh), interpolation=cv2.INTER_NEAREST)[:, 0]).to(device).float()
    c = torch.from_numpy(cv2.resize(cols, (nw, 1), interpolation=cv2.INTER_NEAREST)[0]).to(device).float()
    return (r[:, None] * c[None, :])[None, None]


class Segmenter:
    def __init__(self, checkpoint, device=None, cache_images=2):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ck = load_refunet(checkpoint, self.dev)
        self.roi = bool(getattr(self.model, "roi_ref", False))
        self.cache_images = cache_images
        self._feats = {}          # image hash -> {scale: features}; most recent last
        self._decoded = {}

    @classmethod
    def from_hub(cls, repo=HUB_REPO, out="/tmp/fgavg78.pth", **kw):
        """Rebuild the .pth from the published safetensors + config.json (as scripts/hf_ckpt_to_pth.py)."""
        if not os.path.exists(out):
            import json
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            tok = os.environ.get("HF_TOKEN") or None
            w = hf_hub_download(repo, "model.safetensors", token=tok)
            cfg = json.load(open(hf_hub_download(repo, "config.json", token=tok)))
            torch.save({"model": load_file(w), "args": cfg["args"], "epoch": cfg.get("epoch")}, out)
        return cls(out, **kw)

    def _image(self, data):
        key = hashlib.sha1(data).hexdigest()
        if key not in self._decoded:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("could not decode the image")
            self._decoded = {key: img}            # one decoded image is enough (the app sends one sheet at a time)
        if key not in self._feats:
            self._feats[key] = {}
            while len(self._feats) > self.cache_images:
                self._feats.pop(next(iter(self._feats)))
        return key, self._decoded[key]

    def run(self, data, box, scales=product.SCALES):
        t, t_all = {}, time.perf_counter()
        key, img = self._image(data)
        h0, w0 = img.shape[:2]
        x, y, bw, bh = (int(round(v)) for v in box)
        x, y = max(0, min(x, w0 - 1)), max(0, min(y, h0 - 1))
        bw, bh = max(1, min(bw, w0 - x)), max(1, min(bh, h0 - y))
        box = (x, y, bw, bh)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ref = _normalize_chw(cv2.resize(rgb[y:y + bh, x:x + bw], (224, 224), interpolation=cv2.INTER_LINEAR))[None].to(self.dev)
        feats, maps = self._feats[key], {}
        for z in scales:
            t0 = time.perf_counter()
            s = z / max(h0, w0); nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                im = _normalize_on(cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR), self.dev)[None]
                if z not in feats:
                    feats[z] = self.model.features(im)
                logit = self.model.forward_cached(feats[z], im.shape[-2:], ref,
                                                  _plane(box, h0, w0, nh, nw, self.dev) if self.roi else None, image=im)
                maps[z] = (logit.sigmoid().float()[0, 0] * 255).round().to(torch.uint8).cpu().numpy()
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            t[f"model_{z}"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        polys = product.product_polygons(maps, img, box)
        t["post"] = round(time.perf_counter() - t0, 3)
        t["total"] = round(time.perf_counter() - t_all, 3)
        polys = [{"outer": [[int(a), int(b)] for a, b in p["outer"]],
                  "holes": [[[int(a), int(b)] for a, b in h] for h in p["holes"]]} for p in polys]
        return {"polygons": polys, "vertices": product.nverts(polys), "size": [w0, h0], "box": list(box),
                "timing_s": t, "device": torch.cuda.get_device_name(0) if self.dev.type == "cuda" else "cpu"}
