#!/usr/bin/env python3
"""Per-query latency of the inference pipeline (2026-10-22; product budget ~2 s per query).
Times, on one GPU, warm, for real Gemini30 sheets: (a) the image backbone at 4096 (once per sheet; cacheable across a
user's queries), (b) one question's reference encoder + decoder at 4096, (c) the 2048 pass for multi-scale (backbone +
question), (d) CPU post-processing (linear upsample, polygons, colour cut) on the native-resolution map.
    PYTHONPATH=. python3 scripts/latency_probe.py CKPT [N_SHEETS]"""
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import common
import stack
from evaluate_refunet_selection import _normalize_on, _prepare_sheet, load_refunet
from refmask2former.dataset import _normalize_chw


def sync():
    torch.cuda.synchronize()


def main():
    ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    dev = torch.device("cuda")
    model, _ = load_refunet(ck, dev)
    T = {k: [] for k in ("backbone4096", "question4096", "pass2048", "postproc", "total_single", "total_ms")}
    with torch.inference_mode():
        for idx, rec, ov, fixes, _ in common.jobs("gemini30")[:n + 1]:
            sh = {sz: _prepare_sheet((idx, rec, sz, 224, False, ov, fixes, True)) for sz in (4096, 2048)}
            img0 = cv2.imread(rec["image_path"])
            ref_idx, cat, box, crop, plane = sh[4096]["questions"][0]
            out = {}
            for sz in (4096, 2048):
                s = sh[sz]
                q = s["questions"][0]
                sync(); t0 = time.perf_counter()
                img_t = _normalize_on(s["image"], dev).unsqueeze(0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = model.features(img_t)
                sync(); t1 = time.perf_counter()
                ref = _normalize_chw(q[3]).unsqueeze(0).to(dev)
                pl = (torch.from_numpy(q[4][0]).float()[:, None] * torch.from_numpy(q[4][1]).float()[None, :])[None, None].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model.forward_cached(f, img_t.shape[-2:], ref, pl).sigmoid().float()[0, 0]
                pu8 = (p * 255).round().to(torch.uint8).cpu().numpy()
                sync(); t2 = time.perf_counter()
                out[sz] = (t1 - t0, t2 - t1, pu8)
            t0 = time.perf_counter()
            h0, w0 = img0.shape[:2]
            P = cv2.resize(out[4096][2], (w0, h0), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            P2 = cv2.resize(out[2048][2], (w0, h0), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            Pm = 0.7 * P + 0.3 * P2
            s4 = out[4096][2].shape[1] / w0
            m = stack.to_mask(Pm, s4, img0, box)
            post = time.perf_counter() - t0
            if idx == common.jobs("gemini30")[0][0]:
                continue                     # warm-up sheet
            T["backbone4096"].append(out[4096][0]); T["question4096"].append(out[4096][1])
            T["pass2048"].append(out[2048][0] + out[2048][1]); T["postproc"].append(post)
            T["total_single"].append(out[4096][0] + out[4096][1] + post)
            T["total_ms"].append(out[4096][0] + out[4096][1] + out[2048][0] + out[2048][1] + post)
            print(f"sheet {idx} {w0}x{h0}: " + " ".join(f"{k} {v[-1]:.2f}s" for k, v in T.items()), flush=True)
    print("median over", len(T["postproc"]), "sheets:", " ".join(f"{k} {np.median(v):.2f}s" for k, v in T.items()))


if __name__ == "__main__":
    main()
