#!/usr/bin/env python3
"""Model-in-the-loop post-processing (2026-10-22): ask the model again from inside its own answer.

For each question, A = the answer to the user's box (threshold 0.35 at inference resolution). K points spread over A
(farthest-point sampling, away from the user's box, each the centre of a box that fits inside A, no larger than the
user's box) become back-queries. A back-answer B_k is RECIPROCAL if it covers >= half of the user's box: the model,
asked from that point, names the user's material. Variants (all upsampled nearest to native, as the evaluator):
  base    A
  cut     A minus every non-reciprocal B_k (that point sits on another material, and B_k is that material), the user's
          box itself never cut
  ens     mean of the user's map and the reciprocal back-maps, > 0.35 (several references of the same family)
  ens_cut ens, then the cut
Settings fixed before any scoring: K=6, thresholds 0.35 / 0.5, minimum back-box side 24 inference px.
    PYTHONPATH=. python3 scripts/reciprocity_probe.py CKPT SPLIT OUT.json [--shard i/n]   (split: gemini30 hf14 hf14fix2 val)
"""
import argparse
import io
import json
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import common
from evaluate_refunet_selection import _normalize_on, _prepare_sheet, load_refunet
from refmask2former.dataset import _normalize_chw

K, T, RECIP, MIN_SIDE = 6, 0.35, 0.5, 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--size", type=int, default=4096)
    a = ap.parse_args()
    si, sn = (int(v) for v in a.shard.split("/"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_refunet(a.checkpoint, device)
    jobs = common.jobs(a.split)[si::sn]
    rows = []
    with torch.inference_mode():
        for idx, rec, ov, fixes, _ in jobs:
            sh = _prepare_sheet((idx, rec, a.size, 224, False, ov, fixes, True))
            h0, w0, nh, nw = sh["h0"], sh["w0"], sh["nh"], sh["nw"]
            data = rec["image"] if "image" in rec else Path(rec["image_path"]).read_bytes()
            data = data["bytes"] if isinstance(data, dict) else data
            image0 = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            img_t = _normalize_on(sh["image"], device).unsqueeze(0)
            feats = model.features(img_t)

            def ask(boxes):
                out = []
                for c0 in range(0, len(boxes), 8):
                    chunk = boxes[c0:c0 + 8]
                    refs, planes = [], []
                    for x, y, w, h in chunk:
                        refs.append(_normalize_chw(cv2.resize(image0[y:y + h, x:x + w], (224, 224),
                                                              interpolation=cv2.INTER_LINEAR)))
                        rows_ = np.zeros((h0, 1), np.uint8); rows_[y:y + h] = 1
                        cols = np.zeros((1, w0), np.uint8); cols[0, x:x + w] = 1
                        r = cv2.resize(rows_, (1, nh), interpolation=cv2.INTER_NEAREST)[:, 0]
                        c = cv2.resize(cols, (nw, 1), interpolation=cv2.INTER_NEAREST)[0]
                        planes.append(torch.from_numpy(r).float()[:, None] * torch.from_numpy(c).float()[None, :])
                    refs = torch.stack(refs).to(device)
                    planes = torch.stack(planes)[:, None].to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logit = model.forward_cached(feats, img_t.shape[-2:], refs, planes)
                    out.extend(logit.sigmoid().float()[:, 0].cpu().numpy())
                return out

            for ref_idx, cat, (x, y, w, h), _, _ in sh["questions"]:
                bits, _ = sh["targets"][cat]
                tgt = np.unpackbits(bits, count=h0 * w0).reshape(h0, w0).astype(bool)
                P0 = ask([(x, y, w, h)])[0]
                A = P0 > T
                s = nw / w0
                ux0, uy0, ux1, uy1 = int(x * s), int(y * s), max(int(x * s) + 1, int((x + w) * s)), max(int(y * s) + 1, int((y + h) * s))
                ubox = np.zeros_like(A); ubox[uy0:uy1, ux0:ux1] = True
                half = max(MIN_SIDE // 2, min(ux1 - ux0, uy1 - uy0) // 2)
                dt = cv2.distanceTransform(A.astype(np.uint8), cv2.DIST_L2, 5)
                far = cv2.distanceTransform((~ubox).astype(np.uint8), cv2.DIST_L2, 5)
                cand = np.argwhere((dt >= MIN_SIDE / 2) & (far > 2 * half))
                rng = np.random.default_rng(idx * 1000 + ref_idx)
                if len(cand) > 4000:
                    cand = cand[rng.choice(len(cand), 4000, replace=False)]
                pts = []
                if len(cand):
                    dmin = np.hypot(cand[:, 0] - (uy0 + uy1) / 2, cand[:, 1] - (ux0 + ux1) / 2)
                    for _ in range(K):
                        j = int(np.argmax(dmin))
                        if dmin[j] <= 0:
                            break
                        pts.append(cand[j])
                        dmin = np.minimum(dmin, np.hypot(cand[:, 0] - cand[j][0], cand[:, 1] - cand[j][1]))
                boxes = []
                for py, px in pts:
                    r = int(min(dt[py, px] / np.sqrt(2), half))
                    r = max(r, MIN_SIDE // 2)
                    bx0, by0 = max(0, int((px - r) / s)), max(0, int((py - r) / s))
                    bx1, by1 = min(w0, int((px + r) / s)), min(h0, int((py + r) / s))
                    if bx1 - bx0 >= 4 and by1 - by0 >= 4:
                        boxes.append((bx0, by0, bx1 - bx0, by1 - by0))
                backs = ask(boxes) if boxes else []
                cov = [float((b > T)[ubox].mean()) for b in backs]
                recip = [c >= RECIP for c in cov]
                cut = np.zeros_like(A)
                for b, ok in zip(backs, recip):
                    if not ok:
                        cut |= b > T
                cut &= ~ubox
                ens_p = (P0 + sum(b for b, ok in zip(backs, recip) if ok)) / (1 + sum(recip))
                v = {"base": A, "cut": A & ~cut, "ens": ens_p > T, "ens_cut": (ens_p > T) & ~cut}
                row = {"sheet": idx, "q": ref_idx, "n_back": len(backs), "n_recip": int(sum(recip)), "cov": [round(c, 3) for c in cov],
                       "back_in_tgt": [round(float(tgt[int(b[1] + b[3] / 2), int(b[0] + b[2] / 2)]), 0) for b in boxes]}
                for k, m in v.items():
                    m0 = cv2.resize(m.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
                    row[k] = float((m0 & tgt).sum() / max(1, (m0 | tgt).sum()))
                rows.append(row)
            print(f"sheet {idx}: {len(rows)} q so far", flush=True)
    json.dump(rows, open(a.out, "w"))
    print(a.split, len(rows), "q", " ".join(f"{k} {np.mean([r[k] for r in rows]):.4f}" for k in ("base", "cut", "ens", "ens_cut")))


if __name__ == "__main__":
    main()
