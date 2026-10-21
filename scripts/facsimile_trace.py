#!/usr/bin/env python3
"""Trace twins of UNLABELLED Gemini sheets (2026-10-21; the user's goal: hand over a new Gemini image, get a near-same
synthetic sheet with exact labels).

1. Gemini names every distinct material pattern on the sheet with a generator kind and one small box inside it.
2. The best segmentation model (xblend-kq30-kms30, reference-conditioned) segments each pattern from its box: one
   mask per family; overlaps go to the higher probability, specks are dropped.
3. facsimile_relabel.family_style repaints the inside of each mask with the generator's texture for that kind; the
   sheet's own geometry, linework, openings and text stay.
4. The labels ARE the repainted masks (one instance per connected piece, holes kept), so they are exact by
   construction: a segmentation miss makes the twin differ from the original, never a twin label wrong.

    GEMINI_API_KEY=... python3 scripts/facsimile_trace.py --images DIR --out data/synthetic/trace_r4 [--names a,b]
writes images/, annotations/ (the pool format), families.json (Gemini's answer + mask areas), compare.jpg.
"""
import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import box as sbox

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import generate_synthetic_v6 as G  # noqa: E402
from evaluate_refunet_selection import load_refunet  # noqa: E402
from facsimile_kinds import KINDS  # noqa: E402
from facsimile_relabel import family_style  # noqa: E402
from refmask2former.dataset import _normalize_chw  # noqa: E402

CKPT = "data/hf_ckpt/xblend-kq30-kms30.pth"
PROMPT = ("This is an architectural drawing. List EVERY distinct material pattern drawn on it (siding, masonry, stucco, "
          "roofing, floor finishes, hatches) -- one entry per distinct pattern, even if it appears in several places. "
          f"For each give its kind from {json.dumps(KINDS)} and 1-3 small boxes [ymin, xmin, ymax, xmax] normalised to "
          "0-1000, each in a DIFFERENT area where the pattern appears (all of them if it is in several places, up to 3), "
          "each lying entirely INSIDE that area of the pattern (well clear of its edges, windows, text and "
          "lines that are not part of the pattern), roughly 4-10% of the drawing's width on a side. Answer with ONE "
          'JSON object: {"shaded": true if surfaces are colour-filled else false, "families": [{"kind": k, "boxes": [[...], ...]}]}')


def ask_gemini(client, types, img):
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); pil.thumbnail((2048, 2048))
    buf = io.BytesIO(); pil.save(buf, "JPEG", quality=90)
    r = client.models.generate_content(
        model="gemini-2.5-pro", contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
    return json.loads(r.text)


@torch.inference_mode()
def segment(model, img, boxes_px, max_size=2048, ref_size=224):
    """Probability map (native resolution) per reference box (x, y, w, h), as the evaluator runs the model."""
    h0, w0 = img.shape[:2]
    s = max_size / max(h0, w0)
    nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    it = _normalize_chw(small).unsqueeze(0)
    feats = model.features(it)
    out = []
    for (x, y, w, h) in boxes_px:
        crop = cv2.resize(rgb[y:y + h, x:x + w], (ref_size, ref_size), interpolation=cv2.INTER_LINEAR)
        rows = np.zeros((h0, 1), np.uint8); rows[y:y + h] = 1
        cols = np.zeros((1, w0), np.uint8); cols[0, x:x + w] = 1
        r_ = cv2.resize(rows, (1, nh), interpolation=cv2.INTER_NEAREST)[:, 0]
        c_ = cv2.resize(cols, (nw, 1), interpolation=cv2.INTER_NEAREST)[0]
        plane = (torch.from_numpy(r_).float()[:, None] * torch.from_numpy(c_).float()[None, :])[None, None]
        logit = model.forward_cached(feats, it.shape[-2:], _normalize_chw(crop).unsqueeze(0), plane)
        p = logit.sigmoid()[0, 0].numpy()
        out.append(cv2.resize(p, (w0, h0), interpolation=cv2.INTER_LINEAR))
    return out


def instances(mask, min_px):
    """Connected pieces of a family mask as pool annotations (outer ring + holes, flat xy lists)."""
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    for i, c in enumerate(cs):
        if hier[0][i][3] != -1 or cv2.contourArea(c) < min_px or len(c) < 3:
            continue
        seg = [c.reshape(-1).astype(float).tolist()]
        j = hier[0][i][2]
        while j != -1:
            if len(cs[j]) >= 3 and cv2.contourArea(cs[j]) > 16:
                seg.append(cs[j].reshape(-1).astype(float).tolist())
            j = hier[0][j][0]
        out.append(seg)
    return out


def trace(img, fam, model, seed):
    h, w = img.shape[:2]
    shaded = bool(fam.get("shaded"))
    fams = [f for f in fam.get("families", []) if f.get("kind") in KINDS]
    boxes = []                                                   # per family: its reference boxes (x, y, w, h)
    for f in fams:
        bl = []
        for b in (f.get("boxes") or ([f["box"]] if "box" in f else []))[:3]:
            if not (isinstance(b, list) and len(b) == 4):
                continue
            y0, x0, y1, x1 = [float(v) for v in b]
            if max(y0, x0, y1, x1) > 1000 or y1 <= y0 or x1 <= x0:
                continue
            cy, cx, bh, bw = (y0 + y1) / 2 * h / 1000, (x0 + x1) / 2 * w / 1000, (y1 - y0) * h / 1000, (x1 - x0) * w / 1000
            bh, bw = max(24, 0.7 * bh), max(24, 0.7 * bw)        # shrink toward the centre: Gemini's boxes run loose
            bl.append((int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(bw, w - 1)), int(min(bh, h - 1))))
        boxes.append(bl)
    keep = [i for i, bl in enumerate(boxes) if bl]
    flat = [b for i in keep for b in boxes[i]]
    probs = segment(model, img, flat) if flat else []
    if not probs:
        return img.copy(), [], []
    P, k = [], 0
    for i in keep:                                               # a family = its strongest reference at each pixel
        P.append(np.max(probs[k:k + len(boxes[i])], axis=0)); k += len(boxes[i])
    P = np.stack(P)
    sig = max(1.0, w / 400)
    Pb = np.stack([cv2.GaussianBlur(p, (0, 0), sig) for p in P])  # competing families: smooth borders, not scribbles
    win = Pb.argmax(0)
    out = img.copy()
    anns, info = [], []
    S = max(8.0, w / 60.0)
    for j, i in enumerate(keep):
        m = (win == j) & (Pb[j] > 0.5)
        kk = max(3, int(w / 500) | 1)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((kk, kk), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8)).astype(bool)
        segs = instances(m, min_px=0.0005 * h * w)
        mm = np.zeros((h, w), np.uint8)
        for sg in segs:                                           # the label exactly as written
            cv2.fillPoly(mm, [np.array(sg[0]).reshape(-1, 2).astype(np.int32)], 1)
            for hole in sg[1:]:
                cv2.fillPoly(mm, [np.array(hole).reshape(-1, 2).astype(np.int32)], 0)
        mm = mm.astype(bool)
        if not mm.any():
            continue
        cat = f"pattern{len(info) + 1}"
        st = family_style(fams[i]["kind"], img, mm, shaded, S, seed * 101 + j, cat)
        ys, xs = np.nonzero(mm)
        tmp = out.copy()
        G.draw_fill(tmp, sbox(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), st, S, w, h)
        out[mm] = tmp[mm]
        anns += [{"segmentation": sg, "category_name": cat} for sg in segs]
        info.append({"category": cat, "kind": fams[i]["kind"], "boxes_xywh": [list(b) for b in boxes[i]],
                     "pixels": int(mm.sum())})
    return out, anns, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None, help="comma list of image stems")
    a = ap.parse_args()
    from google import genai
    from google.genai import types
    client = genai.Client()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model, _ = load_refunet(CKPT, torch.device("cpu"))
    os.makedirs(f"{a.out}/images", exist_ok=True); os.makedirs(f"{a.out}/annotations", exist_ok=True)
    fpath = f"{a.out}/families.json"
    allfam = json.load(open(fpath)) if os.path.exists(fpath) else {}
    paths = sorted(Path(a.images).glob("*.png"))
    if a.names:
        want = set(a.names.split(",")); paths = [p for p in paths if p.stem in want]
    rows = []
    for k, p in enumerate(paths):
        img = cv2.imread(str(p))
        if p.stem not in allfam or "families" not in allfam[p.stem]:
            try:
                allfam[p.stem] = ask_gemini(client, types, img)
            except Exception as e:                               # noqa: BLE001 - rerun to retry
                print(p.stem, "gemini error", str(e)[:120], flush=True); continue
            json.dump(allfam, open(fpath, "w"), indent=1)
        twin, anns, info = trace(img, allfam[p.stem], model, seed=k)
        allfam[p.stem]["traced"] = info
        name = f"trace_{p.stem}.png"
        cv2.imwrite(f"{a.out}/images/{name}", twin)
        h, w = img.shape[:2]
        json.dump({"image": {"file_name": name, "width": w, "height": h}, "mode": "freeform", "annotations": anns},
                  open(f"{a.out}/annotations/trace_{p.stem}.json", "w"))
        json.dump(allfam, open(fpath, "w"), indent=1)
        print(f"{p.stem}: {len(info)} families, {len(anns)} instances", flush=True)
        # side by side: original | twin | twin labels
        lab = twin.copy()
        for j, inf in enumerate(info):
            col = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)][j % 6]
            for an in [x for x in anns if x["category_name"] == inf["category"]]:
                cv2.polylines(lab, [np.array(an["segmentation"][0]).reshape(-1, 2).astype(np.int32)], True, col[::-1],
                              max(3, w // 400))
        W = 620
        row = np.hstack([cv2.resize(im, (W, int(h * W / w))) for im in (img, twin, lab)])
        rows.append(row)
    if rows:
        Wm = max(r.shape[1] for r in rows)
        cv2.imwrite(f"{a.out}/compare.jpg", np.vstack([cv2.copyMakeBorder(r, 0, 6, 0, Wm - r.shape[1], cv2.BORDER_CONSTANT,
                                                                              value=(200, 200, 200)) for r in rows]))


if __name__ == "__main__":
    main()
