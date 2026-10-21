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
COLOUR_SNAP = [True]
PROMPT = ("This is an architectural drawing. List EVERY distinct material pattern drawn on it (siding, masonry, stucco, "
          "roofing, floor finishes, hatches) -- one entry per distinct pattern, even if it appears in several places. "
          "The SAME kind of pattern in a different colour, tone, line spacing or direction is a DIFFERENT entry (e.g. tan "
          "lap siding and olive lap siding are two entries). Boxes must never touch callout text, leaders, labels or "
          "dimension strings. "
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


def colour_snap(img, Pb, win, tol=9.0, overlap=0.3):
    """Colour-filled sheets: families snapped to the fills. Each family's fill colour (Lab median of its confident,
    non-ink pixels); every pixel of a median-blurred copy (hatch lines wash out) takes the nearest family colour within
    `tol` (dark outline pixels take none, so regions stop at the drawn edges); a family gets every connected colour
    region its prediction overlaps by >= `overlap`. Boundaries follow the fills; same-colour areas the model missed come
    in. Returns the new winner map (-1 = none)."""
    h, w = img.shape[:2]
    k = max(3, int(w / 400) | 1)
    blur = cv2.medianBlur(img, k)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 60
    cols = []
    for j in range(len(Pb)):
        sel = (win == j) & (Pb[j] > 0.7) & ~ink
        cols.append(np.median(lab[sel], 0) if sel.sum() > 200 else None)
    ok = [j for j, c in enumerate(cols) if c is not None]
    if not ok:
        return win
    D = np.stack([np.linalg.norm(lab - cols[j], axis=-1) for j in ok])
    near = D.argmin(0)
    lab_map = np.where(D.min(0) < tol, np.array(ok)[near], -1)
    lab_map[ink] = -1
    out = np.full((h, w), -1, np.int32)
    for j in ok:
        n, cc = cv2.connectedComponents((lab_map == j).astype(np.uint8), connectivity=4)
        pred = (win == j) & (Pb[j] > 0.5)
        hit = np.bincount(cc.ravel(), weights=pred.ravel().astype(np.float64), minlength=n)
        size = np.bincount(cc.ravel(), minlength=n)
        good = (hit / np.maximum(size, 1) >= overlap)
        good[0] = False
        out[good[cc]] = j
    # ink lines inside a family's region belong to it (the hatch / outline pixels the fill wraps)
    for j in ok:
        m = (out == j).astype(np.uint8)
        cl = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k + 2, k + 2), np.uint8)).astype(bool)
        out[cl & (out == -1)] = j
    return out


def discover(model, img, grid=9, min_frac=0.004):
    """Families without Gemini's boxes: candidate reference boxes on a grid over the inked drawing, one segmentation
    each; a candidate is kept when its own box is inside its mask (mean probability there >= 0.6) and the mask is not a
    speck; kept masks are clustered greedily (IoU >= 0.5 with a cluster's mean map = same family). Returns (probability
    maps, the boxes behind each family)."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (g < np.percentile(g, 92) - 25).astype(np.float32)
    bs = int(0.055 * min(h, w) + 0.02 * max(h, w))
    dens = cv2.blur(ink, (bs, bs))
    cands = []
    for gy in range(grid):
        for gx in range(int(grid * w / h + 0.5) if w > h else grid):
            cy = int((gy + 0.5) * h / grid); cx = int((gx + 0.5) * w / max(1, int(grid * w / h + 0.5) if w > h else grid))
            if 0.04 < dens[cy, cx] < 0.6:                       # pattern-like ink density (not paper, not solid fill)
                cands.append((max(0, cx - bs // 2), max(0, cy - bs // 2), min(bs, w - 1), min(bs, h - 1)))
    if not cands:
        return [], []
    probs = segment(model, img, cands)
    keep = []
    for b, p in zip(cands, probs):
        x, y, bw, bh = b
        if p[y:y + bh, x:x + bw].mean() >= 0.6 and (p > 0.5).mean() >= min_frac:
            keep.append((b, p))
    keep.sort(key=lambda t: -(t[1] > 0.5).sum())
    clusters = []                                               # [sum of maps, count, boxes]
    for b, p in keep:
        m = p > 0.5
        for c in clusters:
            cm = (c[0] / c[1]) > 0.5
            if (m & cm).sum() / max(1, (m | cm).sum()) >= 0.5:
                c[0] += p; c[1] += 1; c[2].append(b); break
        else:
            clusters.append([p.copy(), 1, [b]])
    return [c[0] / c[1] for c in clusters], [c[2] for c in clusters]


def colour_split(img, Pb, win, min_de=14.0, min_share=0.15):
    """Colour-filled sheets: a family whose fills form two clearly different colours (two siding colours Gemini or the
    clustering lumped) becomes two families. k-means (k=2) on the Lab fill colours of its confident pixels; split when
    the centres are >= `min_de` apart and the smaller holds >= `min_share`. Returns (Pb, win, parent index per family)."""
    k = max(3, int(img.shape[1] / 400) | 1)
    lab = cv2.cvtColor(cv2.medianBlur(img, k), cv2.COLOR_BGR2LAB).astype(np.float32)
    newP, parent = [], []
    for j in range(len(Pb)):
        sel = (win == j) & (Pb[j] > 0.5)
        px = lab[sel]
        if len(px) > 2000:
            sub = px[np.random.default_rng(j).choice(len(px), min(20000, len(px)), replace=False)]
            _, lbl, cen = cv2.kmeans(sub, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3,
                                     cv2.KMEANS_PP_CENTERS)
            share = np.bincount(lbl.ravel(), minlength=2) / len(lbl)
            # full Lab: a chroma-only test (shadow-proof in principle) split 054 / 070 wrongly (r4: 0.726 -> 0.681)
            if np.linalg.norm(cen[0] - cen[1]) >= min_de and share.min() >= min_share:
                d = np.stack([np.linalg.norm(lab - c, axis=-1) for c in cen])
                d = np.stack([cv2.GaussianBlur(x, (0, 0), k) for x in d])
                a = d.argmin(0)
                for c in range(2):
                    newP.append(np.where(a == c, Pb[j], 0.0).astype(np.float32)); parent.append(j)
                continue
        newP.append(Pb[j]); parent.append(j)
    Pn = np.stack(newP)
    return Pn, Pn.argmax(0), parent


def oracle_families(pool, kinds_path, stem, hw):
    """Families from a labelled pool's hand labels: kind from the family -> kind map, up to 3 boxes sampled inside the
    family's largest pieces (the evaluator's sampler), in Gemini's 0-1000 [ymin, xmin, ymax, xmax] format."""
    import glob
    from refmask2former.dataset import render_instance_mask, sample_reference_box
    h, w = hw
    f = [x for x in sorted(glob.glob(f"{pool}/annotations/*.json")) if os.path.basename(x).split("_png")[0] == stem][0]
    si = sorted(glob.glob(f"{pool}/annotations/*.json")).index(f)
    km = json.load(open(kinds_path)).get(str(si), {})
    A = json.load(open(f))
    pieces = {}
    for an in A["annotations"]:
        pieces.setdefault(an["category_name"], []).append(render_instance_mask(an["segmentation"], h, w).astype(bool))
    fams = []
    for cat, ms in sorted(pieces.items()):
        rng = random.Random(len(fams) * 7919 + 13)
        boxes = []
        for m in sorted(ms, key=lambda m: -m.sum())[:3]:
            if m.sum() < 400:
                continue
            x, y, bw, bh = sample_reference_box(m, 128, 512, rng=rng)
            boxes.append([y * 1000 / h, x * 1000 / w, (y + bh) * 1000 / h, (x + bw) * 1000 / w])
        if boxes:
            fams.append({"kind": km.get("kinds", {}).get(cat, "lap"), "boxes": boxes, "true": cat})
    return {"shaded": bool(km.get("shaded")), "families": fams}


def gemini_probs(img, fam, model):
    """Families from Gemini's (or the oracle's) boxes: one map per family = max over its boxes. Returns
    (maps, kinds, boxes)."""
    h, w = img.shape[:2]
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
    P, kept, k = [], [], 0
    for i in keep:                                               # a family = its strongest reference at each pixel
        ps = probs[k:k + len(boxes[i])]
        # a box its own segmentation does not cover sat on text / a leader / another material: drop it (the family keeps
        # its best box unless even that is junk: faint patterns on photographed sheets score low everywhere)
        sc = [float(p[y:y + bh, x:x + bw].mean()) for p, (x, y, bw, bh) in zip(ps, boxes[i])]
        best = int(np.argmax(sc)) if sc else 0
        good = [p for j, (p, c) in enumerate(zip(ps, sc)) if c >= 0.4 or (j == best and c >= 0.15)]
        k += len(boxes[i])
        if good:
            P.append(np.max(good, axis=0)); kept.append(i)
    return P, [fams[i]["kind"] for i in kept], [boxes[i] for i in kept]


MISSING = ("The first image is an architectural drawing; the second is the same drawing with the material patterns found "
           "so far outlined (LEGEND). List the material patterns that are NOT inside any outline, and any outline that "
           "holds two clearly different patterns (a different colour, tone, spacing or direction counts) -- for those, "
           "the pattern the outline's name does not describe. Ignore text, windows, doors, trim, glass and blank paper. "
           f"For each give its kind from {json.dumps(KINDS)} and 1-3 small boxes [ymin, xmin, ymax, xmax] normalised to "
           "0-1000, each entirely INSIDE an area of that pattern, never touching text or leaders. Answer with ONE JSON "
           'object: {"families": [{"kind": k, "boxes": [[...], ...]}]} -- an empty list if nothing is missing.')


def refine_boxes(model, img, P, boxes, seed=0):
    """Second look with in-distribution references: for each family, up to 3 boxes drawn the way training drew them
    (sample_reference_box, 128-512 px, inside the family's largest confident pieces it wins), one more segmentation,
    and the family's map becomes the max over its old and new references. Gemini's boxes are loose and sized by eye."""
    from refmask2former.dataset import sample_reference_box
    if not len(P):
        return P, boxes
    Pa = np.stack(P)
    win = Pa.argmax(0)
    jobs = []
    for j, p in enumerate(P):
        m = ((win == j) & (p > 0.7)).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        order = [i for i in np.argsort(-st[1:, 4]) + 1 if st[i, 4] > 4000][:3]
        rng = random.Random(seed * 7919 + j)
        for i in order:
            x, y, bw, bh = sample_reference_box(cc == i, 128, 512, rng=rng)
            jobs.append((j, (x, y, bw, bh)))
    if not jobs:
        return P, boxes
    new = segment(model, img, [b for _, b in jobs])
    P = [p.copy() for p in P]
    boxes = [list(b) for b in boxes]
    for (j, b), q in zip(jobs, new):
        x, y, bw, bh = b
        if q[y:y + bh, x:x + bw].mean() >= 0.4:                 # the same self-check as Gemini's boxes
            P[j] = np.maximum(P[j], q); boxes[j].append(b)
    return P, boxes


def _box_lab(img, boxes):
    """Median Lab fill colour inside a family's reference boxes (median-blurred: hatch lines wash out). The boxes sit on
    the family's own colour; its probability map often spreads over a neighbour of the same texture."""
    if not boxes:
        return None
    lab = cv2.cvtColor(cv2.medianBlur(img, max(3, int(img.shape[1] / 400) | 1)), cv2.COLOR_BGR2LAB).astype(np.float32)
    px = np.concatenate([lab[y:y + h, x:x + w].reshape(-1, 3) for x, y, w, h in boxes])
    return np.median(px, 0) if len(px) else None


def _fill_lab(img, p):
    lab = cv2.cvtColor(cv2.medianBlur(img, max(3, int(img.shape[1] / 400) | 1)), cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = p > 0.7
    return np.median(lab[sel], 0) if sel.sum() > 200 else None


def merge_second(img, P, kinds, bxs, P2, kinds2, bxs2, shaded, min_de=12.0):
    """Second-pass families join a first-pass family of the same kind (Gemini re-listing a family it already had splits
    it in two), unless the sheet is colour-filled and their fills differ by >= `min_de` -- the two-colour case the
    second pass is for -- or whose mask overlaps a first-pass family's at IoU >= 0.65 whatever it is called (colours
    from the reference boxes, which sit on each family's own colour). Joining = max of the maps."""
    for p2, k2, b2 in zip(P2, kinds2, bxs2):
        same = [i for i, k in enumerate(kinds) if k == k2]
        target = None
        m2 = p2 > 0.5
        c2 = _box_lab(img, b2) if shaded else None
        for i, p in enumerate(P):                               # the same region under another name (shingle / asphalt)
            m1 = p > 0.5
            if (m1 & m2).sum() / max(1, (m1 | m2).sum()) >= 0.65:
                c1 = _box_lab(img, bxs[i]) if shaded else None
                if c1 is not None and c2 is not None and np.linalg.norm(c1 - c2) >= min_de:
                    continue                                    # a lumped two-colour family being split: keep it
                target = i; break
        if target is not None:
            P[target] = np.maximum(P[target], p2); bxs[target] = bxs[target] + b2
            continue
        for i in same:
            if not shaded:
                target = i; break
            c1, c2 = _box_lab(img, bxs[i]), _box_lab(img, b2)
            if c1 is None or c2 is None or np.linalg.norm(c1 - c2) < min_de:
                target = i; break
        if target is None:
            P.append(p2); kinds.append(k2); bxs.append(b2)
        else:
            P[target] = np.maximum(P[target], p2); bxs[target] = bxs[target] + b2
    return P, kinds, bxs


def ask_missing(client, types, img, P, kinds):
    from facsimile_kinds import overlay
    from PIL import Image
    fams = {f"{kinds[j]} ({j + 1})": p > 0.5 for j, p in enumerate(P)}
    ov, legend = overlay(img, fams)
    parts = []
    for im in (img, ov):
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); pil.thumbnail((2048, 2048))
        buf = io.BytesIO(); pil.save(buf, "JPEG", quality=88)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    r = client.models.generate_content(model="gemini-2.5-pro", contents=parts + [MISSING.replace("LEGEND", "; ".join(legend))],
                                       config=types.GenerateContentConfig(response_mime_type="application/json",
                                                                          temperature=0))
    return json.loads(r.text)


DISCOVER_KINDS = dict(KINDS, none="not a material pattern (text, glass, trim, doors, sky, dimension lines, blank wall)")


def name_families(client, types, img, P):
    """Gemini names each discovered family's kind from an outline overlay (or 'none')."""
    from facsimile_kinds import overlay
    from PIL import Image
    fams = {f"pattern{j + 1}": p > 0.5 for j, p in enumerate(P)}
    ov, legend = overlay(img, fams)
    parts = []
    for im in (img, ov):
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); pil.thumbnail((2048, 2048))
        buf = io.BytesIO(); pil.save(buf, "JPEG", quality=88)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    prompt = (f"The first image is an architectural drawing; the second is the same drawing with regions outlined: "
              f"{'; '.join(legend)}. For each outlined region name the material pattern that fills MOST of it, choosing "
              f"from {json.dumps(DISCOVER_KINDS)}. Answer with ONE JSON object: "
              '{"shaded": true if surfaces are colour-filled else false, "kinds": {"<label name>": "<kind key>"}}')
    r = client.models.generate_content(model="gemini-2.5-pro", contents=parts + [prompt],
                                       config=types.GenerateContentConfig(response_mime_type="application/json",
                                                                          temperature=0))
    return json.loads(r.text)


def finish(img, P, kinds, shaded, seed, boxes=None):
    """Winner map from the family maps (blurred argmax; on colour sheets split two-colour families, snap to the fills),
    repaint each family with the generator's texture for its kind, labels = the repainted masks."""
    h, w = img.shape[:2]
    if not len(P):
        return img.copy(), [], []
    P = np.stack(P)
    sig = max(1.0, w / 400)
    Pb = np.stack([cv2.GaussianBlur(p, (0, 0), sig) for p in P])  # competing families: smooth borders, not scribbles
    win = Pb.argmax(0)
    parent = list(range(len(Pb)))
    if shaded and COLOUR_SNAP[0]:
        Pb, win, parent = colour_split(img, Pb, win)
        snapped = colour_snap(img, Pb, win)
        # the fills only MOVE pixels between families; where they say nothing the model's winner stands
        win = np.where(snapped >= 0, snapped, np.where(Pb.max(0) > 0.5, win, -1))
        Pb = np.where(win[None] == np.arange(len(Pb))[:, None, None], 1.0, 0.0).astype(np.float32)
    if not shaded:
        # line-only sheets: blank paper is never a pattern (Gemini lists "flat" for sky / plain roofs and it floods the
        # empty sheet) -- areas with almost no ink in a ~3%-of-width window leave every family
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (g < np.percentile(g, 90) - 30).astype(np.float32)
        kw = max(15, int(w / 35))
        win = np.where(cv2.blur(ink, (kw, kw)) < 0.012, -1, win)
    out = img.copy()
    anns, info = [], []
    S = max(8.0, w / 60.0)
    for j in range(len(Pb)):
        kind = kinds[parent[j]]
        if kind not in KINDS or (kind == "flat" and not shaded):
            continue
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
        st = family_style(kind, img, mm, shaded, S, seed * 101 + j, cat)
        ys, xs = np.nonzero(mm)
        tmp = out.copy()
        G.draw_fill(tmp, sbox(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), st, S, w, h)
        out[mm] = tmp[mm]
        anns += [{"segmentation": sg, "category_name": cat} for sg in segs]
        info.append({"category": cat, "kind": kind, "pixels": int(mm.sum()),
                     "boxes_xywh": [list(b) for b in boxes[parent[j]]] if boxes else []})
    return out, anns, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None, help="comma list of image stems")
    ap.add_argument("--oracle", default=None, help="labelled pool: take the families (and 1-3 boxes inside each) from its "
                    "hand labels instead of Gemini -- the segmentation stage's ceiling (diagnostic)")
    ap.add_argument("--force", action="store_true", help="re-trace sheets already in --out")
    ap.add_argument("--families", default="gemini", choices=("gemini", "discover"),
                    help="gemini: Gemini lists families with boxes; discover: the model finds them from a grid of "
                         "self-consistent references and Gemini only names their kinds")
    ap.add_argument("--colour-snap", type=int, default=1, help="colour-filled sheets: snap families to the fills")
    ap.add_argument("--refine-boxes", type=int, default=0,
                    help="1 = re-segment each family from references sampled inside its own confident area (r4: 0.726 -> 0.680 in-label IoU, 054 collapses; off)")
    ap.add_argument("--second-pass", type=int, default=1,
                    help="1 = show Gemini what was found and ask for the patterns still missing (or lumped)")
    ap.add_argument("--oracle-kinds", default="image_generation/facsimile/r4_family_kinds.json")
    a = ap.parse_args()
    COLOUR_SNAP[0] = bool(a.colour_snap)
    client, types = None, None
    if not a.oracle:
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
        if not a.force and os.path.exists(f"{a.out}/annotations/trace_{p.stem}.json"):
            continue
        img = cv2.imread(str(p))
        if a.families == "discover" and not a.oracle:
            P, dboxes = discover(model, img)
            fam = allfam.get(p.stem, {})
            if fam.get("n_discovered") != len(P) or "kinds" not in fam:
                try:
                    fam = name_families(client, types, img, P) if P else {"shaded": False, "kinds": {}}
                except Exception as e:                           # noqa: BLE001 - rerun to retry
                    print(p.stem, "gemini error", str(e)[:120], flush=True); continue
                fam["n_discovered"] = len(P)
                allfam[p.stem] = fam
                json.dump(allfam, open(fpath, "w"), indent=1)
            kinds = [fam["kinds"].get(f"pattern{j + 1}", "none") for j in range(len(P))]
            twin, anns, info = finish(img, P, kinds, bool(fam.get("shaded")), k, dboxes)
        else:
            if a.oracle:
                allfam[p.stem] = oracle_families(a.oracle, a.oracle_kinds, p.stem, img.shape[:2])
            elif p.stem not in allfam or "families" not in allfam[p.stem]:
                try:
                    allfam[p.stem] = ask_gemini(client, types, img)
                except Exception as e:                           # noqa: BLE001 - rerun to retry
                    print(p.stem, "gemini error", str(e)[:120], flush=True); continue
                json.dump(allfam, open(fpath, "w"), indent=1)
            P, kinds, bxs = gemini_probs(img, allfam[p.stem], model)
            if a.second_pass and not a.oracle and P:
                if "missing" not in allfam[p.stem]:
                    try:
                        allfam[p.stem]["missing"] = ask_missing(client, types, img, P, kinds).get("families", [])
                    except Exception as e:                       # noqa: BLE001 - rerun to retry
                        print(p.stem, "gemini error (second pass)", str(e)[:120], flush=True); continue
                    json.dump(allfam, open(fpath, "w"), indent=1)
                P2, kinds2, bxs2 = gemini_probs(img, {"families": allfam[p.stem]["missing"]}, model)
                P, kinds, bxs = merge_second(img, list(P), kinds, bxs, P2, kinds2, bxs2,
                                             bool(allfam[p.stem].get("shaded")))
            if a.refine_boxes and not a.oracle:
                P, bxs = refine_boxes(model, img, list(P), bxs, seed=k)
            twin, anns, info = finish(img, P, kinds, bool(allfam[p.stem].get("shaded")), k, bxs)
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
