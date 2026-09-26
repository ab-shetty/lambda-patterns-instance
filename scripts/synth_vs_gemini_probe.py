#!/usr/bin/env python3
"""Can a classifier tell synthetic crops from Gemini crops?  If yes, the synth
does not look like Gemini yet -- and its most confidently "synthetic" crops
show where it gives itself away.

The bar is visual, not a statistic the generator is tuned to: fix what the top
crops show, regenerate, re-run, repeat until held-out AUC nears 0.5.

Cheap cues are equalised so the probe has to judge content:
  * framing/scale -- synth sheets are cropped to their ink bounding box and
    resized to Gemini's 3168 px long side (Gemini drawings fill the frame);
  * compression   -- synth sheets are JPEG-encoded at quality 75, which is what
    Roboflow's export gives Gemini (luma quant-table mean 29.0 == PIL q75);
  * location      -- crops are centred inside labelled regions (materials),
    not on blank paper, at the same side range (px at 3168) for both sources.

Features: frozen DINOv2 ViT-S/14 (CLS + mean patch), logistic regression,
5-fold cross-validation grouped by IMAGE so no sheet is in train and test.
CPU is fine: ~4k crops embed in a few minutes.

    python3 scripts/synth_vs_gemini_probe.py \
      --gemini data/roboflow/floz-gen-gemini-r2-raw data/roboflow/floz-gen-gemini-r3-raw \
               data/roboflow/floz-gen-gemini-r4-raw \
      --synth data/synthetic/v6d_probe300 --out data/probes/synth_vs_gemini_v6d
"""
import argparse
import glob
import io
import json
import os
import random

import cv2
import numpy as np
from PIL import Image

LONG = 3168
JPEG_Q = 75
OUT_PX = 224


def _poly_mask(segs, shape, scale=1.0, off=(0, 0)):
    m = np.zeros(shape, np.uint8)
    if segs and not isinstance(segs[0], list):
        segs = [segs]
    for s in segs:
        p = (np.array(s, np.float32).reshape(-1, 2) - off) * scale
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m


def load_gemini(dirs):
    """(image, [region masks]) per labelled Gemini image; remove polygons cut out."""
    for d in dirs:
        for coco in sorted(glob.glob(os.path.join(d, "*", "_annotations.coco.json"))):
            js = json.load(open(coco))
            cats = {c["id"]: c["name"] for c in js["categories"]}
            by_img = {}
            for a in js["annotations"]:
                by_img.setdefault(a["image_id"], []).append(a)
            for im in js["images"]:
                img = cv2.imread(os.path.join(os.path.dirname(coco), im["file_name"]))
                if img is None:
                    continue
                s = LONG / max(img.shape[:2])
                if abs(s - 1) > 1e-3:
                    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                anns = by_img.get(im["id"], [])
                hole = np.zeros(img.shape[:2], np.uint8)
                for a in anns:
                    if cats[a["category_id"]] == "remove":
                        hole |= _poly_mask(a["segmentation"], img.shape[:2], s)
                masks = [_poly_mask(a["segmentation"], img.shape[:2], s) & (1 - hole)
                         for a in anns if cats[a["category_id"]] not in ("remove", "pattern")]
                yield f"gem:{os.path.basename(d)}/{im['file_name'][:24]}", img, [m for m in masks if m.any()]


def load_synth(d, limit):
    """v6-format sheets, cropped to ink, resized to LONG, JPEG q75 round-tripped."""
    for f in sorted(glob.glob(os.path.join(d, "annotations", "*.json")))[:limit]:
        ann = json.load(open(f))
        img = cv2.imread(os.path.join(d, "images", ann["image"]["file_name"]))
        if img is None:
            continue
        paper = np.median(img.reshape(-1, 3), 0)
        ink = np.abs(img.astype(np.int16) - paper).sum(-1) > 40
        ys, xs = np.nonzero(ink)
        if len(xs) < 100:
            continue
        pad = int(0.03 * max(xs.ptp(), ys.ptp()))
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(img.shape[1], xs.max() + pad), min(img.shape[0], ys.max() + pad)
        img = img[y0:y1, x0:x1]
        s = LONG / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        buf = io.BytesIO()
        Image.fromarray(img[..., ::-1]).save(buf, "JPEG", quality=JPEG_Q)
        img = np.array(Image.open(buf))[..., ::-1].copy()
        masks = []
        for a in ann["annotations"]:
            m = _poly_mask(a["segmentation"], img.shape[:2], s, off=(x0, y0))
            if m.any():
                masks.append(m)
        yield f"syn:{os.path.basename(f)[:-5]}", img, masks


def sample_crops(img, masks, k, rng, side=(256, 768)):
    """k crops centred on labelled pixels (regions drawn uniformly), resized to OUT_PX."""
    out = []
    H, W = img.shape[:2]
    for _ in range(k * 3):
        if len(out) >= k or not masks:
            break
        m = masks[rng.randrange(len(masks))]
        ys, xs = np.nonzero(m[::4, ::4])
        if not len(xs):
            continue
        i = rng.randrange(len(xs))
        cy, cx = ys[i] * 4, xs[i] * 4
        sd = min(rng.randint(*side), H, W)
        x0 = int(np.clip(cx - sd // 2, 0, W - sd))
        y0 = int(np.clip(cy - sd // 2, 0, H - sd))
        if m[y0:y0 + sd, x0:x0 + sd].mean() < 0.35:     # mostly the region, not its surround
            continue
        c = cv2.resize(img[y0:y0 + sd, x0:x0 + sd], (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
        out.append((c, (x0, y0, sd)))
    return out


def embed(crops, bs=64):
    import timm
    import torch
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0, img_size=OUT_PX).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    with torch.inference_mode():
        for i in range(0, len(crops), bs):
            x = torch.from_numpy(np.stack(crops[i:i + bs])[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255
            t = model.forward_features((x - mean) / std)
            feats.append(torch.cat([t[:, 0], t[:, 1:].mean(1)], 1).numpy())
            print(f"  embedded {min(i + bs, len(crops))}/{len(crops)}", flush=True)
    return np.concatenate(feats)


def contact(crops, labels, path, ncol=8):
    tiles = []
    for c, lab in zip(crops, labels):
        t = c.copy()
        cv2.rectangle(t, (0, OUT_PX - 18), (OUT_PX, OUT_PX), (255, 255, 255), -1)
        cv2.putText(t, lab[:34], (3, OUT_PX - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(128, 128, 128)))
    while len(tiles) % ncol:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemini", nargs="+", required=True)
    ap.add_argument("--synth", required=True, help="v6-format dir (images/, annotations/)")
    ap.add_argument("--synth-limit", type=int, default=300)
    ap.add_argument("--crops-gemini", type=int, default=14, help="crops per Gemini image")
    ap.add_argument("--crops-synth", type=int, default=8, help="crops per synth sheet")
    ap.add_argument("--top", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    crops, y, groups, where = [], [], [], []
    for lab, src, k in ((0, load_gemini(args.gemini), args.crops_gemini),
                        (1, load_synth(args.synth, args.synth_limit), args.crops_synth)):
        for name, img, masks in src:
            for c, box in sample_crops(img, masks, k, rng):
                crops.append(c); y.append(lab); groups.append(name); where.append(box)
    y = np.array(y)
    print(f"crops: gemini {int((y == 0).sum())} from {len({g for g, t in zip(groups, y) if t == 0})} images, "
          f"synth {int(y.sum())} from {len({g for g, t in zip(groups, y) if t == 1})} sheets", flush=True)
    X = embed(crops)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    acc = float(((oof > 0.5) == y).mean())
    # per-image: mean crop score, AUC over images
    imgs = sorted(set(groups))
    g_score = {g: float(np.mean(oof[[i for i, gg in enumerate(groups) if gg == g]])) for g in imgs}
    img_auc = roc_auc_score([g.startswith("syn") for g in imgs], [g_score[g] for g in imgs])
    print(f"held-out crop AUC {auc:.3f}  acc {acc:.3f}  |  image AUC {img_auc:.3f}")

    syn = np.nonzero(y == 1)[0]
    gem = np.nonzero(y == 0)[0]
    lab = lambda i: f"{groups[i].split(':')[1][-14:]} p={oof[i]:.2f}"
    top_syn = syn[np.argsort(-oof[syn])][:args.top]
    passing = syn[np.argsort(oof[syn])][:args.top]
    gem_synthlike = gem[np.argsort(-oof[gem])][:args.top // 2]
    gem_typical = gem[np.argsort(oof[gem])][:args.top // 2]
    contact([crops[i] for i in top_syn], [lab(i) for i in top_syn], f"{args.out}/synth_most_obvious.jpg")
    contact([crops[i] for i in passing], [lab(i) for i in passing], f"{args.out}/synth_most_gemini_like.jpg")
    contact([crops[i] for i in gem_typical], [lab(i) for i in gem_typical], f"{args.out}/gemini_most_typical.jpg")
    contact([crops[i] for i in gem_synthlike], [lab(i) for i in gem_synthlike], f"{args.out}/gemini_most_synth_like.jpg")
    json.dump({"synth": args.synth, "crop_auc": auc, "crop_acc": acc, "image_auc": img_auc,
               "n_gemini_crops": int(len(gem)), "n_synth_crops": int(len(syn)),
               "synth_frac_p_below_0.5": float((oof[syn] < 0.5).mean()),
               "per_image": g_score,
               "crops": [{"src": groups[i], "box": where[i], "p_synth": float(oof[i])} for i in range(len(y))]},
              open(f"{args.out}/summary.json", "w"), indent=1)
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
