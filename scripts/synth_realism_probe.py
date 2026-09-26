#!/usr/bin/env python3
"""Can a classifier tell synthetic crops from reference crops?  If yes, the
synth does not look like the reference yet -- and its most confidently
"synthetic" crops show where it gives itself away.

Reference: the 14 VALIDATION plans (default). Val 14 vs HF14 is 0.423 on this
probe (chance p95 0.71), so val stands in for the test look without touching
HF14. Gemini (0.95-0.997 vs eval) and the scraped real pool (0.965 vs eval,
resolution matched) are both a different look. Chance level with 14 reference
images is high (~0.7): read the crops, not the AUC, until the AUC is low.

The bar is visual, not a statistic the generator is tuned to: fix what the top
crops show, regenerate, re-run, repeat until held-out AUC nears 0.5.

Cheap cues are equalised so the probe has to judge content:
  * framing/scale -- synth sheets are cropped to their ink bounding box; every
    image is resized to --long-side (640 = the real pool's native size);
  * compression   -- anything resized is re-encoded at JPEG quality 75, which
    is what Roboflow exports (luma quant-table mean 29.0 == PIL q75);
  * location      -- crops are centred inside labelled regions (materials),
    not on blank paper, at the same fraction of the long side for all sources.

Features: frozen DINOv2 ViT-S/14 (CLS + mean patch), logistic regression,
5-fold cross-validation grouped by IMAGE so no sheet is in train and test.
CPU is fine: ~4k crops embed in a few minutes.

    python3 scripts/synth_realism_probe.py --synth data/synthetic/v6d_probe300 \
      --out data/probes/realism_v6d            # reference: val 14 at 3168 px
    # or any Roboflow COCO export:  --reference <dir> --long-side 640
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
VAL14 = [4, 5, 6, 8, 9, 10, 13, 15, 17, 19, 20, 21, 22, 26]
EVAL_PARQUET = "data/hf_eval/real-world-test/test-00000.parquet"
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


def _jpeg(img):
    buf = io.BytesIO()
    Image.fromarray(img[..., ::-1]).save(buf, "JPEG", quality=JPEG_Q)
    return np.array(Image.open(buf))[..., ::-1].copy()


def load_coco(dirs):
    """(image, [region masks]) per labelled image of Roboflow COCO exports; remove polygons cut out."""
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
                    img = _jpeg(cv2.resize(img, None, fx=s, fy=s,
                                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
                anns = by_img.get(im["id"], [])
                hole = np.zeros(img.shape[:2], np.uint8)
                for a in anns:
                    if cats[a["category_id"]] == "remove":
                        hole |= _poly_mask(a["segmentation"], img.shape[:2], s)
                masks = [_poly_mask(a["segmentation"], img.shape[:2], s) & (1 - hole)
                         for a in anns if cats[a["category_id"]] not in ("remove", "pattern")]
                yield f"ref:{os.path.basename(d)}/{im['file_name'][:24]}", img, [m for m in masks if m.any()]


def load_eval(indices, parquet=EVAL_PARQUET):
    """The HF real-world-test images at `indices` (validation only -- never HF14)."""
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet, columns=["image", "annotations"]).to_pylist()
    for i in indices:
        r = rows[i]
        b = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
        img = np.array(Image.open(io.BytesIO(b)).convert("RGB"))[..., ::-1]
        s = LONG / max(img.shape[:2])
        img = _jpeg(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
        masks = [_poly_mask(a["segmentation"], img.shape[:2], s) for a in json.loads(r["annotations"])
                 if a.get("role") != "remove"]
        yield f"ref:eval{i:02d}", img, [m for m in masks if m.any()]


SYNTH_META = {}   # name -> (mode, appearance, [family per mask])


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
        img = _jpeg(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
        masks, fams = [], []
        for a in ann["annotations"]:
            m = _poly_mask(a["segmentation"], img.shape[:2], s, off=(x0, y0))
            if m.any():
                masks.append(m); fams.append(a.get("family", "?"))
        name = f"syn:{os.path.basename(f)[:-5]}"
        SYNTH_META[name] = (ann.get("mode", "?"), ann.get("appearance", "?"), fams)
        yield name, img, masks


def sample_crops(img, masks, k, rng, side_frac=(0.08, 0.24)):
    """k crops centred on labelled pixels (regions drawn uniformly), side a fraction
    of the image's long side (0.08-0.24 = 256-768 px at 3168), resized to OUT_PX."""
    out = []
    H, W = img.shape[:2]
    side = (max(24, int(side_frac[0] * max(H, W))), int(side_frac[1] * max(H, W)))
    for _ in range(k * 3):
        if len(out) >= k or not masks:
            break
        mi = rng.randrange(len(masks))
        m = masks[mi]
        ys, xs = np.nonzero(m[::4, ::4])
        if not len(xs):
            continue
        i = rng.randrange(len(xs))
        cy, cx = ys[i] * 4 + 2, xs[i] * 4 + 2
        sd = min(rng.randint(*side), H, W)
        x0 = int(np.clip(cx - sd // 2, 0, W - sd))
        y0 = int(np.clip(cy - sd // 2, 0, H - sd))
        if m[y0:y0 + sd, x0:x0 + sd].mean() < 0.35:     # mostly the region, not its surround
            continue
        c = cv2.resize(img[y0:y0 + sd, x0:x0 + sd], (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
        out.append((c, (x0, y0, sd), mi))
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
    global LONG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", nargs="+", default=None,
                    help="Roboflow COCO export dir(s) instead of the validation 14")
    ap.add_argument("--long-side", type=int, default=LONG, help="every image is resized to this")
    ap.add_argument("--synth", required=True, help="v6-format dir (images/, annotations/)")
    ap.add_argument("--synth-limit", type=int, default=320)
    ap.add_argument("--seeds", type=int, default=3, help="AUC is the mean over this many subsample/fold seeds")
    ap.add_argument("--top", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    LONG = args.long_side
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    os.makedirs(args.out, exist_ok=True)

    # Fixed protocol (do not tune to lower the AUC): embed a pool of crops once --
    # 150 per reference image (cached), 12 per synth sheet -- then for each seed
    # subsample 60 per reference image and 8 per sheet, assign whole images to
    # 5 random folds, and score held-out crops. With val 14 fixed, one run's sd
    # is ~0.01 and a between-pool difference needs >~0.02 (mean of 3 seeds).
    key = "val14" if not args.reference else "_".join(os.path.basename(r.rstrip("/")) for r in args.reference)
    cache = f"data/probes/_refcache_{key}_{LONG}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        rX, rg, rcrops = z["X"], z["g"], list(z["crops"])
    else:
        rng = random.Random(0); rcrops, rg = [], []
        for name, img, masks in (load_coco(args.reference) if args.reference else load_eval(VAL14)):
            for c, _, _ in sample_crops(img, masks, 150, rng):
                rcrops.append(c); rg.append(name)
        rX, rg = embed(rcrops), np.array(rg)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, X=rX, g=rg, crops=np.stack(rcrops))
    rng = random.Random(0); scrops, sg, smeta = [], [], []
    for name, img, masks in load_synth(args.synth, args.synth_limit):
        mode, appearance, fams = SYNTH_META[name]
        for c, _, mi in sample_crops(img, masks, 12, rng):
            scrops.append(c); sg.append(name); smeta.append((mode, appearance, fams[mi]))
    print(f"crop pools: reference {len(rg)} from {len(set(rg))} images, synth {len(sg)} from {len(set(sg))} sheets", flush=True)
    sX, sg = embed(scrops), np.array(sg)
    X = np.concatenate([rX, sX]); g = np.concatenate([rg, sg]); y = np.r_[np.zeros(len(rg)), np.ones(len(sg))]
    meta = [("ref", "ref", "ref")] * len(rg) + smeta
    crops_all = rcrops + scrops

    aucs, by_mode, by_fam, first = [], {}, {}, None
    for seed in range(args.seeds):
        r = np.random.default_rng(seed)
        idx = np.concatenate([r.choice(np.nonzero(g == gg)[0], min(n, int((g == gg).sum())), replace=False)
                              for gg, n in [(gg, 60) for gg in np.unique(rg)] + [(gg, 8) for gg in np.unique(sg)]])
        ug = np.unique(g[idx]); fold = dict(zip(ug, r.integers(0, 5, len(ug)))); f = np.array([fold[x] for x in g[idx]])
        oof = np.zeros(len(idx))
        for k in range(5):
            tr, te = f != k, f == k
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
            clf.fit(X[idx][tr], y[idx][tr]); oof[te] = clf.predict_proba(X[idx][te])[:, 1]
        yy = y[idx]; aucs.append(roc_auc_score(yy, oof))
        for key_i, store in ((0, by_mode), (2, by_fam)):
            vals = np.array([meta[i][key_i] for i in idx])
            for v in set(vals[yy == 1]):
                sel = (yy == 0) | (vals == v)
                if (vals == v).sum() >= 20:
                    store.setdefault(v, []).append((roc_auc_score(yy[sel], oof[sel]), int((vals == v).sum())))
        if first is None:
            first = (idx, oof)
    auc = float(np.mean(aucs))
    print(f"crop AUC mean of {args.seeds} seeds: {auc:.3f}  (runs {', '.join(f'{a:.3f}' for a in aucs)})")
    for name, store in (("sheet type", by_mode), ("family", by_fam)):
        print(f"  by {name}: " + ", ".join(f"{k} {np.mean([a for a, _ in v]):.3f} (n~{v[0][1]})"
                                         for k, v in sorted(store.items(), key=lambda kv: -np.mean([a for a, _ in kv[1]]))))

    idx, oof = first
    yy = y[idx]; syn = np.nonzero(yy == 1)[0]; ref = np.nonzero(yy == 0)[0]
    lab = lambda j: f"{g[idx[j]].split(':')[-1][-12:]} {meta[idx[j]][2]} p={oof[j]:.2f}"
    for fname, sel in (("synth_most_obvious", syn[np.argsort(-oof[syn])][:args.top]),
                       ("synth_most_reference_like", syn[np.argsort(oof[syn])][:args.top]),
                       ("reference_most_typical", ref[np.argsort(oof[ref])][:args.top // 2]),
                       ("reference_most_synth_like", ref[np.argsort(-oof[ref])][:args.top // 2])):
        contact([crops_all[idx[j]] for j in sel], [lab(j) for j in sel], f"{args.out}/{fname}.jpg")
    json.dump({"synth": args.synth, "reference": args.reference or "val14", "long_side": LONG,
               "crop_auc_mean": auc, "crop_auc_runs": aucs,
               "by_sheet_type": {k: float(np.mean([a for a, _ in v])) for k, v in by_mode.items()},
               "by_family": {k: float(np.mean([a for a, _ in v])) for k, v in by_fam.items()}},
              open(f"{args.out}/summary.json", "w"), indent=1)
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
