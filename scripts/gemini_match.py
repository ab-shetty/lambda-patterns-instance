#!/usr/bin/env python3
"""How far is the generator from Gemini? A domain classifier as the yardstick (2026-10-20, CPU only).

Images of both sources are capped to the same long side (2048, the training scale), and each is described by ImageNet
Swin-T features (the project's backbone, cached weights) of a whole-sheet thumbnail and of K texture crops (224 px at the
2048 scale, sampled where there is ink). A logistic regression is cross-validated to tell generator from Gemini:
AUC 0.5 = indistinguishable, 1.0 = trivially separable. Lower is the target. It also writes the most separable images of
each side (what still gives the generator away) and per-mode AUCs, so a generator change can be judged by whether it moves
the AUC down -- for elevations against Gemini elevations, plans against plans.

    python3 scripts/gemini_match.py --gemini data/image_generation/gemini_r5 \
        --gemini-modes data/image_generation/gemini_r5/specs_described.jsonl --gen data/synthetic/pd_probe --out data/diag/match_20d
Features are cached per image (by path + mtime) in --cache, so re-scoring a new pool only embeds the new images.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

CACHE = Path("data/diag/match_cache")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def model():
    m = torchvision.models.swin_t()
    sd = torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth"), map_location="cpu")
    m.load_state_dict(sd)
    m.head = torch.nn.Identity()
    return m.eval()


def crops(path, K, rng):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    s = 2048 / max(h, w)
    im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]
    thumb = np.full((224, 224, 3), 255, np.uint8)
    t = cv2.resize(im, (224, max(1, round(224 * h / w))) if w >= h else (max(1, round(224 * w / h)), 224),
                   interpolation=cv2.INTER_AREA)
    thumb[:t.shape[0], :t.shape[1]] = t
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ink = cv2.blur((g < 200).astype(np.float32), (224, 224))
    ys, xs = np.nonzero(ink[112:h - 112, 112:w - 112] > 0.04) if h > 224 and w > 224 else ([], [])
    out = [thumb]
    for _ in range(K):
        if len(ys):
            j = rng.integers(len(ys)); y, x = ys[j], xs[j]
        else:
            y, x = rng.integers(0, max(1, h - 224)), rng.integers(0, max(1, w - 224))
        c = im[y:y + 224, x:x + 224]
        c = cv2.copyMakeBorder(c, 0, 224 - c.shape[0], 0, 224 - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        out.append(c)
    return out


@torch.no_grad()
def embed(m, paths, K=6):
    CACHE.mkdir(parents=True, exist_ok=True)
    feats = []
    for n, p in enumerate(paths):
        key = hashlib.md5(f"{p}:{os.path.getmtime(p)}:{K}".encode()).hexdigest()
        f = CACHE / f"{key}.npy"
        if f.exists():
            feats.append(np.load(f)); continue
        rng = np.random.default_rng(int(key[:8], 16))
        x = np.stack([(c[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD for c in crops(p, K, rng)]).transpose(0, 3, 1, 2)
        e = m(torch.from_numpy(x)).numpy()
        v = np.concatenate([e[0], e[1:].mean(0)])
        np.save(f, v); feats.append(v)
        if (n + 1) % 50 == 0:
            print(f"  embedded {n + 1}/{len(paths)}", flush=True)
    return np.array(feats)


def cv_auc(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        p[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p), p


def gen_modes(pool):
    out = {}
    for a in Path(pool, "annotations").glob("*.json"):
        d = json.load(open(a))
        out[d["image"]["file_name"]] = d.get("mode")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemini", required=True)
    ap.add_argument("--gemini-modes", default=None, help="specs_described.jsonl (mode per image) for per-mode AUCs")
    ap.add_argument("--gen", required=True, help="generator pool (images/ + annotations/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    gp = sorted(Path(a.gemini).glob("*.png"))[:a.n]
    sp = sorted(Path(a.gen, "images").glob("*.png"))[:a.n]
    gm = {}
    if a.gemini_modes:
        gm = {r["name"] + ".png": r.get("mode") for r in map(json.loads, open(a.gemini_modes)) if "error" not in r}
    sm = gen_modes(a.gen)
    m = model()
    print(f"gemini {len(gp)}, generator {len(sp)}", flush=True)
    X = np.vstack([embed(m, gp), embed(m, sp)])
    y = np.r_[np.zeros(len(gp)), np.ones(len(sp))]
    modes = [gm.get(p.name) for p in gp] + [sm.get(p.name) for p in sp]
    auc, p = cv_auc(X, y)
    res = {"auc_all": round(auc, 4), "n_gemini": len(gp), "n_gen": len(sp)}
    for md in ("elevation", "roof_plan", "freeform"):
        idx = [i for i, mm in enumerate(modes) if mm == md]
        if sum(y[idx] == 0) >= 10 and sum(y[idx] == 1) >= 10:
            res[f"auc_{md}"] = round(cv_auc(X[idx], y[idx])[0], 4)
    os.makedirs(a.out, exist_ok=True)
    names = [str(q) for q in gp + sp]
    order = np.argsort(p)
    res["most_gemini_like_generator"] = [names[i] for i in order if y[i] == 1][:8]
    res["most_generator_like_gemini"] = [names[i] for i in order[::-1] if y[i] == 0][:8]
    res["most_obvious_generator"] = [names[i] for i in order[::-1] if y[i] == 1][:8]
    json.dump(res, open(f"{a.out}/match.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("auc", "n_"))}))


if __name__ == "__main__":
    main()
