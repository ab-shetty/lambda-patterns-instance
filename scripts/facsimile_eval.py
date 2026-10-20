#!/usr/bin/env python3
"""Per-image facsimiles of Gemini sheets, and how faithful they are (2026-10-20; the user's aim: hand over a new Gemini
image with its description, get a near-same synthetic version from the generator).

  specs  DESCRIBED.jsonl OUT.json   described images (describe_gemini_specs.py) -> --spec list for generate_synthetic_fc.py
                                    (elevations; keys the generator does not know are dropped and listed as gaps)
  score  ORIG_DIR TWIN_POOL OUT_DIR one score per pair: cosine similarity of Swin-T features (thumbnail + texture crops,
                                    gemini_match.embed) between the twin and its original, and retrieval -- the rank of the
                                    original among all originals when queried with the twin (1 = the twin is closest to its
                                    own source). Writes side-by-sides of the worst pairs.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HOUSE_KEYS = {"stories", "roof", "units", "wing", "ell", "porch", "chimney", "dormers", "accent", "accent2", "label_trim",
              "cross_gables", "split", "ridge"}
WALL = {"lap", "vertical", "bb", "shingle", "brick", "stone", "stipple", "flat", "panel"}
ROOF = {"asphalt", "seam", "tile_roof", "shingle", "flat_roof"}


def to_specs(described, out):
    specs = []
    for r in map(json.loads, open(described)):
        if "error" in r or r.get("mode") != "elevation":
            continue
        s = {"name": r["name"], "mode": "elevation", "shaded": bool(r.get("shaded")),
             "views": r.get("views") if r.get("views") in (1, 2, 4) else 1, "house": {}, "styles": {},
             "gaps": list(r.get("gaps", []))}
        vl = [v for v in (r.get("view_list") or []) if v in ("front", "rear", "left", "right")]
        if vl:
            s["view_list"] = vl[:4]
            s["views"] = 1 if len(vl) == 1 else 2 if len(vl) == 2 else 4
        fa = (r.get("house") or {}).get("facade_aspect")
        if isinstance(fa, (int, float)) and 0.3 < fa < 15:
            s["facade_aspect"] = float(fa)
        for k, v in (r.get("house") or {}).items():
            if k == "facade_aspect":
                continue
            if k == "split" and not v:
                continue                        # absence of a split is not a constraint worth searching for
            if k in HOUSE_KEYS and v is not None and not (k in ("units",) and v in (0, 1)):
                s["house"][k] = v
            elif k not in HOUSE_KEYS:
                s["gaps"].append(f"house.{k}={v}")
        for role, st in (r.get("styles") or {}).items():
            kind = (st or {}).get("kind")
            ok = (kind in ROOF) if role == "roof" else (kind in WALL)
            if not ok:
                s["gaps"].append(f"style {role}={kind}")
                continue
            e = {"kind": kind}
            if s["shaded"] and isinstance(st.get("base"), list) and len(st["base"]) == 3:
                e["base"] = [int(v) for v in st["base"]]
            s["styles"][role] = e
        specs.append(s)
    json.dump(specs, open(out, "w"), indent=1)
    print(f"{len(specs)} elevation specs -> {out}")


def score(orig_dir, pool, out_dir):
    sys.path.insert(0, str(Path(__file__).parent))
    import gemini_match as gm
    matches = json.load(open(Path(pool, "spec_matches.json")))
    imgs = {p.stem.split("_")[-1]: p for p in Path(pool, "images").glob("*.png")}
    pairs = [(Path(orig_dir, m["name"] + ".png"), imgs.get(str(m["image_id"])), m) for m in matches]
    pairs = [p for p in pairs if p[1] is not None and p[0].exists()]
    m = gm.model()
    O = gm.embed(m, [p[0] for p in pairs]); T = gm.embed(m, [p[1] for p in pairs])
    def norm(X):
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    On, Tn = norm(np.vstack([O, T]))[:len(O)], norm(np.vstack([O, T]))[len(O):]
    sim = Tn @ On.T
    own = np.diag(sim)
    rank = (sim > own[:, None]).sum(1) + 1
    rows = [{"name": p[2]["name"], "image_id": p[2]["image_id"], "unmet": p[2]["unmet"], "cos": round(float(c), 4),
             "rank": int(k)} for p, c, k in zip(pairs, own, rank)]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(Path(out_dir, "fidelity.json"), "w"), indent=1)
    n = len(rows)
    print(f"{n} pairs: own-pair cosine median {np.median(own):.3f} (random pairs {np.median(sim[~np.eye(n, dtype=bool)]):.3f}); "
          f"retrieval top-1 {np.mean(rank == 1):.2f}, top-5 {np.mean(rank <= 5):.2f}, median rank {np.median(rank):.0f} of {n}")
    def fit(f, H=260, W=620):
        im = cv2.imread(str(f)); h, w = im.shape[:2]; s = min(W / w, H / h)
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return cv2.copyMakeBorder(im, 2, H + 2 - im.shape[0], 2, W + 2 - im.shape[1], cv2.BORDER_CONSTANT, value=(200, 200, 200))
    order = np.argsort(own)
    for tag, idx in (("worst", order[:6]), ("best", order[::-1][:6])):
        cv2.imwrite(str(Path(out_dir, f"{tag}.jpg")), np.vstack([np.hstack([fit(pairs[i][0]), fit(pairs[i][1])]) for i in idx]))
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "specs":
        to_specs(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4])
