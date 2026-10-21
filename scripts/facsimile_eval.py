#!/usr/bin/env python3
"""Per-image facsimiles of Gemini sheets, and how faithful they are (2026-10-20; the user's aim: hand over a new Gemini
image with its description, get a near-same synthetic version from the generator).

  specs  DESCRIBED.jsonl OUT.json [IMAGE_DIR]   described images (describe_gemini_specs.py) -> --spec list for generate_synthetic_fc.py
                                    (elevations; keys the generator does not know are dropped and listed as gaps)
  score  ORIG_DIR TWIN_POOL OUT_DIR one score per pair: cosine similarity of Swin-T features (thumbnail + texture crops,
                                    gemini_match.embed) between the twin and its original, and retrieval -- the rank of the
                                    original among all originals when queried with the twin (1 = the twin is closest to its
                                    own source). Writes side-by-sides of the worst pairs.
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HOUSE_KEYS = {"stories", "roof", "units", "wing", "ell", "porch", "chimney", "dormers", "accent", "accent2", "label_trim",
              "cross_gables", "split", "ridge"}
WALL = {"lap", "vertical", "bb", "shingle", "brick", "stone", "stipple", "flat", "panel"}
ROOF = {"asphalt", "seam", "tile_roof", "shingle", "flat_roof"}


def refine_box(g, box):
    """Gemini's building box ([ymin, xmin, ymax, xmax] 0-1000) with its top corrected on the ink: from a third of the way
    down, upward through the roof (middle 60% of the width) to the first blank band -- Gemini's boxes often cut off the
    roof peak. x and the bottom are kept (an ink-based x split hip roofs and caught level markers). Pixel (x0, y0, x1, y1)."""
    H, W = g.shape
    y0, x0, y1, x1 = [int(round(v * (H if i % 2 == 0 else W) / 1000)) for i, v in enumerate(box)]
    if y1 - y0 < 10 or x1 - x0 < 10:
        return None
    ink = g < min(200, np.percentile(g, 95) - 30)
    nx0, nx1 = x0, x1
    c0, c1 = nx0 + (nx1 - nx0) // 5, nx1 - (nx1 - nx0) // 5
    top, gap, y = y0 + (y1 - y0) // 3, 0, y0 + (y1 - y0) // 3
    while y > max(0, y0 - (y1 - y0) // 2):
        y -= 1
        if ink[y, c0:c1].any():
            top, gap = y, 0
        else:
            gap += 1
            if gap > max(4, 0.03 * (y1 - y0)):
                break
    return nx0, top, nx1, y1


EXCLUDED = Path(__file__).parent.parent / "image_generation/facsimile/r5_excluded.json"


def to_specs(described, out, image_dir=None):
    specs = []
    excluded = json.load(open(EXCLUDED)) if EXCLUDED.exists() else {}
    for r in map(json.loads, open(described)):
        if "error" in r or r.get("mode") != "elevation":
            continue
        if "perspective_or_3d" in (r.get("gap_tags") or []):
            continue                            # 3-D / perspective sheets: not what clients send (user, 2026-10-21)
        if r["name"] in excluded:
            continue
        s = {"name": r["name"], "mode": "elevation", "shaded": bool(r.get("shaded")),
             "views": r.get("views") if r.get("views") in (1, 2, 4) else 1, "house": {}, "styles": {},
             "gaps": list(r.get("gaps", []))}
        s["bay"] = "bay_or_bump_out" in (r.get("gap_tags") or [])     # forced (--spec), not searched
        s["apartment"] = "apartment_or_multistorey_block" in (r.get("gap_tags") or [])
        s["balconies"] = "balconies" in (r.get("gap_tags") or [])
        gt = " | ".join(r.get("gaps") or []).lower()     # massing the vocabulary has no key for, read from the gap text
        s["recessed_balconies"] = s["balconies"] and bool(re.search(r"(inset|recess)\w* balcon|balcon\w* (are )?(inset|recess)", gt))
        s["sections"] = bool(re.search(r"projecting and recessed|recessed and projecting|staggered|articulated|"
                                       r"projects? forward|bumps? out slightly|recessed wall section", gt))
        s["recessed_entry"] = bool(re.search(r"recessed (front )?entry|entry (is )?recessed", gt))
        s["upper_setback"] = bool(re.search(r"(second|third|upper|top) (story|storey|floor)( section)? (is )?(set|stepped)[ -]back|"
                                            r"stepped[ -]back (second|third|upper)", gt))
        vl = [v for v in (r.get("view_list") or []) if v in ("front", "rear", "left", "right")]
        if vl:
            s["view_list"] = vl[:4]
            s["views"] = 1 if len(vl) == 1 else 2 if len(vl) == 2 else 4
        if image_dir and Path(image_dir, r["name"] + ".png").exists():
            # the drawing's ink box, not the canvas (Gemini's canvases share one wide aspect whatever the layout):
            # picks the twin's view layout (row / grid / column)
            g = cv2.cvtColor(cv2.imread(str(Path(image_dir, r["name"] + ".png"))), cv2.COLOR_BGR2GRAY)
            ys, xs = np.nonzero(g < np.percentile(g, 95) - 30)
            if len(xs):
                s["sheet_aspect"] = round((np.percentile(xs, 99.5) - np.percentile(xs, 0.5)) /
                                          max(1, np.percentile(ys, 99.5) - np.percentile(ys, 0.5)), 3)
        fa = (r.get("house") or {}).get("facade_aspect")
        if isinstance(fa, (int, float)) and 0.3 < fa < 15:
            s["facade_aspect"] = float(fa)
        vb = r.get("view_boxes") or []
        if len(vb) == 4 and all(isinstance(v, (int, float)) for v in vb):
            vb = [vb]                           # one view given as a bare box
        if image_dir and vb and isinstance(vb[0], list) and len(vb[0]) == 4 and max(vb[0]) <= 1000 and Path(image_dir, r["name"] + ".png").exists():
            # the first view's building box (pixels, refined on the ink): a better facade aspect than the describer's
            # own estimate
            g = cv2.cvtColor(cv2.imread(str(Path(image_dir, r["name"] + ".png"))), cv2.COLOR_BGR2GRAY)
            bx = refine_box(g, vb[0])
            if bx:
                s["facade_aspect"] = round((bx[2] - bx[0]) / max(1, bx[3] - bx[1]), 3)
        for k, v in (r.get("house") or {}).items():
            if k == "facade_aspect":
                continue
            if k in ("garage", "garage_doors"):
                s[k] = v                        # forced (--spec), not searched
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
        to_specs(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4])
