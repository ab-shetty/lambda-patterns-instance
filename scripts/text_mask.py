#!/usr/bin/env python3
"""Text regions of a drawing (Tesseract word boxes, upright and rotated 90 degrees), as a pixel mask.

Hand-labelled sheets (real, Gemini) put reference boxes on room tags, material callouts and slope
labels: a user never selects those. `text_mask` returns the native-size bool mask of detected words,
dilated by a fraction of the text height; `--probe` checks it against a hand list of unrealistic boxes
on a local pool (share of each question's box covered by text).

    PYTHONPATH=. python3 scripts/text_mask.py --probe --pool POOL --metrics M.json --bad BAD.json
"""
import argparse
import json
import re
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pytesseract

WORD = re.compile(r"[A-Za-z0-9]")


def _words(gray, conf_min):
    # >= 3 characters at conf >= 60: Tesseract also "reads" hatching and stone as 1-2 character words
    # (gemini30 probe: 2.8% -> 1.0% of labelled area excluded, 17 / 33 hand-flagged boxes still caught)
    d = pytesseract.image_to_data(gray, config="--psm 11", output_type=pytesseract.Output.DICT)
    out = []
    for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"]):
        t = t.strip()
        if float(c) >= conf_min and len(WORD.findall(t)) >= 3 and 4 <= h <= 200 and w <= 40 * h:
            out.append((x, y, w, h))
    return out


def text_mask(img_bgr, conf_min=60, pad=0.35):
    """Bool mask (native size) of detected text, each word box grown by `pad` x its height."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    m = np.zeros((H, W), np.uint8)
    boxes = [(x, y, w, h) for x, y, w, h in _words(gray, conf_min)]
    rot = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)          # vertical text reads upright here
    for x, y, w, h in _words(rot, conf_min):                  # rotated (x, y) -> original
        boxes.append((y, H - x - w, h, w))
    for x, y, w, h in boxes:
        p = int(round(pad * min(w, h)))
        m[max(0, y - p):y + h + p, max(0, x - p):x + w + p] = 1
    return m.astype(bool)


def _sheet(args):
    path, = args
    return text_mask(cv2.imread(str(path)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--bad", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pool = Path(a.pool)
    files = sorted((pool / "annotations").glob("*.json"))
    imgs = [pool / "images" / json.loads(f.read_text())["image"]["file_name"] for f in files]
    with Pool(30) as p:
        masks = p.map(_sheet, [(i,) for i in imgs])
    bad = {tuple(k) for k in json.load(open(a.bad))["dropped"]}
    rows = json.load(open(a.metrics))["selections"]
    cov, y = [], []
    for r in rows:
        x, yy, w, h = r["reference_box_native"]
        cov.append(float(masks[r["image_index"]][yy:yy + h, x:x + w].mean()))
        y.append((r["image_index"], r["reference_instance"]) in bad)
    cov, y = np.array(cov), np.array(y)
    print(f"sheets {len(files)}, text pixels {np.mean([m.mean() for m in masks]):.1%} of each sheet on average")
    for t in (0.01, 0.05, 0.1, 0.2):
        fl = cov > t
        print(f"text covers > {t:.0%} of box: flagged {fl.sum():3d}  recall {fl[y].mean():.2f} "
              f"({fl[y].sum()}/{y.sum()})  precision {y[fl].mean() if fl.any() else 0:.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps([{"image_index": r["image_index"], "reference_instance":
                                            r["reference_instance"], "text_cover": c, "bad": bool(b)}
                                           for r, c, b in zip(rows, cov, y)], indent=1))


if __name__ == "__main__":
    main()
