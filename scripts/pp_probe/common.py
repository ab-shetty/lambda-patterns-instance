"""Shared loader: for a split, yield per sheet (idx, native BGR image, anyann, [(ref, cat, target, prob_u8)])."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from evaluate_refunet_selection import _prepare_sheet, hand_boxes
from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask
HF14 = [12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24]
VAL = [4, 5, 6, 8, 9, 10, 13, 15, 17, 19, 20, 21, 22, 26]
WITH_BOX = [False]      # pmap.py sets True to also get the reference box (native x, y, w, h)
MAPS = Path("data/floz-diag/xblend-kq30-kms30")

def jobs(split):
    fix = split == "hf14fix2"
    fixes = json.load(open("eval_labels/hf14_fixes_v2.json")) if fix else {}
    if split == "gemini30":
        pool_dir = Path("data/eval_pools/gemini30_s20261006"); recs = {}
        for i, f in enumerate(sorted((pool_dir / "annotations").glob("*.json"))):
            ann = json.loads(f.read_text())
            recs[i] = {"image_path": str(pool_dir / "images" / ann["image"]["file_name"]), "annotations": ann["annotations"],
                       "ref_exclude": ann.get("ref_exclude")}
        hb = json.load(open("eval_boxes/gemini30_v1.json")); idxs = sorted(recs)
    else:
        recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
        hb = hand_boxes(); idxs = HF14 if split.startswith("hf14") else VAL
    mapdir = MAPS / f"{'hf14' if split.startswith('hf14') else split}_4096" / "probs"
    return [(i, {k: recs[i][k] for k in ("image", "image_path", "annotations", "ref_exclude") if k in recs[i]},
             hb.get(str(i), {}), fixes.get(str(i), []), str(mapdir)) for i in idxs]

def load(job):
    idx, rec, ov, fixes, mapdir = job
    sh = _prepare_sheet((idx, rec, 4096, 224, False, ov, fixes, False))
    h0, w0 = sh["h0"], sh["w0"]
    if "image_path" in rec: img = cv2.imread(rec["image_path"])
    else:
        b = rec["image"]; b = b["bytes"] if isinstance(b, dict) else b
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if img.shape[:2] != (h0, w0): img = cv2.resize(img, (w0, h0))
    anns = rec["annotations"]; anns = json.loads(anns) if isinstance(anns, str) else anns
    anyann = np.zeros((h0, w0), bool)
    for a in anns: anyann |= render_instance_mask(a["segmentation"], h0, w0).astype(bool)
    qs = []
    for ref_idx, cat, box, _, _ in sh["questions"]:
        bits, n = sh["targets"][cat]
        tgt = np.unpackbits(bits, count=h0 * w0).reshape(h0, w0).astype(bool)
        p = np.load(Path(mapdir) / f"{idx:03d}_{ref_idx:03d}.npz")["prob"]
        qs.append((ref_idx, cat, tgt, p) if not WITH_BOX[0] else (ref_idx, cat, tgt, p, box))
    return idx, img, anyann, qs

def ink_map(img, T=60):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.dilate(g, np.ones((9, 9), np.uint8))
    return (bg.astype(np.int16) - g) > T

def edge(m):
    m8 = m.astype(np.uint8)
    return (cv2.dilate(m8, np.ones((3, 3), np.uint8)) != cv2.erode(m8, np.ones((3, 3), np.uint8))) & m

def ink_map_v2(img, k=7, min_len=12):
    """Thin dark strokes only: black top-hat (closing - image, ellipse k) responds to lines narrower than k and not to the edges
    of flat paint; threshold from the sheet's own stroke contrast (half its 99.5th percentile, clipped to 15..60) so faint-line
    sheets register; ink components whose bounding box is shorter than min_len on both sides (fill grain, stucco dots) dropped."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    T = float(np.clip(0.5 * np.percentile(bh, 99.5), 15, 60))
    ink = bh > T
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = np.maximum(st[1:, 2], st[1:, 3]) >= min_len
    return keep[lab]
