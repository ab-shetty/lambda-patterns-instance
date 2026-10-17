#!/usr/bin/env python3
"""Offline: (a) what the remaining real-sheet error is made of, (b) what simple mask post-processing buys.
Reads saved --save-probs maps (uint8, inference res); reproduces the evaluator's targets via _prepare_sheet.
usage: PYTHONPATH=. python3 scripts/postproc_probe.py MAPDIR SPLIT(hf14|val|gemini30) [fix]   (MAPDIR = a --save-probs dir @4096)
Writes per-question JSON to data/diag/postproc_<split>[_fix].json. First run 2026-10-17 (synth_progress.md)."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from evaluate_refunet_selection import _prepare_sheet, hand_boxes
from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask

HF14 = [12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24]
VAL = [4, 5, 6, 8, 9, 10, 13, 15, 17, 19, 20, 21, 22, 26]
T = 0.35


def fill_holes(b, max_area):
    inv = (~b).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = b.copy()
    h, w = b.shape
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        border = x == 0 or y == 0 or x + ww == w or y + hh == h
        if not border and (max_area is None or a <= max_area):
            out[lab == i] = True
    return out


def drop_small(b, min_frac):
    n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    if n <= 1:
        return b
    tot = st[1:, 4].sum()
    keep = np.zeros(n, bool); keep[1:] = st[1:, 4] >= min_frac * tot
    return keep[lab]


def hysteresis(p, lo, hi):
    b = p > lo
    n, lab = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)
    strong = np.unique(lab[p > hi]); keep = np.zeros(n, bool); keep[strong] = True; keep[0] = False
    return keep[lab]


def morph(b, op, k):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(b.astype(np.uint8), op, ker).astype(bool)


def variants(p):
    """p float [0,1] at inference res -> dict name -> binary at inference res (or ('native', prob) for linear)."""
    b = p > T
    v = {"base": b}
    for t in (0.25, 0.3, 0.4, 0.45, 0.5):
        v[f"thr{t}"] = p > t
    for f in (0.002, 0.01, 0.03):
        v[f"drop{f}"] = drop_small(b, f)
    for a in (256, 4096, None):
        v[f"fill{a}"] = fill_holes(b, a)
    for k in (3, 7, 15):
        v[f"close{k}"] = morph(b, cv2.MORPH_CLOSE, k)
        v[f"open{k}"] = morph(b, cv2.MORPH_OPEN, k)
    for hi in (0.6, 0.8, 0.9):
        v[f"hyst{hi}"] = hysteresis(p, T, hi)
    v["drop0.01+fill4096"] = fill_holes(drop_small(b, 0.01), 4096)
    v["hyst0.8+fill4096"] = fill_holes(hysteresis(p, T, 0.8), 4096)
    v["close7+drop0.01+fill"] = fill_holes(drop_small(morph(b, cv2.MORPH_CLOSE, 7), 0.01), None)
    return v


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / max(u, 1)


def decompose(pred, tgt, anyann, band=4):
    """loss buckets in IoU units (sum = 1 - IoU)."""
    u = max((pred | tgt).sum(), 1)
    fp, fn = pred & ~tgt, tgt & ~pred
    out = {}
    # FP: components of the prediction that barely touch the target are 'wrong region'
    n, lab, st, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    ov = np.bincount(lab[tgt], minlength=n)
    wrong = np.zeros(n, bool); wrong[1:] = ov[1:] < 0.1 * st[1:, 4]
    fp_wrong = fp & wrong[lab]
    out["fp_wrong_other_family"] = (fp_wrong & anyann).sum() / u
    out["fp_wrong_unlabelled"] = (fp_wrong & ~anyann).sum() / u
    fp_attached = fp & ~fp_wrong
    edge = cv2.dilate(tgt.astype(np.uint8), np.ones((2 * band + 1,) * 2, np.uint8)).astype(bool) & \
        ~cv2.erode(tgt.astype(np.uint8), np.ones((2 * band + 1,) * 2, np.uint8)).astype(bool)
    out["fp_boundary"] = (fp_attached & edge).sum() / u
    out["fp_spill_other_family"] = (fp_attached & ~edge & anyann).sum() / u
    out["fp_spill_unlabelled"] = (fp_attached & ~edge & ~anyann).sum() / u
    n, lab, st, _ = cv2.connectedComponentsWithStats(tgt.astype(np.uint8), connectivity=8)
    hit = np.bincount(lab[pred], minlength=n)
    missed = np.zeros(n, bool); missed[1:] = hit[1:] < 0.25 * st[1:, 4]
    fn_missed = fn & missed[lab]
    out["fn_missed_region"] = fn_missed.sum() / u
    out["fn_boundary"] = (fn & ~fn_missed & edge).sum() / u
    out["fn_holes"] = (fn & ~fn_missed & ~edge).sum() / u
    return out


def sheet_job(args):
    idx, rec, ov, fixes, mapdir = args
    sh = _prepare_sheet((idx, rec, 4096, 224, False, ov, fixes, False))
    h0, w0 = sh["h0"], sh["w0"]
    anns = rec["annotations"]
    anns = json.loads(anns) if isinstance(anns, str) else anns
    anyann = np.zeros((h0, w0), bool)
    for a in anns:
        anyann |= render_instance_mask(a["segmentation"], h0, w0).astype(bool)
    rows = []
    for ref_idx, cat, box, _, _ in sh["questions"]:
        bits, n = sh["targets"][cat]
        tgt = np.unpackbits(bits, count=h0 * w0).reshape(h0, w0).astype(bool)
        p = np.load(Path(mapdir) / f"{idx:03d}_{ref_idx:03d}.npz")["prob"].astype(np.float32) / 255
        r = {"image": idx, "ref": ref_idx}
        up = lambda b: cv2.resize(b.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
        for name, b in variants(p).items():
            r[name] = iou(up(b), tgt)
        lin = cv2.resize(p, (w0, h0), interpolation=cv2.INTER_LINEAR) > T
        r["linear"] = iou(lin, tgt)
        r["linear+fill"] = iou(lin | (up(fill_holes(p > T, 4096)) & ~up(p > T)), tgt)
        base = up(p > T)
        r["decomp"] = decompose(base, tgt, anyann & ~tgt)
        # 'shape' stats for end users: number of components and holes in the base prediction
        r["n_comp"] = cv2.connectedComponents((p > T).astype(np.uint8), connectivity=8)[0] - 1
        r["n_comp_tgt"] = cv2.connectedComponents(cv2.resize(tgt.astype(np.uint8), (p.shape[1], p.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST), connectivity=8)[0] - 1
        rows.append(r)
    return rows


def main():
    mapdir, split = sys.argv[1], sys.argv[2]
    fix = len(sys.argv) > 3
    fixes = json.load(open("eval_labels/hf14_fixes_v2.json")) if fix else {}
    if split == "gemini30":     # the evaluator's --local-pool records and --boxes eval_boxes/gemini30_v1.json
        pool_dir = Path("data/eval_pools/gemini30_s20261006")
        recs = {}
        for i, f in enumerate(sorted((pool_dir / "annotations").glob("*.json"))):
            ann = json.loads(f.read_text())
            recs[i] = {"image_path": str(pool_dir / "images" / ann["image"]["file_name"]),
                       "annotations": ann["annotations"], "ref_exclude": ann.get("ref_exclude")}
        hb = json.load(open("eval_boxes/gemini30_v1.json"))
        idxs = sorted(recs)
    else:
        recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
        hb = hand_boxes()
        idxs = HF14 if split == "hf14" else VAL
    jobs = [(i, {k: recs[i][k] for k in ("image", "image_path", "annotations", "ref_exclude") if k in recs[i]},
             hb.get(str(i), {}), fixes.get(str(i), []), mapdir) for i in idxs]
    with Pool(16) as pool:
        rows = [r for rs in pool.map(sheet_job, jobs) for r in rs]
    out = Path("data/diag") / f"postproc_{split}{'_fix' if fix else ''}.json"
    json.dump(rows, open(out, "w"))
    names = [k for k in rows[0] if k not in ("image", "ref", "decomp", "n_comp", "n_comp_tgt")]
    def persheet(k):
        by = {}
        for r in rows:
            by.setdefault(r["image"], []).append(r[k])
        return np.mean([np.mean(v) for v in by.values()])
    b = np.array([r["base"] for r in rows])
    print(f"{split}: n={len(rows)} base mean {b.mean():.4f} per-sheet {persheet('base'):.4f}")
    for k in names:
        a = np.array([r[k] for r in rows]); d = a - b
        print(f"  {k:24s} {a.mean():.4f} ({d.mean():+.4f}; up {int((d > 0.005).sum())} / down {int((d < -0.005).sum())};"
              f" worst {d.min():+.3f}) per-sheet {persheet(k):.4f}")
    dk = rows[0]["decomp"].keys()
    print("  loss decomposition (IoU units, mean over questions):")
    for k in dk:
        print(f"    {k:24s} {np.mean([r['decomp'][k] for r in rows]):.4f}")
    print(f"    total                    {1 - b.mean():.4f}")
    nc = np.array([r["n_comp"] for r in rows]); nt = np.array([r["n_comp_tgt"] for r in rows])
    print(f"  components per prediction: median {np.median(nc):.0f}, p90 {np.percentile(nc, 90):.0f}, max {nc.max()};"
          f" target median {np.median(nt):.0f}, p90 {np.percentile(nt, 90):.0f}")


if __name__ == "__main__":
    main()
