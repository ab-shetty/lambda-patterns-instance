#!/usr/bin/env python3
"""One edge convention for training (2026-11-19). The real labels include the boundary stroke of a region (a 1 px outward
polygon offset gains real sheets +0.003..0.007), the Gemini labels stop inside it (the same offset loses Gemini30 -0.0072,
synth_progress 2026-11-11). Training on both teaches two conventions. This script moves Gemini labels to the real one.

  measure   per labelled family, the share of drawn ink within R_BAND px of the label edge that lies INSIDE the label
            (pixel-weighted over a split). Labels only, no model.
              python3 scripts/edge_convention.py measure --real 4,5,6,...      (real val, label fixes v4)
              python3 scripts/edge_convention.py measure --pool data/eval_pools/gemini30_s20261006 [--r 2]
  convert   grow every instance over the ink it touches within r px (never into another family's label), rewrite polygons:
              python3 scripts/edge_convention.py convert --pool SRC --out DST --r 2
Ink: common.ink_map (pixels > 60 grey levels darker than their 9 px neighbourhood maximum), the edgeink / inkgrow probes' map.
"""
import argparse
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "pp_probe"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "selftrain"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refmask2former.dataset import render_instance_mask, IGNORE_PREFIX  # noqa: E402

R_BAND = 3
CFG = {}


def ink_map(img, T=60):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (cv2.dilate(g, np.ones((9, 9), np.uint8)).astype(np.int16) - g) > T


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def grow(fam, others, ink, r):
    """fam grown over ink pixels within r px of it (iteratively, so a stroke is followed only where it touches), never into
    pixels of other families."""
    if r <= 0:
        return fam
    m = fam.copy()
    allowed = ink & ~others
    for _ in range(r):
        m = m | (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & allowed)
    return m


def inside_share(fam, ink):
    """(ink pixels inside fam within R_BAND of its edge, ink pixels within R_BAND of the edge on either side)."""
    f8 = fam.astype(np.uint8)
    band = cv2.dilate(f8, disk(R_BAND)).astype(bool) & ~cv2.erode(f8, disk(R_BAND)).astype(bool)
    b = band & ink
    return int((b & fam).sum()), int(b.sum())


def pool_families(pool, stem):
    ann = json.loads((Path(pool) / "annotations" / f"{stem}.json").read_text())
    img = cv2.imread(str(Path(pool) / "images" / ann["image"]["file_name"]), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    fams = {}
    for a in ann["annotations"]:
        c = a.get("category_name", "pattern")
        if c.startswith(IGNORE_PREFIX):
            continue
        fams.setdefault(c, np.zeros((h, w), bool))
        fams[c] |= render_instance_mask(a["segmentation"], h, w).astype(bool)
    return img, fams, ann


def measure_one(job):
    kind, key = job
    if kind == "pool":
        img, fams, _ = pool_families(CFG["pool"], key)
    else:
        from pseudolabel import human_targets, CFG as PCFG
        from types import SimpleNamespace
        PCFG.update(recs=CFG["recs"], fixes=CFG["fixes"])
        rec = CFG["recs"][key]
        img = cv2.imdecode(np.frombuffer(rec["image"], np.uint8), cv2.IMREAD_COLOR)
        fams = {c: T for c, (T, _) in human_targets(f"real{key:03d}", SimpleNamespace(real=True)).items()}
    ink = ink_map(img)
    out = []
    for c, m in fams.items():
        others = np.logical_or.reduce([o for k, o in fams.items() if k != c]) if len(fams) > 1 else np.zeros_like(m)
        a, b = inside_share(grow(m, others, ink, CFG["r"]), ink)
        out.append((a, b))
    return out


def convert_one(stem):
    img, fams, ann = pool_families(CFG["pool"], stem)
    h, w = img.shape[:2]
    ink = ink_map(img)
    grown = {}
    for c, m in fams.items():
        others = np.logical_or.reduce([o for k, o in fams.items() if k != c]) if len(fams) > 1 else np.zeros_like(m)
        grown[c] = grow(m, others, ink, CFG["r"])
    # two families may both claim one stroke pixel next to both: give it to neither (it stays as each label had it)
    for c in grown:
        for k in grown:
            if k != c:
                both = grown[c] & grown[k] & ~fams[c]
                grown[c] &= ~both
    from pseudolabel import mask_segs
    anns = [a for a in ann["annotations"] if str(a.get("category_name", "")).startswith(IGNORE_PREFIX)]
    added = 0
    for c, m in grown.items():
        added += int((m & ~fams[c]).sum())
        segs, specks = mask_segs(m, 1)
        anns += [{"segmentation": s, "category_name": c} for s in segs]
    ann = {**ann, "annotations": anns, "edge_convention": {"r": CFG["r"], "added_px": added}}
    out = Path(CFG["out"])
    (out / "annotations" / f"{stem}.json").write_text(json.dumps(ann))
    src = Path(CFG["pool"]) / "images" / ann["image"]["file_name"]
    dst = out / "images" / ann["image"]["file_name"]
    if not dst.exists():
        dst.hardlink_to(src)
    return added, int(sum(m.sum() for m in fams.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["measure", "convert"])
    ap.add_argument("--pool"); ap.add_argument("--real"); ap.add_argument("--out")
    ap.add_argument("--r", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    CFG.update(pool=a.pool, out=a.out, r=a.r)
    if a.mode == "measure":
        if a.real:
            from evaluate_refunet_selection import load_parquet_records
            CFG["recs"] = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
            CFG["fixes"] = json.load(open("eval_labels/real_fixes_v4.json"))
            jobs = [("real", int(i)) for i in a.real.split(",")]
        else:
            jobs = [("pool", p.stem) for p in sorted(Path(a.pool, "annotations").glob("*.json"))]
        with Pool(a.workers) as p:
            res = [x for r in p.map(measure_one, jobs) for x in r]
        A, B = sum(x for x, _ in res), sum(y for _, y in res)
        fam = np.mean([x / y for x, y in res if y])
        print(f"{a.real and 'real ' + a.real or a.pool} r={a.r}: ink within {R_BAND} px of the edge inside the label "
              f"{A / max(B, 1):.3f} (pixel-weighted), {fam:.3f} (family mean), {len(res)} families")
        return
    out = Path(a.out)
    (out / "annotations").mkdir(parents=True, exist_ok=True); (out / "images").mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in sorted(Path(a.pool, "annotations").glob("*.json"))]
    with Pool(a.workers) as p:
        res = p.map(convert_one, stems)
    for f in Path(a.pool).iterdir():
        if f.is_file() and not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    print(f"converted {len(stems)} sheets at r={a.r}: +{sum(x for x, _ in res) / max(sum(y for _, y in res), 1):.4f} of label area")


if __name__ == "__main__":
    main()
