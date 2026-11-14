#!/usr/bin/env python3
"""Pre-labelling, stage D: round-5 drafts into a NEW Roboflow project for the user to correct (2026-11-14).

Risk guards (agreed with the user):
  * omissions made visible -- large inked areas no confident family covers get a `check_here` polygon (delete it once
    checked); families found by a single box are `uncertain_N`, confident ones `patternN`;
  * anchoring measured -- HOLDOUT sheets (fixed, seeded) go up with NO drafts, to be labelled from scratch; comparing
    agreement on drafted vs blank sheets says whether drafts bias the labels;
  * provenance -- every uploaded draft is kept locally (data/selftrain/drafts_uploaded/<sheet>.json); a later export is
    diffed against it: unchanged polygons = draft, changed = human.
Nothing is uploaded without --execute.

    PYTHONPATH=.:scripts python3 scripts/selftrain/upload_drafts.py --disc data/selftrain/disc2_r5_all \
        --images data/image_generation/gemini_r5 --project floz-gen-gemini-r5 [--execute]
"""
import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

import pseudolabel as PLB
import product

HOLDOUT_N, HOLDOUT_SEED = 15, 20261114
UNCERTAIN = False        # 2026-11-14, Gemini30: uncertain drafts added 3 real families for 8.5 mostly-noise polygons per sheet
                         # (furniture / window blobs on round 5); off: 47 drafted, 6 flagged by check_here, 4 silent of 57
CHECK_MIN = 0.005        # uncovered inked area (share of the sheet) that earns a check_here marker


def check_regions(img, covered, win_frac=0.01, dens=0.08):
    """Boxes around areas DENSE with line work (hatching, stipple, brick: stroke pixels > dens of a window ~1% of the long side)
    that no included draft covers, text excluded (scripts/text_mask.py). Single outlines are not dense, so markers sit on
    undrafted materials instead of wrapping whole elevations (the first version: ink components, which joined everything)."""
    from text_mask import text_mask
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stroke = ((cv2.dilate(g, np.ones((9, 9), np.uint8)).astype(np.int16) - g) > 20).astype(np.float32)
    k = max(15, int(win_frac * max(g.shape)) | 1)
    dense = cv2.blur(stroke, (k, k)) > dens
    dense &= ~cv2.dilate(text_mask(img).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    dense &= ~cv2.dilate(covered.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    dense = cv2.morphologyEx(dense.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
    H, W = dense.shape
    out = []
    for j in range(1, n):
        x, y, w, h = (int(v) for v in st[j, :4])
        if st[j, 4] < CHECK_MIN * dense.size:
            continue
        edge = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        if edge and min(w, h) < 0.03 * max(H, W):       # a thin strip along the sheet border = the border's own line work
            continue
        out.append((x, y, w, h))
    return out


def drafts(d, images):
    meta = json.loads((d / "families.json").read_text())
    src = next(p for p in Path(images).iterdir() if p.stem == meta["sheet"])
    img = cv2.imdecode(np.frombuffer(src.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    fams = PLB.family_layers(d, img, meta)
    out, n_conf, n_unc = [], 0, 0
    covered = np.zeros(img.shape[:2], bool)
    for f in fams:
        if f["pl"]:
            n_conf += 1; label = f"pattern{n_conf}"
        elif UNCERTAIN and f["support"] >= 2 and f["plain"] < 0.5 and not f.get("fragment"):
            n_unc += 1; label = f"uncertain_{n_unc}"
        else:
            continue        # single-box / blank-paper families: left out (the dry run had ~55 such polygons per sheet);
                            # what they covered is not "covered", so check_here still marks it
        covered |= f["draft"]
        for p in f["polys"]:
            out.append((label, p["outer"]))
            # Roboflow polygons carry no holes: holes become their own `hole` polygons for the reviewer to subtract
            out += [("hole_" + label, hp) for hp in p["holes"]]
    for x, y, w, h in check_regions(img, covered):
        out.append(("check_here", [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    return meta["sheet"], src, img.shape[:2], out


def coco(name, hw, polys):
    cats = sorted({l for l, _ in polys})
    c = {"images": [{"id": 0, "file_name": name, "width": hw[1], "height": hw[0]}],
         "categories": [{"id": i + 1, "name": n} for i, n in enumerate(cats)], "annotations": []}
    for k, (l, p) in enumerate(polys):
        a = np.asarray(p, float)
        x0, y0 = a.min(0); x1, y1 = a.max(0)
        c["annotations"].append({"id": k + 1, "image_id": 0, "category_id": cats.index(l) + 1, "iscrowd": 0,
                                 "segmentation": [a.ravel().tolist()], "bbox": [x0, y0, x1 - x0, y1 - y0],
                                 "area": float(cv2.contourArea(a.astype(np.float32)))})
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disc", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--project", required=True); ap.add_argument("--execute", action="store_true")
    ap.add_argument("--elevations", default="data/selftrain/r5_elevations.txt")
    a = ap.parse_args()
    PLB.CFG.update(hi=PLB.HI, lo=PLB.LO, s_min=PLB.S_MIN, min_area=PLB.MIN_AREA)
    el = Path(a.elevations).read_text().split()
    hold = set(random.Random(HOLDOUT_SEED).sample(sorted(el), HOLDOUT_N))
    keep = Path("data/selftrain/drafts_uploaded"); keep.mkdir(parents=True, exist_ok=True)
    (keep / "_holdout.json").write_text(json.dumps(sorted(hold), indent=1))
    dirs = sorted(p for p in Path(a.disc).iterdir() if (p / "families.json").exists())
    stats = []
    for d in dirs:
        sid, src, hw, polys = drafts(d, a.images)
        if sid in hold:
            polys = []
        c = coco(src.name, hw, polys)
        (keep / f"{sid}.json").write_text(json.dumps(c))
        stats.append((sid, sid in hold, sum(l.startswith("pattern") for l, _ in polys),
                      sum(l.startswith("uncertain") for l, _ in polys), sum(l == "check_here" for l, _ in polys)))
    print(f"{len(stats)} sheets, {len(hold)} holdout (no drafts); polygons: "
          f"pattern {sum(s[2] for s in stats)}, uncertain {sum(s[3] for s in stats)}, check_here {sum(s[4] for s in stats)}")
    if not a.execute:
        print("dry run: nothing uploaded (pass --execute)")
        return
    from roboflow import Roboflow
    key = os.environ["ROBOFLOW_API_KEY"]
    ws = Roboflow(api_key=key).workspace("perceive-ai")
    try:
        proj = ws.project(a.project)
    except Exception:
        proj = ws.create_project(project_name=a.project, project_type="instance-segmentation",
                                 project_license="CC BY 4.0", annotation="pattern")
    for sid, hold_, *_ in stats:
        src = next(p for p in Path(a.images).iterdir() if p.stem == sid)
        c = json.loads((keep / f"{sid}.json").read_text())
        tags = ["r5", "holdout-no-draft" if hold_ else "draft-fgavg78"]
        if c["annotations"]:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(c, f); ann = f.name
            proj.single_upload(image_path=str(src), annotation_path=ann, batch_name="r5-drafts", tag_names=tags)
        else:
            proj.single_upload(image_path=str(src), batch_name="r5-drafts", tag_names=tags)
        print("uploaded", sid, tags, len(c["annotations"]), flush=True)


if __name__ == "__main__":
    main()
