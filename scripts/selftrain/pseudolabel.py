#!/usr/bin/env python3
"""Self-training / pre-labelling, stage B (CPU): turn stage A's families (discover.py) into
  * drafts   -- each family's product polygons (scripts/product.py, the shipped post-processing), for a reviewer;
  * pseudo-labels -- per family a CONFIDENT target (draft & blended P > HI), confident background (P < LO, outside the
    draft) and everything else IGNORED (no loss). Pixels confident for two families are ignored for both. Only families
    found by >= S_MIN agreeing boxes become pseudo-labels.
and, on labelled sheets (--local-pool / --real), score both against the human labels:
  draft quality   per human family: IoU of the best-matching draft; misses (best < 0.5); merges (a draft covering
                  >= 30% of two or more human families);
  pseudo-labels   pixel precision of the confident target, its recall, target pixels wrongly marked confident background
                  ("neg error"), and the share of the target left ignored.

    PYTHONPATH=.:scripts python3 scripts/selftrain/pseudolabel.py --disc data/selftrain/disc_gemini30 \
        --local-pool data/eval_pools/gemini30_s20261006 --report data/selftrain/report_gemini30.json
    ... --disc data/selftrain/disc_val --real --report ...
    ... --disc D --images DIR --write-pool OUT        # unlabelled: write the pseudo-labelled local pool (stage C)
"""
import argparse
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import product  # noqa: E402

HI, LO, S_MIN, MIN_AREA = 0.9, 0.1, 3, 0.005   # chosen on Gemini30 + val labels (2026-11-14), never HF14
CFG = {}


def plain_share(img_bgr, boxes):
    """Share of the boxes that are blank paper: median colour within 12 Lab of the paper (as product.colour_cut) and < 3% of
    pixels inked (40 grey levels darker than the paper). Gemini30 / val (2026-11-14): families made of such boxes are blobs on
    empty paper, plain walls and room floors -- 50 / 105 unmatched families on Gemini30, 1 of the matched ones."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) * [100 / 255, 1, 1] - [0, 128, 128]
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hi = g >= np.percentile(g, 90)
    paper, pg = np.median(lab[hi], 0), np.median(g[hi])
    n = 0
    for x, y, w, h in boxes:
        dE = np.linalg.norm(np.median(lab[y:y + h, x:x + w].reshape(-1, 3), 0) - paper)
        n += dE <= 12 and (g[y:y + h, x:x + w] < pg - 40).mean() < 0.03
    return n / max(len(boxes), 1)


def family_layers(d, img_bgr, meta):
    """Each family's draft polygons and, for families eligible as pseudo-labels (`pl`: >= s_min agreeing boxes, not mostly
    blank-paper boxes, confident area >= min_area of the sheet), confident target / background masks."""
    h, w = meta["h"], meta["w"]
    out = []
    for k, f in enumerate(meta["families"]):
        maps = {z: np.load(d / "maps" / str(z) / f"{k:03d}.npz")["prob"] for z in product.SCALES}
        P = product.blend(maps, (h, w))
        polys = product.product_polygons(maps, img_bgr, tuple(f["box"]))
        draft = product.rasterize(polys, (h, w))
        pos = draft & (P > CFG["hi"])
        plain = plain_share(img_bgr, f["members"])
        pl = f["support"] >= CFG["s_min"] and plain < 0.5 and pos.mean() >= CFG.get("min_area", 0)
        out.append(dict(k=k, support=f["support"], box=f["box"], overlaps=f["overlaps"], polys=polys, draft=draft,
                        pos=pos, neg=(P < CFG["lo"]) & ~draft, plain=round(float(plain), 3), pl=pl))
    # a pixel confident for two eligible families is ignored for both
    elig = [o["pos"] for o in out if o["pl"]]
    if len(elig) >= 2:
        two = np.sum(elig, 0) >= 2
        for o in out:
            o["pos"] = o["pos"] & ~two
    return out


def human_targets(sid, a):
    """{category: (target bool, ignore bool | None)} as the evaluator builds them (label fixes included for real sheets)."""
    from evaluate_refunet_selection import _prepare_sheet
    if a.real:
        i = int(sid[4:])
        rec = {k: CFG["recs"][i][k] for k in ("image", "annotations")}
        fixes = CFG["fixes"].get(str(i), [])
    else:
        ann = json.loads((Path(a.local_pool) / "annotations" / f"{sid}.json").read_text())
        rec = {"image_path": str(Path(a.local_pool) / "images" / ann["image"]["file_name"]),
               "annotations": ann["annotations"], "ref_exclude": ann.get("ref_exclude")}
        fixes = []
    sh = _prepare_sheet((0, rec, 1024, 224, False, {}, fixes, False))
    h0, w0 = sh["h0"], sh["w0"]
    out = {}
    for c, (bits, n, ign) in sh["targets"].items():
        T = np.unpackbits(bits, count=h0 * w0).reshape(h0, w0).astype(bool)
        I = None if ign is None else np.unpackbits(ign, count=h0 * w0).reshape(h0, w0).astype(bool)
        out[c] = (T, I)
    return out


def load_img(sid, a):
    if a.real:
        data = CFG["recs"][int(sid[4:])]["image"]
    elif a.local_pool:
        ann = json.loads((Path(a.local_pool) / "annotations" / f"{sid}.json").read_text())
        data = (Path(a.local_pool) / "images" / ann["image"]["file_name"]).read_bytes()
    else:
        data = next(p for p in Path(a.images).iterdir() if p.stem == sid).read_bytes()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def score_sheet(args):
    d, a = args
    meta = json.loads((d / "families.json").read_text())
    sid = meta["sheet"]
    img = load_img(sid, a)
    fams = family_layers(d, img, meta)
    tg = human_targets(sid, a)
    allT = np.logical_or.reduce([T for T, _ in tg.values()]) if tg else np.zeros(img.shape[:2], bool)
    rows, merges = [], 0
    for f in fams:
        cover = [c for c, (T, _) in tg.items() if (f["draft"] & T).sum() >= 0.3 * max(T.sum(), 1)]
        f["merge"] = len(cover) >= 2
        merges += int(f["merge"])
    for c, (T, I) in tg.items():
        keep = ~I if I is not None else np.ones_like(T)
        best, bi = 0.0, None
        for f in fams:
            D = f["draft"] & keep
            iou = (D & T).sum() / max((D | T).sum(), 1)
            if iou > best:
                best, bi = iou, f
        r = dict(sheet=sid, category=c, target_px=int(T.sum()), draft_iou=round(float(best), 4), miss=bool(best < 0.5))
        if bi is not None and best >= 0.5 and bi["pl"]:
            pos, neg = bi["pos"] & keep, bi["neg"] & keep
            r.update(pl=True, support=bi["support"], pos_px=int(pos.sum()), pos_tp=int((pos & T).sum()),
                     neg_on_T=int((neg & T).sum()), ign_on_T=int((T & ~pos & ~neg).sum()))
        else:
            r.update(pl=False, support=None if bi is None else bi["support"])
        rows.append(r)
    # pseudo-label families that match no human family: where does their confident target land?
    extra = []
    for f in fams:
        if f["pl"] and not any(
                (f["draft"] & T).sum() / max((f["draft"] | T).sum(), 1) >= 0.5 for T, _ in tg.values()):
            extra.append(dict(support=f["support"], pos_px=int(f["pos"].sum()), on_labels=int((f["pos"] & allT).sum())))
    return dict(sheet=sid, n_fam=len(fams), n_pl=sum(f["pl"] for f in fams), merges=merges,
                rows=rows, extra=extra)


def mask_segs(m, min_px):
    """Instances of a bool mask as the dataset's segmentation ([outer, hole, ...] per instance, flat xy lists); components
    smaller than min_px are returned separately as a mask (the caller ignores them rather than calling them background)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool); small[1:] = st[1:, 4] < min_px
    keep = m & ~small[lab]
    cs, hier = cv2.findContours(keep.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    segs = []
    if hier is not None:
        idx = {}
        for j, hh in enumerate(hier[0]):
            if hh[3] == -1 and len(cs[j]) >= 3:
                idx[j] = len(segs); segs.append([cs[j].reshape(-1).astype(float).tolist()])
        for j, hh in enumerate(hier[0]):
            if hh[3] in idx and len(cs[j]) >= 3:
                segs[idx[hh[3]]].append(cs[j].reshape(-1).astype(float).tolist())
    return segs, m & small[lab]


def write_sheet(args):
    """Stage C, one unlabelled sheet -> one record of a local pool (images/ + annotations/)."""
    from refmask2former.dataset import IGNORE_PREFIX
    d, a = args
    meta = json.loads((d / "families.json").read_text())
    sid = meta["sheet"]
    src = next(p for p in Path(a.images).iterdir() if p.stem == sid)
    img = cv2.imdecode(np.frombuffer(src.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    fams = [f for f in family_layers(d, img, meta) if f["pl"]]
    h, w = img.shape[:2]
    min_px = max(16, int(2e-4 * h * w))
    anns, kept = [], []
    for f in fams:
        if f["pos"].sum() < 20 * min_px:          # too little confident area to teach anything
            continue
        cat = f"pattern{len(kept) + 1}"
        segs, specks = mask_segs(f["pos"], min_px)
        if not segs:
            continue
        ign = (~f["pos"] & ~f["neg"]) | specks
        anns += [{"segmentation": sg, "category_name": cat} for sg in segs]
        isegs, _ = mask_segs(ign, 1)
        anns += [{"segmentation": sg, "category_name": IGNORE_PREFIX + cat} for sg in isegs]
        kept.append(dict(cat=cat, support=f["support"], box=f["box"], pos_frac=round(float(f["pos"].mean()), 4),
                         ign_frac=round(float(ign.mean()), 4)))
    if not kept:
        return sid, None
    out = Path(a.write_pool)
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "annotations").mkdir(parents=True, exist_ok=True)
    name = f"{sid}.png"
    cv2.imwrite(str(out / "images" / name), img)
    (out / "annotations" / f"{sid}.json").write_text(json.dumps(
        {"image": {"file_name": name, "width": w, "height": h}, "mode": "selftrain", "annotations": anns,
         "selftrain": {"families": kept, "checkpoint": meta["checkpoint"], "params": dict(hi=CFG["hi"], lo=CFG["lo"],
                                                                                           s_min=CFG["s_min"])}}))
    return sid, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disc", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-pool"); src.add_argument("--real", action="store_true"); src.add_argument("--images")
    ap.add_argument("--report"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--hi", type=float, default=HI); ap.add_argument("--lo", type=float, default=LO)
    ap.add_argument("--s-min", type=int, default=S_MIN)
    ap.add_argument("--min-area", type=float, default=MIN_AREA, help="confident area (share of the sheet) for a pseudo family")
    ap.add_argument("--write-pool", help="with --images: write the pseudo-labelled local pool here (stage C)")
    ap.add_argument("--only", help="file of sheet stems to use (one per line)")
    a = ap.parse_args()
    CFG.update(hi=a.hi, lo=a.lo, s_min=a.s_min, min_area=a.min_area)
    if a.real:
        from evaluate_refunet_selection import load_parquet_records
        CFG["recs"] = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
        CFG["fixes"] = json.load(open("eval_labels/real_fixes_v4.json"))
    dirs = sorted(p for p in Path(a.disc).iterdir() if (p / "families.json").exists())
    if a.only:
        keep = set(Path(a.only).read_text().split())
        dirs = [p for p in dirs if p.name in keep]
    if a.write_pool:
        with Pool(a.workers) as p:
            res = p.map(write_sheet, [(d, a) for d in dirs])
        done = {sid: k for sid, k in res if k}
        summ = dict(sheets_in=len(dirs), sheets_written=len(done), families=sum(len(k) for k in done.values()),
                    mean_pos_frac=round(float(np.mean([f["pos_frac"] for k in done.values() for f in k])), 4) if done else 0,
                    mean_ign_frac=round(float(np.mean([f["ign_frac"] for k in done.values() for f in k])), 4) if done else 0,
                    params=dict(hi=a.hi, lo=a.lo, s_min=a.s_min, min_area=a.min_area))
        print(json.dumps(summ, indent=1))
        Path(a.write_pool, "selftrain_manifest.json").write_text(json.dumps(dict(summary=summ, sheets=done), indent=1))
        return
    with Pool(a.workers) as p:
        res = p.map(score_sheet, [(d, a) for d in dirs])
    rows = [r for s in res for r in s["rows"]]
    pl = [r for r in rows if r["pl"]]
    T = sum(r["target_px"] for r in pl)
    summ = dict(
        sheets=len(res), human_families=len(rows), families_found=sum(s["n_fam"] for s in res),
        pseudo_families=sum(s["n_pl"] for s in res), merges=sum(s["merges"] for s in res),
        draft_miou=round(float(np.mean([r["draft_iou"] for r in rows])), 4) if rows else None,
        draft_misses=sum(r["miss"] for r in rows),
        pl_covered_families=len(pl),
        pl_pos_precision=round(sum(r["pos_tp"] for r in pl) / max(sum(r["pos_px"] for r in pl), 1), 4),
        pl_pos_recall=round(sum(r["pos_tp"] for r in pl) / max(T, 1), 4),
        pl_neg_error=round(sum(r["neg_on_T"] for r in pl) / max(T, 1), 4),
        pl_ignored=round(sum(r["ign_on_T"] for r in pl) / max(T, 1), 4),
        unmatched_pl_families=sum(len(s["extra"]) for s in res),
        unmatched_pl_on_labels=round(sum(e["on_labels"] for s in res for e in s["extra"]) /
                                     max(sum(e["pos_px"] for s in res for e in s["extra"]), 1), 4),
        params=dict(hi=a.hi, lo=a.lo, s_min=a.s_min, min_area=a.min_area))
    print(json.dumps(summ, indent=1, default=int))
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(dict(summary=summ, sheets=res), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
