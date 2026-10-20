"""Deleting wrong polygons (the user's most common edit, scripts/editcost.py). One row per polygon of the default output
(scripts/polygonize.py, rasterised; polygon = 8-connected component of the rasterised mask), with features the deployed system
has (probability map, drawing, the user's reference box; not other questions' maps) and the label wrong = < 10% of it on the target.
  extract SPLIT   -> data/diag/polys_SPLIT.json   (features + label + pixels on / off target, per-question totals)
  eval            -> cross-fit: train Gemini30+val -> HF14, Gemini30+HF14 -> val, 5-fold sheet-grouped CV on Gemini30;
                     delete when P(wrong) > 0.5 (fixed in advance); IoU after deleting vs the default."""
import sys, json
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts')
import common
from common import jobs, load
from polygonize import probs_to_polygons, rasterize
common.WITH_BOX[0] = True
FEATS = ["frac", "area_sheet", "rank", "ncomp", "pmax", "pmean", "p90", "f80", "f50", "pref", "dref", "has_ref",
         "dmain", "is_main", "lab_ref", "lab_main", "bh_ratio_ref", "bh_ratio_main", "ori_ref", "ori_main", "comp_fill"]


def ori_hist(g, m):
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)[m]; ang = (np.arctan2(gy, gx)[m] % np.pi)
    h = np.bincount(np.minimum((ang / np.pi * 8).astype(int), 7), weights=mag, minlength=8)
    return h / max(h.sum(), 1e-6)


def chi2(a, b): return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))


def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    bh = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)
    diag = float(np.hypot(h0, w0)); out = []
    for ref, cat, G, p, box in qs:
        M = rasterize(probs_to_polygons(p, (h0, w0)), (h0, w0))
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        x, y, w, h = [int(round(v)) for v in box]
        rb = np.zeros((h0, w0), bool); rb[max(y, 0):y + h, max(x, 0):x + w] = True
        q = {"i": idx, "q": ref, "tgt": int(G.sum()), "inter": int((M & G).sum()), "pred": int(M.sum()), "polys": []}
        n, labc, st, _ = cv2.connectedComponentsWithStats(M.astype(np.uint8), connectivity=8)
        if n <= 1: out.append(q); continue
        order = np.argsort(-st[1:, 4]) + 1; main = order[0]
        rlab = lab[rb].mean(0); rbh = bh[rb].mean() + 1e-3; rori = ori_hist(g, rb)
        mm = labc == main; mlab = lab[mm].mean(0); mbh = bh[mm].mean() + 1e-3; mori = ori_hist(g, mm)
        dmain_map = cv2.distanceTransform((~mm).astype(np.uint8), cv2.DIST_L2, 3)
        dref_map = cv2.distanceTransform((~rb).astype(np.uint8), cv2.DIST_L2, 3)
        tot = st[1:, 4].sum()
        for rank, j in enumerate(order):
            cm = labc == j; a = int(st[j, 4]); pv = P[cm]
            xx, yy, ww, hh = st[j, :4]
            f = dict(frac=a / tot, area_sheet=a / (h0 * w0), rank=rank, ncomp=n - 1, pmax=float(pv.max()), pmean=float(pv.mean()),
                     p90=float(np.percentile(pv, 90)), f80=float((pv > .8).mean()), f50=float((pv > .5).mean()),
                     pref=float(P[rb].mean()), dref=float(dref_map[cm].min()) / diag, has_ref=float((cm & rb).any()),
                     dmain=float(dmain_map[cm].min()) / diag, is_main=float(j == main),
                     lab_ref=float(np.linalg.norm(lab[cm].mean(0) - rlab)), lab_main=float(np.linalg.norm(lab[cm].mean(0) - mlab)),
                     bh_ratio_ref=float(np.log((bh[cm].mean() + 1e-3) / rbh)), bh_ratio_main=float(np.log((bh[cm].mean() + 1e-3) / mbh)),
                     ori_ref=chi2(ori_hist(g, cm), rori), ori_main=chi2(ori_hist(g, cm), mori), comp_fill=a / max(ww * hh, 1))
            on = int((cm & G).sum())
            q["polys"].append({"f": f, "area": a, "on": on, "wrong": on < 0.1 * a})
        out.append(q)
    print(idx, flush=True, file=sys.stderr)
    return out


def iou_after(q, delete):
    I, Pn = q["inter"], q["pred"]
    for pl, d in zip(q["polys"], delete):
        if d: I -= pl["on"]; Pn -= pl["area"]
    return I / max(Pn + q["tgt"] - I, 1)


def XY(qs):
    X = [[pl["f"][k] for k in FEATS] for q in qs for pl in q["polys"]]
    Y = [pl["wrong"] for q in qs for pl in q["polys"]]
    return np.array(X, float), np.array(Y, bool)


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)


def evaluate(train, test, name, thr=0.5):
    X, Y = XY(train); clf = model().fit(X, Y)
    base, new, nd, ndw, lost_on = [], [], 0, 0, 0
    for q in test:
        if not q["polys"]: b = q["inter"] / max(q["pred"] + q["tgt"] - q["inter"], 1); base.append(b); new.append(b); continue
        Xq, _ = XY([q]); pw = clf.predict_proba(Xq)[:, 1]
        dele = [(p > thr) and not pl["f"]["is_main"] for p, pl in zip(pw, q["polys"])]   # never delete the main polygon
        base.append(iou_after(q, [False] * len(dele))); new.append(iou_after(q, dele))
        nd += sum(dele); ndw += sum(d and pl["wrong"] for d, pl in zip(dele, q["polys"]))
    d = np.array(new) - np.array(base)
    from scipy.stats import wilcoxon
    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    nwrong = sum(pl["wrong"] for q in test for pl in q["polys"])
    print(f"{name}: IoU {np.mean(base):.4f} -> {np.mean(new):.4f} ({d.mean():+.4f}; up {(d > .005).sum()} / down {(d < -.005).sum()}, "
          f"p={p:.3f}); deleted {nd} polygons, {ndw} truly wrong ({ndw / max(nd, 1):.0%} precision; {ndw}/{nwrong} of the wrong ones)")
    return clf


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        for split in sys.argv[2:]:
            with Pool(2) as pool: qs = [q for qq in pool.imap(sheet, jobs(split)) for q in qq]
            json.dump(qs, open(f"data/diag/polys_{split}.json", "w"))
            print(split, len(qs), "questions", sum(len(q["polys"]) for q in qs), "polygons",
                  sum(pl["wrong"] for q in qs for pl in q["polys"]), "wrong")
    else:
        D = {s: json.load(open(f"data/diag/polys_{s}.json")) for s in ("gemini30", "val", "hf14")}
        evaluate(D["gemini30"] + D["val"], D["hf14"], "train Gemini30+val -> HF14")
        evaluate(D["gemini30"] + D["hf14"], D["val"], "train Gemini30+HF14 -> val")
        sheets = sorted({q["i"] for q in D["gemini30"]})
        for k in range(5):
            te = set(sheets[k::5])
            evaluate([q for q in D["gemini30"] if q["i"] not in te] + D["val"] + D["hf14"], [q for q in D["gemini30"] if q["i"] in te],
                     f"train Gemini30 (other folds)+val+HF14 -> Gemini30 fold {k}")
