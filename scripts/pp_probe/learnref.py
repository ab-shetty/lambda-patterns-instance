"""Learned boundary refinement from classical features (CPU, gradient-boosted trees). Trained on Gemini30 only.
Every pixel within B native px of the default mask's edge (lin_fill) is described by local features and classified in / out;
pixels outside the band keep lin_fill. Features (native px):
  probability: P, P smoothed (s 2, 6), |grad P|, Laplacian of P, signed distance to lin_fill's edge
  drawing: grey, |grad grey|, black-hat (7, 15), ink v2, distance to ink, ink density (9)
  local inside/outside contrast (a per-pixel GrabCut): the Lab colour and the black-hat texture density (15) of the confident inside
  (P > 0.8) and confident outside (P < 0.2) averaged over 21 and 61 px windows; features |c - in|, |c - out| and their difference
Usage: learnref.py extract SPLIT   -> data/diag/lref/SPLIT_<sheet>.npz (band pixels: features, label, question id) + SPLIT_meta.json
       learnref.py cv              -> 5-fold sheet-grouped CV on Gemini30 (IoU recomposed from the band; no polygons)
       learnref.py apply SPLIT     -> train on all Gemini30, score SPLIT once as polygons"""
import sys, json, os
from pathlib import Path
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load, ink_map_v2
from methods import fill_holes, polygons, iou
T, B = 0.35, 4
OUT = Path("data/diag/lref")
NAMES = ["P", "Ps2", "Ps6", "gradP", "lapP", "sd", "grey", "gradg", "bh7", "bh15", "ink", "dink", "inkd",
         "c21in", "c21out", "c21d", "c61in", "c61out", "c61d", "t21in", "t21out", "t21d", "t61in", "t61out", "t61d"]


def sheet_feats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); gf = g.astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bh7 = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)
    bh15 = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(np.float32)
    ink = ink_map_v2(img)
    dink = np.minimum(cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5), 10)
    gx, gy = cv2.Sobel(gf, cv2.CV_32F, 1, 0), cv2.Sobel(gf, cv2.CV_32F, 0, 1)
    return dict(g=gf, lab=lab, bh7=bh7, bh15=bh15, ink=ink.astype(np.float32), dink=dink,
                inkd=cv2.blur(ink.astype(np.float32), (9, 9)), gradg=np.hypot(gx, gy), tex=cv2.blur(bh7, (15, 15)))


def q_feats(F, P, lf, band, y1, y2, x1, x2):
    sl = (slice(y1, y2), slice(x1, x2)); bb = band[sl]
    Pc = P[sl]; lfc = lf[sl]
    Ps2, Ps6 = cv2.GaussianBlur(Pc, (0, 0), 2), cv2.GaussianBlur(Pc, (0, 0), 6)
    P1 = cv2.GaussianBlur(Pc, (0, 0), 1)
    gP = np.hypot(cv2.Sobel(P1, cv2.CV_32F, 1, 0), cv2.Sobel(P1, cv2.CV_32F, 0, 1))
    lP = cv2.Laplacian(Ps2, cv2.CV_32F)
    din = cv2.distanceTransform(lfc.astype(np.uint8), cv2.DIST_L2, 5); dout = cv2.distanceTransform((~lfc).astype(np.uint8), cv2.DIST_L2, 5)
    sd = np.clip(din - dout, -B - 1, B + 1)
    cols = [Pc, Ps2, Ps6, gP, lP, sd] + [F[k][sl] for k in ("g", "gradg", "bh7", "bh15", "ink", "dink", "inkd")]
    cin, cout = (Pc > 0.8).astype(np.float32), (Pc < 0.2).astype(np.float32)
    lab, tex = F["lab"][sl], F["tex"][sl]
    for kind, X in (("c", lab), ("t", tex[..., None])):
        for w in (21, 61):
            nin, nout = cv2.blur(cin, (w, w)) + 1e-6, cv2.blur(cout, (w, w)) + 1e-6
            mi = np.dstack([cv2.blur(X[..., c] * cin, (w, w)) / nin for c in range(X.shape[2])])
            mo = np.dstack([cv2.blur(X[..., c] * cout, (w, w)) / nout for c in range(X.shape[2])])
            di, do = np.linalg.norm(X - mi, axis=2), np.linalg.norm(X - mo, axis=2)
            di[nin < 1e-3] = 255; do[nout < 1e-3] = 255
            cols += [di, do, di - do]
    return np.stack([c[bb] for c in cols], 1).astype(np.float16)


def prep(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]
    F = sheet_feats(img); X, Y, Q, meta = [], [], [], []
    for qi, (ref, cat, tgt, p) in enumerate(qs):
        s = p.shape[1] / w0; ha = 4096 / s / s
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lf = fill_holes(P > T, ha)
        din = cv2.distanceTransform(lf.astype(np.uint8), cv2.DIST_L2, 5); dout = cv2.distanceTransform((~lf).astype(np.uint8), cv2.DIST_L2, 5)
        band = (din <= B) & (dout <= B)
        ys, xs = np.nonzero(band)
        m = dict(i=idx, q=ref, tgt=int(tgt.sum()), inter_out=int((lf & tgt & ~band).sum()), pred_out=int((lf & ~band).sum()),
                 ref_poly=iou(polygons(lf)[0], tgt), base=iou(cv2.resize((p > T * 255).astype(np.uint8), (w0, h0),
                 interpolation=cv2.INTER_NEAREST).astype(bool), tgt), nb=int(len(ys)))
        meta.append(m)
        if not len(ys): continue
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        X.append(q_feats(F, P, lf, band, y1, y2, x1, x2)); Y.append(tgt[y1:y2, x1:x2][band[y1:y2, x1:x2]]); Q.append(np.full(len(ys), qi, np.int16))
    if X:
        np.savez(OUT / f"{idx:03d}_{SPLIT[0]}.npz", X=np.concatenate(X), Y=np.concatenate(Y), Q=np.concatenate(Q))
    return meta


def apply_sheet(job):
    import joblib
    clf = joblib.load(OUT / "clf_gemini30.joblib")
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]; F = sheet_feats(img); rows = []
    for ref, cat, tgt, p in qs:
        s = p.shape[1] / w0; ha = 4096 / s / s
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lf = fill_holes(P > T, ha)
        din = cv2.distanceTransform(lf.astype(np.uint8), cv2.DIST_L2, 5); dout = cv2.distanceTransform((~lf).astype(np.uint8), cv2.DIST_L2, 5)
        band = (din <= B) & (dout <= B); ys, xs = np.nonzero(band); m = lf.copy()
        if len(ys):
            y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            X = q_feats(F, P, lf, band, y1, y2, x1, x2).astype(np.float32)
            sub = m[y1:y2, x1:x2]; sub[band[y1:y2, x1:x2]] = clf.predict_proba(X)[:, 1] > 0.5
        rows.append(dict(i=idx, q=ref, base=iou(cv2.resize((p > T * 255).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool), tgt),
                         ref=iou(polygons(lf)[0], tgt), learn=iou(polygons(fill_holes(m, ha))[0], tgt)))
    print(idx, flush=True, file=sys.stderr)
    return rows


def band_ious(meta, pred_by_q):
    """meta: list per question; pred_by_q: {qi: (pred bool, label bool)} -> per-question IoU recomposed."""
    out = []
    for qi, m in enumerate(meta):
        yp, yt = pred_by_q.get(qi, (np.zeros(0, bool), np.zeros(0, bool)))
        I = m["inter_out"] + int((yp & yt).sum()); Pn = m["pred_out"] + int(yp.sum())
        out.append(I / max(Pn + m["tgt"] - I, 1))
    return out


def load_split(split):
    meta = json.load(open(OUT / f"{split}_meta.json")); data = {}
    for f in sorted(OUT.glob(f"*_{split}.npz")):
        z = np.load(f); data[int(f.name[:3])] = (z["X"].astype(np.float32), z["Y"], z["Q"])
    return meta, data


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0)


def fit(data, sheets, per_sheet=150000, seed=0):
    rng = np.random.default_rng(seed); Xs, Ys = [], []
    for i in sheets:
        X, Y, _ = data[i]; k = rng.choice(len(Y), min(per_sheet, len(Y)), replace=False); Xs.append(X[k]); Ys.append(Y[k])
    return model().fit(np.concatenate(Xs), np.concatenate(Ys))


def score(meta, data, clf, sheets):
    rows = []
    for i in sheets:
        mi = [m for m in meta if m["i"] == i]
        if i in data:
            X, Y, Q = data[i]; yp = clf.predict_proba(X)[:, 1] > 0.5
            by = {qi: (yp[Q == qi], Y[Q == qi]) for qi in np.unique(Q)}
        else: by = {}
        for m, v in zip(mi, band_ious(mi, by)): rows.append(dict(i=i, q=m["q"], base=m["base"], ref_poly=m["ref_poly"], learn=v))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "extract":
        SPLIT = [sys.argv[2]]
        with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as pool: meta = [m for ms in pool.imap(prep, jobs(SPLIT[0])) for m in ms]
        json.dump(meta, open(OUT / f"{SPLIT[0]}_meta.json", "w")); print("extracted", len(meta))
    elif cmd == "cv":
        meta, data = load_split("gemini30"); sheets = sorted({m["i"] for m in meta})
        # the lin_fill reference recomposed the same way (band pixels = lin_fill's own decision: sd >= 0)
        ref = {}
        for i in sheets:
            mi = [m for m in meta if m["i"] == i]
            if i in data:
                X, Y, Q = data[i]; by = {qi: (X[Q == qi, NAMES.index("sd")] >= 0, Y[Q == qi]) for qi in np.unique(Q)}
            else: by = {}
            for m, v in zip(mi, band_ious(mi, by)): ref[(i, m["q"])] = v
        folds = [sheets[k::5] for k in range(5)]; rows = []
        for k, te in enumerate(folds):
            clf = fit(data, [s for s in sheets if s not in te and s in data])
            for r in score(meta, data, clf, te): r["ref"] = ref[(r["i"], r["q"])]; rows.append(r)
            print("fold", k, flush=True)
        json.dump(rows, open(OUT / "cv_gemini30.json", "w"))
        d = np.array([r["learn"] - r["ref"] for r in rows])
        by = {}
        for r in rows: by.setdefault(r["i"], []).append(r["learn"] - r["ref"])
        sd = np.array([np.mean(v) for v in by.values()])
        print(f"Gemini30 5-fold sheet CV: learned band vs lin_fill {d.mean():+.4f} (q up {(d > .005).sum()} / down {(d < -.005).sum()}, worst {d.min():+.3f}); "
              f"per-sheet {sd.mean():+.4f}, sheets down >0.005: {(sd < -.005).sum()}, worst {sd.min():+.3f}; ref {np.mean([r['ref'] for r in rows]):.4f}")
        clf = fit(data, [s for s in sheets if s in data])
        import joblib; joblib.dump(clf, OUT / "clf_gemini30.joblib")
        from sklearn.inspection import permutation_importance
        X, Y, _ = data[sheets[0]]; k = np.random.default_rng(1).choice(len(Y), min(40000, len(Y)), replace=False)
        imp = permutation_importance(clf, X[k], Y[k], n_repeats=2, random_state=0).importances_mean
        print("importance:", ", ".join(f"{n} {v:.3f}" for n, v in sorted(zip(NAMES, imp), key=lambda t: -t[1])[:10]))
    elif cmd == "apply":
        split = sys.argv[2]
        with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as pool: rows = [r for rs in pool.imap(apply_sheet, jobs(split)) for r in rs]
        json.dump(rows, open(f"data/diag/rows_learnref_{split}.json", "w"))
        by = {}
        for r in rows: by.setdefault(r["i"], []).append(r)
        for k in ("base", "ref", "learn"):
            d = np.array([r[k] - r["base"] for r in rows]); sd = np.array([np.mean([r[k] - r["base"] for r in v]) for v in by.values()])
            print(f"{split} {k:5s} q-mean {np.mean([r[k] for r in rows]):.4f} ({d.mean():+.4f} vs evaluator; per-sheet {sd.mean():+.4f}; "
                  f"sheets down >0.005 {(sd < -.005).sum()}, worst sheet {sd.min():+.3f})")
