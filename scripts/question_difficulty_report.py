#!/usr/bin/env python3
"""Summarise question_difficulty.py outputs: does difficulty explain the gap?

For each file: synthetic vs HF14 means; how the two question sets differ on
each feature; IoU by bucket; the synthetic mean REWEIGHTED to HF14's mix of
(n_fams x lookalike) buckets; and an OLS of IoU on the features plus is_real,
with standard errors clustered by sheet (HF14 has only 14 sheets, so its
questions are not independent).

    python3 scripts/question_difficulty_report.py data/evaluations/qd_*.json
"""
import json
import sys

import numpy as np


def ols_cluster(X, y, groups):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    k = X.shape[1]
    c = G / max(G - 1, 1) * (len(y) - 1) / max(len(y) - k, 1)
    return beta, np.sqrt(np.diag(c * XtX_inv @ meat @ XtX_inv))


for path in sys.argv[1:]:
    rows = json.load(open(path))["rows"]
    real = np.array([r["is_real"] for r in rows])
    iou = np.array([r["iou"] for r in rows])
    nf = np.array([r["n_fams"] for r in rows])
    nr = np.array([r["n_regions"] for r in rows])
    tf = np.array([r["target_frac"] for r in rows])
    la = np.array([r["lookalike"] for r in rows])
    S, R = real == 0, real == 1
    print(f"\n=== {path.split('/')[-1]}")
    print(f"  mean IoU   synthetic {iou[S].mean():.4f} (n={S.sum()})   HF14 {iou[R].mean():.4f} (n={R.sum()})"
          f"   ratio {iou[R].mean() / iou[S].mean():.3f}")
    print(f"  question mix        synthetic         HF14")
    print(f"    multi-family      {np.mean(nf[S] > 1):8.2f}      {np.mean(nf[R] > 1):8.2f}")
    print(f"    lookalike (med)   {np.median(la[S]):8.3f}      {np.median(la[R]):8.3f}")
    print(f"    regions (med)     {np.median(nr[S]):8.1f}      {np.median(nr[R]):8.1f}")
    print(f"    target_frac (med) {np.median(tf[S]):8.4f}      {np.median(tf[R]):8.4f}")
    # buckets
    la_cut = np.quantile(la[nf > 1], [1 / 3, 2 / 3]) if (nf > 1).any() else [0, 0]
    def lab(i):
        if nf[i] == 1:
            return "single-family"
        t = "low" if la[i] < la_cut[0] else ("mid" if la[i] < la_cut[1] else "high")
        return f"multi, lookalike {t}"
    labels = np.array([lab(i) for i in range(len(rows))])
    print(f"  IoU by bucket (lookalike cuts {la_cut[0]:.3f}/{la_cut[1]:.3f})   synthetic        HF14")
    pred, wsum = 0.0, 0
    for b in ["single-family", "multi, lookalike low", "multi, lookalike mid", "multi, lookalike high"]:
        ms, mr = S & (labels == b), R & (labels == b)
        s_ = f"{iou[ms].mean():.3f} (n={ms.sum():4d})" if ms.any() else "   -          "
        r_ = f"{iou[mr].mean():.3f} (n={mr.sum():2d})" if mr.any() else "   -"
        print(f"    {b:24s}              {s_}   {r_}")
        if mr.any():
            pred += mr.sum() * (iou[ms].mean() if ms.any() else iou[S].mean())
            wsum += mr.sum()
    pred /= max(wsum, 1)
    gap = iou[S].mean() - iou[R].mean()
    print(f"  synthetic reweighted to HF14's bucket mix: {pred:.4f}  -> explains "
          f"{(iou[S].mean() - pred) / gap * 100 if gap else 0:.0f}% of the {gap:.4f} gap")
    # regression
    X = np.column_stack([np.ones(len(rows)), real, la, (nf > 1).astype(float),
                         np.log(tf + 1e-6), np.log(nr)])
    groups = np.array([f"{r['src']}_{r['image']}" for r in rows])
    b, se = ols_cluster(X, iou, groups)
    names = ["const", "is_real", "lookalike", "multi_family", "log target_frac", "log n_regions"]
    print("  OLS iou ~ features + is_real (SE clustered by sheet)")
    for n, bb, ss in zip(names, b, se):
        print(f"    {n:16s} {bb:+.4f}  se {ss:.4f}  t {bb / ss if ss else 0:+.2f}")
    print(f"  raw gap {-gap:+.4f}  vs  is_real after difficulty {b[1]:+.4f}")
