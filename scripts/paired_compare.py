#!/usr/bin/env python3
"""Compare two arms PAIRED per selection, not as means with a seed spread.

`startup.md` establishes that the 52 evaluation reference boxes are
byte-identical across every run, epoch and training set, so the eval is 52
FIXED questions. Question difficulty dominates the variance -- image 14's
selections sit near 0.05 while image 24's sit near 0.90 -- so comparing two
arms by their means throws away the pairing and needs many seeds to see
anything. Pairing removes that variance entirely.

This is the test the repo has been recommending to itself and never using.

    python3 scripts/paired_compare.py \
        --a data/runs/ck_ab_unet_res2048_seed7/metrics_epoch_8.json \
        --b data/runs/ck_ab_swin_t_res2048_seed7/metrics_epoch_8.json \
        --label-a resnet50 --label-b swin_t

Accepts either a per-epoch `metrics_epoch_*.json` from training or the
`--metrics-out` JSON from `evaluate_refunet_selection.py`. Reports the paired
mean difference, a paired t, the sign test (distribution-free, in case a
couple of selections carry everything), and the per-image breakdown, which is
where this repo's real findings have tended to live.
"""
import argparse
import json
import math
import os
from collections import defaultdict


def load_rows(path):
    d = json.load(open(path))
    rows = d.get("selections", d.get("rows", d))
    out = {}
    for r in rows:
        key = (r.get("image_index", r.get("image")), r["reference_instance"]
               if "reference_instance" in r else r.get("ref"))
        out[key] = float(r["iou"])
    return out, d


def t_cdf_upper(t, df):
    """P(T > |t|) for Student's t, via the incomplete beta -- no scipy."""
    x = df / (df + t * t)
    def betacf(a, b, x, it=200):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        h = d
        for m in range(1, it):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            de = d * c; h *= de
            if abs(de - 1.0) < 3e-7:
                break
        return h
    a, b = df / 2.0, 0.5
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    if x <= 0:
        ib = 0.0
    elif x >= 1:
        ib = 1.0
    else:
        front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
        ib = front * betacf(a, b, x) if x < (a + 1) / (a + b + 2) else \
            1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) / b \
            * betacf(b, a, 1 - x)
    return 0.5 * ib


def sign_test(n_pos, n_neg):
    """Two-sided exact binomial p at rate 0.5."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--min-delta", type=float, default=0.01,
                    help="ignore differences under this when counting wins")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    A, da = load_rows(args.a)
    B, db = load_rows(args.b)
    keys = sorted(set(A) & set(B), key=lambda k: (str(k[0]), k[1]))
    if not keys:
        raise SystemExit("no selections in common -- are these the same eval set?")
    if len(keys) != len(A) or len(keys) != len(B):
        print(f"  note: {len(A)} vs {len(B)} selections, comparing {len(keys)} shared")

    diffs = [B[k] - A[k] for k in keys]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if var else 0.0
    t = mean / se if se else float("inf") if mean else 0.0
    p = 2 * t_cdf_upper(abs(t), n - 1) if se else float("nan")
    wins = sum(1 for d in diffs if d > args.min_delta)
    losses = sum(1 for d in diffs if d < -args.min_delta)
    ties = n - wins - losses

    print(f"== paired over {n} fixed selections: {args.label_b} - {args.label_a}")
    for label, d in ((args.label_a, da), (args.label_b, db)):
        m = d.get("real_mean_iou")
        if m is not None:
            print(f"   {label:12s} mean {m:.4f}"
                  + (f"  (epoch {d['actual_epoch']})" if "actual_epoch" in d else ""))
    print(f"   paired mean difference {mean:+.4f}   se {se:.4f}   "
          f"t({n-1}) = {t:+.2f}   p = {p:.4g}")
    print(f"   {args.label_b} better on {wins}, worse on {losses}, "
          f"tied (|d|<{args.min_delta}) on {ties}"
          f"   sign-test p = {sign_test(wins, losses):.4g}")

    per = defaultdict(list)
    for k, d in zip(keys, diffs):
        per[k[0]].append(d)
    print(f"\n   per image ({args.label_b} - {args.label_a}), worst first:")
    for img, ds in sorted(per.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        m = sum(ds) / len(ds)
        # Contribution to the overall mean difference, which is what decides
        # whether one image is carrying the whole result.
        print(f"     image {str(img):>22s}  n={len(ds):2d}  mean {m:+.4f}"
              f"   share of total {sum(ds)/ (mean*n) * 100 if mean else 0:6.1f}%")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"a": args.a, "b": args.b, "n": n, "mean_diff": mean,
                   "se": se, "t": t, "p": p, "wins": wins, "losses": losses,
                   "per_selection": {str(k): d for k, d in zip(keys, diffs)}},
                  open(args.out, "w"), indent=1)
        print(f"   wrote {args.out}")


if __name__ == "__main__":
    main()
