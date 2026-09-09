#!/usr/bin/env python3
"""Summarise the augmentation ablation across arms and seeds.

Two protocols, because they can disagree and only one of them is honest about
selection:

  best   the maximum over epochs of val median polygon IoU. This is what the
         recorded 0.853 used, and it is OPTIMISTIC: the epoch is chosen on the
         same 29 sheets the number is then quoted from, so it rewards whichever
         arm got a lucky epoch.
  last3  the mean over the final three epochs. Selects nothing, so it is the
         fair comparison between arms. Prefer it when the two disagree.

A gain is reported as real only when it holds at BOTH seeds and clears the
seed-to-seed spread -- the repo standard, and the reason the recorded result
still carries a "not yet replicated" note.

    python3 scripts/sam3_aug_report.py [--runs data/runs] [--metric poly_median]
"""
import argparse, glob, json, os
import numpy as np

ARM_ORDER = ["none", "light", "nodihedral", "dihedralonly", "nocrop",
             "geom", "photo", "croponly", "default", "strong"]


def load(runs):
    out = {}
    for h in sorted(glob.glob(f"{runs}/sam3_*/history.json")):
        name = os.path.basename(os.path.dirname(h))
        body = name[len("sam3_"):]
        if "_s" not in body:
            continue
        arm, seed = body.rsplit("_s", 1)
        try:
            hist = json.load(open(h))
        except Exception:
            continue
        ep = [e for e in hist if e["epoch"] >= 0]
        if ep:
            out[(arm, int(seed))] = ep
    return out


def stats(ep, metric):
    v = np.array([e[metric] for e in ep], float)
    return dict(best=float(v.max()), best_ep=int(np.argmax(v)),
                last3=float(v[-3:].mean()), n=len(v),
                ge80_best=float(ep[int(np.argmax(v))]["ge80"]),
                ge80_last3=float(np.mean([e["ge80"] for e in ep[-3:]])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--metric", default="poly_median",
                    choices=["poly_median", "poly_mean", "mask_mean"])
    ap.add_argument("--baseline", default="none")
    args = ap.parse_args()

    runs = load(args.runs)
    if not runs:
        print(f"no runs under {args.runs}"); return
    arms = sorted({a for a, _ in runs}, key=lambda a: (ARM_ORDER.index(a)
                  if a in ARM_ORDER else 99, a))
    seeds = sorted({s for _, s in runs})

    print(f"metric: {args.metric}   (val = 29 sheets / 98 instances)\n")
    hdr = f"{'arm':<13}" + "".join(f"  s{s:<2} best  s{s:<2} last3" for s in seeds) \
          + f"  {'best mean':>10} {'last3 mean':>11}  {'>=0.8':>7}"
    print(hdr); print("-" * len(hdr))

    table = {}
    for arm in arms:
        row, bests, lasts, g80 = f"{arm:<13}", [], [], []
        for s in seeds:
            ep = runs.get((arm, s))
            if ep is None:
                row += f"  {'--':>8} {'--':>9}"; continue
            st = stats(ep, args.metric)
            row += f"  {st['best']:8.4f} {st['last3']:9.4f}"
            bests.append(st["best"]); lasts.append(st["last3"])
            g80.append(st["ge80_last3"])
        if bests:
            row += f"  {np.mean(bests):10.4f} {np.mean(lasts):11.4f}  {np.mean(g80):6.1%}"
            table[arm] = (np.mean(bests), np.mean(lasts), bests, lasts)
        print(row)

    if args.baseline not in table:
        return
    b_best, b_last, b_bl, b_ll = table[args.baseline]
    spread = (np.std([v for _, _, bl, _ in table.values() for v in bl])
              if len(seeds) > 1 else 0.0)
    print(f"\nvs '{args.baseline}'   (seed spread across all arms: sd {spread:.4f})")
    for arm, (bb, ll, bl, lls) in table.items():
        if arm == args.baseline:
            continue
        d_last = ll - b_last
        both = (len(lls) == len(b_ll) and len(lls) > 1
                and all(x > y for x, y in zip(lls, b_ll)))
        verdict = ("consistent" if both and abs(d_last) > 2 * spread else
                   "positive at both seeds" if both else
                   "mixed" if abs(d_last) > spread else "null")
        print(f"  {arm:<13} best {bb - b_best:+.4f}   last3 {d_last:+.4f}   {verdict}")
    print("\n'consistent' = positive at every seed and >2x the seed spread; "
          "anything less needs another seed before it is quoted.")


if __name__ == "__main__":
    main()
