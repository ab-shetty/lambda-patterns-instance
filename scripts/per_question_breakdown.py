"""Per-question comparison of two evaluate_refunet_selection.py outputs,
broken down by sheet type (hand-tagged from the 28 real sheets), box size at
the model input, families on the sheet, and over/under-selection.

  python3 scripts/per_question_breakdown.py --a A.json --b B.json [--label-a v6d --label-b revit]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

# Hand-tagged 2026-09-28 from thumbnails of the real-world-test parquet.
#   elev_render: Revit/CAD shaded colour elevations (Las Huertas set)
#   elev_line:   line-only CAD elevations
#   elev_markup: elevations with hand/PDF colour markup fills (lavender, green, magenta)
#   plan:        floor / framing / deck plans
#   roof_plan:   roof plans
SHEET_TYPE = {
    0: "plan", 7: "plan", 11: "plan", 12: "plan",
    1: "roof_plan", 15: "roof_plan", 16: "roof_plan",
    2: "elev_render", 3: "elev_render", 4: "elev_render", 5: "elev_render", 6: "elev_render",
    8: "elev_line", 9: "elev_line", 10: "elev_line", 13: "elev_line", 14: "elev_line",
    17: "elev_markup", 18: "elev_markup", 19: "elev_markup", 20: "elev_markup",
    21: "elev_markup", 22: "elev_markup", 23: "elev_markup", 24: "elev_markup",
    25: "elev_markup", 26: "elev_markup", 27: "elev_markup",
}


def load(path):
    d = json.load(open(path))
    return d, {(r["image_index"], r["reference_instance"]): r for r in d["selections"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--dims", default=None,
                   help="optional JSON {image_index: [w, h]} for box size at the model input")
    args = p.parse_args()
    da, A = load(args.a)
    db, B = load(args.b)
    size = da["image_max_size"]
    dims = {int(k): v for k, v in json.load(open(args.dims)).items()} if args.dims else {}
    keys = sorted(set(A) & set(B))
    fams = defaultdict(set)
    for k in keys:
        fams[k[0]].add(A[k]["category"])

    def bucket_rows(fn):
        g = defaultdict(list)
        for k in keys:
            g[fn(k)].append(k)
        return g

    def box_bucket(k):
        if k[0] not in dims:
            return "?"
        w, h = dims[k[0]]
        bx = A[k]["reference_box_native"]
        side = min(bx[2], bx[3]) * size / max(w, h)
        return "<32" if side < 32 else "32-64" if side < 64 else "64-128" if side < 128 else ">=128"

    def sel_bucket(k):
        r = A[k]
        ratio = r["prediction_pixels"] / max(r["target_pixels"], 1)
        return f"{args.label_a} over(>1.3x)" if ratio > 1.3 else f"{args.label_a} under(<0.7x)" if ratio < 0.7 else f"{args.label_a} ~1x"

    la, lb = args.label_a, args.label_b
    tot = sum(B[k]["iou"] - A[k]["iou"] for k in keys)
    print(f"n={len(keys)}  {la} {np.mean([A[k]['iou'] for k in keys]):.4f}  "
          f"{lb} {np.mean([B[k]['iou'] for k in keys]):.4f}  delta {tot/len(keys):+.4f}")
    for name, fn in [("sheet type", lambda k: SHEET_TYPE.get(k[0], "?")),
                     ("sheet", lambda k: f"#{k[0]:02d} {SHEET_TYPE.get(k[0], '?')}"),
                     ("box side @input", box_bucket),
                     ("families on sheet", lambda k: str(min(len(fams[k[0]]), 3)) + ("+" if len(fams[k[0]]) >= 3 else "")),
                     (f"{la} selection size", sel_bucket)]:
        print(f"\n== {name}")
        print(f"  {'':24s} {'n':>3s} {la:>7s} {lb:>7s} {'delta':>7s} {'share':>6s} {'b>a':>4s} {'b<a':>4s}")
        for g, ks in sorted(bucket_rows(fn).items()):
            a = np.array([A[k]["iou"] for k in ks]); b = np.array([B[k]["iou"] for k in ks])
            d = b - a
            share = d.sum() / tot if abs(tot) > 1e-9 else float("nan")
            print(f"  {g:24s} {len(ks):3d} {a.mean():7.3f} {b.mean():7.3f} {d.mean():+7.3f} "
                  f"{share:6.0%} {(d > 0.02).sum():4d} {(d < -0.02).sum():4d}")


if __name__ == "__main__":
    main()
