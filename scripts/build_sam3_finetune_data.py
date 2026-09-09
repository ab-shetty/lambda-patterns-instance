#!/usr/bin/env python3
"""Build a SAM 3 PVS fine-tuning manifest from the labelled real-ish plans.

Task being trained: box prompt -> mask of the region that box encloses. This is
the labelling-assist path (`scripts/sam3_box_to_polygon.py`), not the product
task -- the product task is one selection -> every matching region, which stays
with RefUNet.

Target convention: the OUTER ring, holes filled. Windows punched out of a wall
are cut afterwards as separate `remove` polygons in the Roboflow pipeline, so a
model that fills them is doing the right thing here.

Split is by SOURCE IMAGE, never by instance: two instances from one plan share a
drawing style, so an instance-level split leaks. HF14 lives in a separate HF
dataset and is not among these pools, so nothing here touches the acceptance set.

    python3 scripts/build_sam3_finetune_data.py --out data/sam3_ft
"""
import argparse, json, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask

POOLS = ["data/roboflow/floz-real-pool-v2-clean",
         "data/roboflow/floz-genreal-v1-clean",
         "data/roboflow/floz-gen-gemini-r23-clean"]


def load_pool(pool, min_area):
    """Read one local-data pool (images/ + annotations/) into scene dicts."""
    out = []
    for f in sorted(glob.glob(f"{pool}/annotations/*.json")):
        d = json.load(open(f))
        img = os.path.join(pool, "images", d["image"]["file_name"])
        if not os.path.exists(img):
            continue
        H, W = d["image"]["height"], d["image"]["width"]
        insts = []
        for a in d["annotations"]:
            seg = a["segmentation"]
            if isinstance(seg, str):
                seg = json.loads(seg)
            outer = render_instance_mask([seg[0]], H, W).astype(bool)
            if outer.sum() < min_area:
                continue
            ys, xs = np.where(outer)
            insts.append({
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "outer_polygon": [round(float(v), 2) for v in seg[0]],
                # kept so a hole model can be trained: the outer ring alone caps
                # >=0.8 at 86.7% against the true annotation, and holed regions
                # score 14.3% there
                "hole_polygons": [[round(float(v), 2) for v in hp] for hp in seg[1:]],
                "n_holes": len(seg) - 1,
                "category_name": a.get("category_name", "pattern"),
                "area_px": int(outer.sum())})
        if insts:
            out.append({"image": img, "width": W, "height": H,
                        "pool": os.path.basename(pool),
                        "mode": d.get("mode", "unknown"), "instances": insts})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=POOLS)
    ap.add_argument("--train-only-pools", nargs="*", default=[],
                    help="extra pools that go entirely into train (synthetic). "
                         "Val stays the real held-out sheets, so adding supply "
                         "never moves the measuring stick.")
    ap.add_argument("--max-per-train-only", type=int, default=0,
                    help="cap images taken from each train-only pool (0 = all)")
    ap.add_argument("--out", default="data/sam3_ft")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scenes = []
    for pool in args.pools:
        scenes.extend(load_pool(pool, args.min_area))

    # split by source image
    idx = rng.permutation(len(scenes))
    n_val = max(1, int(round(args.val_frac * len(scenes))))
    val = set(idx[:n_val].tolist())
    for i, sc in enumerate(scenes):
        sc["split"] = "val" if i in val else "train"

    # Train-only pools (synthetic) never enter val: the measuring stick stays
    # the real held-out sheets, so a gain from added supply is a real gain and
    # not an easier test set.
    for pool in args.train_only_pools:
        extra = load_pool(pool, args.min_area)
        if args.max_per_train_only and len(extra) > args.max_per_train_only:
            keep = rng.permutation(len(extra))[:args.max_per_train_only]
            extra = [extra[k] for k in sorted(keep.tolist())]
        for sc in extra:
            sc["split"] = "train"
        scenes.extend(extra)
        print(f"train-only: {len(extra):5d} images from {os.path.basename(pool)}")

    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "val"):
        rows = [s for s in scenes if s["split"] == split]
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as fh:
            for s in rows:
                fh.write(json.dumps(s) + "\n")
        n_i = sum(len(s["instances"]) for s in rows)
        holes = sum(1 for s in rows for i in s["instances"] if i["n_holes"] > 0)
        print(f"{split:5}: {len(rows):4d} images  {n_i:5d} instances  "
              f"({holes} with holes)")

    by_pool = {}
    for s in scenes:
        by_pool.setdefault(s["pool"], [0, 0])
        by_pool[s["pool"]][0] += 1
        by_pool[s["pool"]][1] += len(s["instances"])
    print("\nby pool:")
    for k, (a, b) in sorted(by_pool.items()):
        print(f"  {k:34} {a:4d} images  {b:5d} instances")
    json.dump({"pools": args.pools, "train_only_pools": args.train_only_pools,
               "max_per_train_only": args.max_per_train_only,
               "val_frac": args.val_frac, "seed": args.seed,
               "n_images": len(scenes),
               "n_instances": sum(len(s["instances"]) for s in scenes),
               "target": "outer ring, holes filled",
               "split": "by source image"},
              open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"\n-> {args.out}/{{train,val}}.jsonl + manifest.json")


if __name__ == "__main__":
    main()
