#!/usr/bin/env python3
"""Pick each run's epoch on the validation split, then report HF14 once.

`train_refunet.py` logs `real_mIoU` on HF14 every epoch, so reading the best
epoch out of a training log selects the checkpoint using the acceptance set.
This re-scores every saved epoch on the validation complement (never trained on,
never used for acceptance), picks the epoch there, and only then reports what
that epoch scores on HF14 -- the protocol `startup.md` asks for.

    PYTHONPATH=. python3 scripts/select_epoch_on_val.py \
        --runs data/runs/ck_synthonly_*  --group-by arm
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from refmask2former import load_parquet_records
from scripts.evaluate_refunet_selection import HOLDOUT, evaluate_model, load_refunet

VALIDATION = "4,5,6,8,9,10,13,15,17,19,20,21,22,26"


def _mean_iou(model, records, indices, size, ref, thresh, device, tta=1):
    rows = evaluate_model(model, records, indices, image_max_size=size,
                          ref_size=ref, mask_thresh=thresh, device=device, tta=tta)
    return float(np.mean([r["iou"] for r in rows])), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--image-max-size", type=int, default=1280)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--out", default=None, help="write per-run JSON here")
    ap.add_argument("--tta", type=int, default=1, choices=(1, 2, 4, 8))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_parquet_records(cache_dir=args.cache_dir,
                                   config="real-world-test", split="test")
    val_idx = [int(x) for x in VALIDATION.split(",")]
    hf_idx = [int(x) for x in HOLDOUT.split(",")]

    results = []
    for run in sorted(args.runs):
        run = Path(run)
        ckpts = sorted(run.glob("epoch_*.pth"),
                       key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        if not ckpts:
            continue
        best = None
        for ck in ckpts:
            model, _ = load_refunet(str(ck), device)
            v, nv = _mean_iou(model, records, val_idx, args.image_max_size,
                              args.ref_size, args.mask_thresh, device, args.tta)
            if best is None or v > best[1]:
                best = (ck, v, model)
            else:
                del model
        ck, v, model = best
        h, nh = _mean_iou(model, records, hf_idx, args.image_max_size,
                          args.ref_size, args.mask_thresh, device, args.tta)
        del model
        torch.cuda.empty_cache()
        epoch = int(re.findall(r"\d+", ck.stem)[0])
        results.append({"run": run.name, "epoch": epoch, "val": v, "hf14": h, "tta": args.tta})
        print(f"{run.name:<34} epoch={epoch}  val={v:.4f}  HF14={h:.4f}")

    print(f"\n{'arm':<10}{'seeds':>6}{'val mean':>12}{'HF14 mean':>12}{'HF14 sd':>10}")
    groups = {}
    for r in results:
        # Everything except the trailing _s<seed>, so pool AND condition are kept
        # distinct; splitting on a single field merges e.g. disc with
        # disc_anchorzero and silently averages two different arms together.
        arm = re.sub(r"^ck_synthonly_", "", re.sub(r"_s\d+$", "", r["run"]))
        groups.setdefault(arm, []).append(r)
    for arm, rs in sorted(groups.items()):
        hf = np.array([r["hf14"] for r in rs])
        va = np.array([r["val"] for r in rs])
        print(f"{arm:<10}{len(rs):>6}{va.mean():>12.4f}{hf.mean():>12.4f}"
              f"{(hf.std(ddof=1) if len(hf) > 1 else 0):>10.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
