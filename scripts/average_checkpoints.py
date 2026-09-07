#!/usr/bin/env python3
"""Average the weights of a run's last epochs and score the result.

Single-epoch selection on 77 validation selections is noisy: the same run can
pick its best or its worst HF14 epoch depending on which side of the noise the
val curve falls. Averaging the weights of the last few epochs of a cosine
schedule (SWA-style) smooths that out and involves no acceptance-set data. The
window is chosen on the validation split; HF14 is scored for every window so
the number is on disk, but only the val-chosen window is the reported one.

    PYTHONPATH=. python3 scripts/average_checkpoints.py --run data/runs/ck_X \
        --windows 4-8,5-8,6-8 --image-max-size 2560
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from refmask2former import load_parquet_records
from scripts.evaluate_refunet_selection import HOLDOUT, evaluate_model, load_refunet
from scripts.select_epoch_on_val import VALIDATION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--windows", default="4-8,5-8,6-8")
    ap.add_argument("--image-max-size", type=int, default=2048)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")
    records = load_parquet_records(cache_dir="./data", config="real-world-test", split="test")
    val_idx = [int(x) for x in VALIDATION.split(",")]
    hf_idx = [int(x) for x in HOLDOUT.split(",")]
    run = Path(args.run)
    results = []
    for win in args.windows.split(","):
        a, b = [int(v) for v in win.split("-")]
        sds = []
        for e in range(a, b + 1):
            ck = torch.load(run / f"epoch_{e}.pth", map_location="cpu")
            sds.append(ck["model"])
        avg = {}
        for k in sds[0]:
            t = sds[0][k]
            if t.is_floating_point():
                avg[k] = sum(sd[k].float() for sd in sds) / len(sds)
                avg[k] = avg[k].to(t.dtype)
            else:
                avg[k] = t
        out_ck = run / f"avg_e{a}-{b}.pth"
        torch.save({"model": avg, "args": ck.get("args", {}), "actual_epoch": f"avg{a}-{b}"}, out_ck)
        model, _ = load_refunet(str(out_ck), device)
        v = float(np.mean([r["iou"] for r in evaluate_model(model, records, val_idx, args.image_max_size,
                                                             224, args.mask_thresh, device)]))
        h = float(np.mean([r["iou"] for r in evaluate_model(model, records, hf_idx, args.image_max_size,
                                                             224, args.mask_thresh, device)]))
        del model
        torch.cuda.empty_cache()
        results.append({"run": run.name, "window": win, "val": v, "hf14": h, "checkpoint": str(out_ck)})
        print(f"{run.name} avg e{a}-{b}: val={v:.4f} HF14={h:.4f}", flush=True)
    best = max(results, key=lambda r: r["val"])
    print(f"val picks {best['window']}: HF14={best['hf14']:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"results": results, "val_pick": best}, indent=2) + "\n")


if __name__ == "__main__":
    main()
