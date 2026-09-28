#!/usr/bin/env python3
"""Clean (un-augmented) loss of a checkpoint on a local pool, comparable across
runs that trained with different augmentation: the logged train loss is on
augmented (e.g. --domain-random shrunk) samples, so it cannot compare a DR run
with a no-DR one. Reports the loss on the run's own held-out split (what
train_refunet.py logs as val=) and on N of its training plans, both with the
deterministic evaluation reference, one question per plan.

    PYTHONPATH=. python3 scripts/clean_loss.py --checkpoint ck.pth \\
        --pool data/synthetic/revit_train2000 --n-train 200
"""
import argparse
import sys
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from refmask2former import collate_fn, load_local_records                     # noqa: E402
from refmask2former.dataset import InstanceSegDataset, build_datasets          # noqa: E402
from scripts.evaluate_refunet_selection import load_refunet                   # noqa: E402
from scripts.train_refunet import mask_loss, union_targets                    # noqa: E402


def run(model, ds, device, roi):
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=8,
                        collate_fn=partial(collate_fn, size_divisible=32, union_only=True))
    out = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device)
            refs = batch["references"].to(device)
            valid = batch["pixel_mask"].to(device)[:, None].float()
            targets = union_targets(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images, refs, ref_box=batch["ref_boxes"].to(device) if roi else None)
            loss, bce, dice = mask_loss(logits.float(), targets, valid, 1.0, 2.0)
            out.append((float(loss), float(bce), float(dice)))
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n-train", type=int, default=200)
    a = ap.parse_args()
    device = torch.device("cuda")
    model, ck = load_refunet(a.checkpoint, device)
    args = ck.get("args", {})
    recs = load_local_records(a.pool)
    train_ds, val_ds = build_datasets(recs, image_max_size=int(args.get("image_max_size", 2048)),
                                      ref_size=224, train_split=float(args.get("train_split", 0.99)),
                                      seed=int(args.get("seed", 31)))
    clean_train = InstanceSegDataset(recs, train_ds.indices[:a.n_train],
                                     int(args.get("image_max_size", 2048)), 224, augment=False)
    roi = bool(getattr(model, "roi_ref", False))
    v = run(model, val_ds, device, roi)
    t = run(model, clean_train, device, roi)
    print(f"held-out loss {v[0]:.4f} (bce {v[1]:.4f} dice {v[2]:.4f}) | "
          f"clean train loss {t[0]:.4f} (bce {t[1]:.4f} dice {t[2]:.4f})")


if __name__ == "__main__":
    main()
