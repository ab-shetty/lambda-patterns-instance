#!/usr/bin/env python3
"""Publish the labelling-assist decoder to the HF Hub so it survives the VM.

The 2026-09-08 checkpoint was written to git-ignored `data/runs/` and existed on
exactly one machine; when that machine went away the model went with it. Nothing
in the repo pointed anywhere durable. This script is the fix: the trained decoder
goes to a Hub repo, and `scripts/sam3_region_model.py` can load it from there by
name on any box with an `HF_TOKEN`.

Only the mask decoder is published (4.2 MB). The 454M vision encoder is frozen
during training and is byte-identical to stock `facebook/sam3`, so republishing
it would be 450MB of duplication -- the loader pulls the encoder from the gated
upstream repo and layers this decoder on top.

Weights go up as **safetensors**, not the raw `.pth`: a published artifact should
not require the loader to unpickle arbitrary objects. Training metrics and the
provenance needed to rebuild travel alongside in `config.json`.

    python3 scripts/publish_sam3.py --checkpoint data/runs/sam3_ft/best.pth \
        --repo abshetty/floz-sam3-labelassist
"""
import argparse, json, os, sys, subprocess

import torch
from safetensors.torch import save_file


CARD = """---
license: apache-2.0
base_model: facebook/sam3
tags: [image-segmentation, sam3, architectural-drawings, labeling-assist]
---

# floz-sam3-labelassist

Fine-tuned **mask decoder** for SAM 3, for box-prompted region segmentation on
architectural plans: draw a rough box over a material region, get a polygon.

This is the *labelling-assist* model. It is not the Floz product model -- the
product task (one reference selection -> every matching region on the sheet)
stays with `RefUNet`. SAM 3 scores 0.308 on that task zero-shot against
RefUNet's 0.769.

## Why it exists

Roboflow's stock SAM 3 Label Assist was rejected in practice: its polygons carry
a median of **284 vertices** where a human draws **5**, which costs a labeller
more time than it saves. Douglas-Peucker takes 284 -> 7 vertices for 1.7 points
of IoU, and fine-tuning the decoder then recovers well past the starting point.

## Results

{RESULTS}

`>= 0.8` is the operationally meaningful number: roughly "accept with a nudge"
versus "redraw".

## Use

```python
from scripts.sam3_region_model import RegionModel

rm = RegionModel(checkpoint="{REPO}")          # or a local .pth, or None = zero-shot
polys = rm.polygons(image, [[x0, y0, x1, y1], ...])   # COCO-ready
```

`facebook/sam3` is gated: the account behind `HF_TOKEN` must have accepted its
terms, since the frozen encoder is pulled from there.

## Training

{PROVENANCE}

The encoder is frozen (4.2M trainable of 458.3M). Targets are the **outer ring,
holes filled** -- windows punched out of a wall are cut afterwards as separate
`remove` polygons, matching the Roboflow import pipeline.

## Known limits

- **1008x1008 is the encoder's fixed input.** A 3168px sheet is squashed before
  the decoder sees it; decoder training does not change this.
- **Holes are not predicted**, by design (see above).
- **Orientation snapping in the polygon regularizer is negative** and monotone in
  the tolerance -- rakes, gables and eaves are not on a rectilinear lattice.
- Where two regions share one material and are divided only by a corner board,
  there is no visual edge and the boundary can wander. This is the residual
  error and it is the case a generic segmenter cannot know about.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="data/runs/sam3_ft/best.pth")
    ap.add_argument("--repo", required=True, help="e.g. abshetty/floz-sam3-labelassist")
    ap.add_argument("--history", default=None, help="history.json from the run")
    ap.add_argument("--results-md", default=None, help="file holding the results table")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = os.environ.get("HF_TOKEN")
    if not tok and not args.dry_run:
        sys.exit("HF_TOKEN not set (set -a; . ~/.env; set +a)")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v.contiguous() for k, v in ck["mask_decoder"].items()}

    meta = {k: v for k, v in ck.items() if k != "mask_decoder"}
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    meta["format"] = "mask_decoder.safetensors -> Sam3TrackerModel.mask_decoder"
    meta["n_params"] = sum(v.numel() for v in sd.values())

    out = "/tmp/sam3_publish"
    os.makedirs(out, exist_ok=True)
    save_file(sd, f"{out}/mask_decoder.safetensors",
              metadata={"format": "pt", "base_model": "facebook/sam3"})
    json.dump(meta, open(f"{out}/config.json", "w"), indent=2)
    if args.history and os.path.exists(args.history):
        json.dump(json.load(open(args.history)),
                  open(f"{out}/history.json", "w"), indent=1)

    results = (open(args.results_md).read() if args.results_md
               and os.path.exists(args.results_md) else
               f"Val polygon IoU (median): **{meta.get('poly_median', float('nan')):.4f}**")
    prov = "\n".join(f"- `{k}`: {v}" for k, v in sorted(meta.items())
                     if k in {"aug", "seed", "epochs", "lr", "jitter",
                              "n_train_images", "n_train_instances",
                              "git_commit", "base_model"})
    open(f"{out}/README.md", "w").write(
        CARD.replace("{RESULTS}", results).replace("{REPO}", args.repo)
            .replace("{PROVENANCE}", prov))

    print(f"staged in {out}:")
    for f in sorted(os.listdir(out)):
        print(f"  {f}  {os.path.getsize(out+'/'+f)/1e6:.2f} MB")
    print(json.dumps(meta, indent=2, default=str))
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.create_repo(args.repo, repo_type="model", private=args.private,
                    exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=args.repo, repo_type="model",
                      commit_message=f"decoder: aug={meta.get('aug')} "
                                     f"seed={meta.get('seed')} "
                                     f"poly_median={meta.get('poly_median')}")
    print(f"\npublished -> https://huggingface.co/{args.repo} "
          f"({'private' if args.private else 'public'})")


if __name__ == "__main__":
    main()
