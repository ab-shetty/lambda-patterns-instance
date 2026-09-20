#!/usr/bin/env python3
"""Publish a RefUNet / RefCrossAttnUNet checkpoint to the HF Hub so it survives
the VM. `data/runs/` is git-ignored and this machine is ephemeral -- see
publish_sam3.py for the identical rationale on the labelling-assist track.

Only the model weights are published (safetensors, not the raw .pth: a
published artifact should not require unpickling arbitrary objects). Optimizer
and scheduler state are dropped -- not needed for inference and the bulk of the
checkpoint's size.

    python3 scripts/publish_refunet.py \
        --checkpoint data/runs/ck_mixr4_res2048_seed31_cont2/epoch_20.pth \
        --repo abshetty/floz-refunet-warmrestart-e20
"""
import argparse, json, os, sys, subprocess

import torch
from safetensors.torch import save_file

CARD = """---
license: apache-2.0
tags: [image-segmentation, reference-conditioned, architectural-drawings]
---

# {REPO_NAME}

`{MODEL_CLASS}` checkpoint for the Floz reference-conditioned region-selection
product model: a user draws a rectangle over a pattern/material region in an
architectural plan, the model returns every region on the sheet sharing that
pattern.

Not the labelling-assist track (`floz-sam3-labelassist*`) -- this is the
product model itself.

## Provenance

{PROVENANCE}

## Load

```python
import torch
from safetensors.torch import load_file
from refmask2former.ref_unet import RefUNet
from refmask2former.ref_attn_unet import RefCrossAttnUNet

sd = load_file("model.safetensors")  # download from this repo first
model = {MODEL_CLASS}(width={WIDTH}{EXTRA_ARGS})
model.load_state_dict(sd)
model.eval()
```

See `codex_doc.md` / `startup.md` in the source repo for the full recipe,
evaluation protocol, and whether this specific checkpoint's result has been
replicated at more than one seed before trusting it as a result rather than a
data point.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True, help="e.g. abshetty/floz-refunet-warmrestart-e20")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = os.environ.get("HF_TOKEN")
    if not tok and not args.dry_run:
        sys.exit("HF_TOKEN not set (set -a; . ~/.env; set +a)")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v.contiguous() for k, v in ck["model"].items()}
    train_args = ck.get("args", {})
    # The class must follow the checkpoint's own --model. Getting this wrong is
    # how floz-refunet-synth100k-e1 ended up carrying another run's weights
    # under the wrong name; a card that misnames the architecture is the same
    # failure one step earlier.
    _m = str(train_args.get("model", "unet"))
    model_class = ("RefCrossAttnUNet" if _m == "crossattn"
                   else "RefSwinUNet" if _m.startswith("swin") else "RefUNet")

    meta = {k: v for k, v in ck.items() if k not in ("model", "optimizer", "scheduler")}
    meta["args"] = train_args
    meta["model_class"] = model_class
    meta["source_checkpoint"] = args.checkpoint
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    meta["n_params"] = sum(v.numel() for v in sd.values())

    out = "/tmp/refunet_publish"
    os.makedirs(out, exist_ok=True)
    save_file(sd, f"{out}/model.safetensors", metadata={"format": "pt"})
    json.dump(meta, open(f"{out}/config.json", "w"), indent=2, default=str)

    prov = "\n".join(f"- `{k}`: {v}" for k, v in sorted(meta.items())
                     if k in {"actual_epoch", "real_mean_iou", "git_commit",
                              "source_checkpoint"})
    prov += "\n" + "\n".join(f"- `{k}`: {v}" for k, v in sorted(train_args.items())
                             if k in {"local_data", "seed", "image_max_size",
                                      "width", "lr", "attn_heads"})
    extra_args = (", num_heads=4" if model_class == "RefCrossAttnUNet"
                  else f", pretrained=False, backbone=\"{_m}\""
                  if model_class == "RefSwinUNet" else ", pretrained=False")
    repo_name = args.repo.split("/")[-1]
    open(f"{out}/README.md", "w").write(
        CARD.replace("{REPO_NAME}", repo_name).replace("{MODEL_CLASS}", model_class)
            .replace("{PROVENANCE}", prov).replace("{WIDTH}", str(train_args.get("width", 128)))
            .replace("{EXTRA_ARGS}", extra_args))

    print(f"staged in {out}:")
    for f in sorted(os.listdir(out)):
        print(f"  {f}  {os.path.getsize(out + '/' + f) / 1e6:.2f} MB")
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}, indent=2, default=str))
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=args.repo, repo_type="model",
                      commit_message=f"{model_class} epoch={meta.get('actual_epoch')} "
                                     f"real_mean_iou={meta.get('real_mean_iou')}")
    print(f"\npublished -> https://huggingface.co/{args.repo} "
          f"({'private' if args.private else 'public'})")


if __name__ == "__main__":
    main()
