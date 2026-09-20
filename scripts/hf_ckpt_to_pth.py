#!/usr/bin/env python3
"""Rebuild a loadable `.pth` from a published HF checkpoint.

`publish_refunet.py` deliberately publishes weights as safetensors plus a
`config.json`, dropping the pickled `.pth`. Every evaluation script in this repo
loads `{"model": state_dict, "args": {...}}`, so reconstitute that shape from
the two published files.

    python3 scripts/hf_ckpt_to_pth.py --repo abshetty/floz-refunet-res2560-e4 \
        --out data/hf_ckpt/res2560-e4.pth
"""
import argparse
import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--local-dir", default="", help="already-downloaded snapshot")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if a.local_dir:
    weights = os.path.join(a.local_dir, "model.safetensors")
    config = os.path.join(a.local_dir, "config.json")
else:
    weights = hf_hub_download(a.repo, "model.safetensors")
    config = hf_hub_download(a.repo, "config.json")

cfg = json.load(open(config))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
torch.save({"model": load_file(weights), "args": cfg["args"],
            "epoch": cfg.get("epoch"), "provenance": {k: v for k, v in cfg.items()
                                                      if k != "args"}}, a.out)
print(f"{a.repo} -> {a.out}")
print(f"  model={cfg['args'].get('model','unet')} width={cfg['args'].get('width')} "
      f"res={cfg['args'].get('image_max_size')} src={cfg.get('source_checkpoint')}")
