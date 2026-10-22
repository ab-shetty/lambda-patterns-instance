#!/usr/bin/env python3
"""Float weight average of checkpoints (2026-10-22, close-out): python3 scripts/blend_ckpts.py OUT.pth A.pth:0.5 B.pth:0.3 C.pth:0.2
Weights are renormalised to sum to 1; non-float tensors come from the first checkpoint (as run_single_pass.sh's blend)."""
import sys
import torch

out, parts = sys.argv[1], [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) for p in sys.argv[2:]]
tot = sum(w for _, w in parts)
cks = [(torch.load(p, map_location="cpu"), w / tot) for p, w in parts]
first = cks[0][0]
avg = {k: (sum(c["model"][k].float() * w for c, w in cks)).to(first["model"][k].dtype)
       if first["model"][k].is_floating_point() else first["model"][k] for k in first["model"]}
torch.save({**first, "model": avg, "epoch": None, "actual_epoch": None}, out)
print(out, [(p, round(w / tot, 3)) for p, w in parts])
