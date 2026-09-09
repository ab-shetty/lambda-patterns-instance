#!/usr/bin/env python3
"""Price inference-time cropping against the 1008x1008 encoder ceiling.

`labeling_assist.md` records the ceiling as fixed: "SAM 3's vision encoder is
fixed at 1008 square, so a 3168px sheet is squashed and fine boundaries are
limited by it. Decoder training does not change this." True -- but that is an
argument about *training*. Nothing forces us to hand the encoder the whole sheet
at inference. Cropping a window around the prompt box puts the region under the
box at a far higher effective resolution, and costs no retraining at all.

The trade is throughput: the full-sheet path embeds a sheet once and every
subsequent box on it is nearly free, which is what suits labelling many regions
per sheet. Cropping means one encoder forward per box. So this only earns its
place if the accuracy gain is real.

    PYTHONPATH=. python3 scripts/sam3_crop_probe.py \
        --checkpoint data/runs/sam3_nodihedral_s7/best.pth --zooms 0 1.5 2 3
    #   zoom 0 = the current full-sheet path (the control)
"""
import argparse, json, os, sys, time
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask
from scripts.regularize_polygon import regularize
from scripts.sam3_region_model import _load_decoder_state


def jitter(box, W, H, frac, rng):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    j = [x0 + rng.uniform(-frac, frac) * bw, y0 + rng.uniform(-frac, frac) * bh,
         x1 + rng.uniform(-frac, frac) * bw, y1 + rng.uniform(-frac, frac) * bh]
    return [max(0.0, j[0]), max(0.0, j[1]), min(W - 1.0, j[2]), min(H - 1.0, j[3])]


def window_for(box, W, H, zoom):
    """Crop window around `box`, `zoom`x its longest side, clipped to the sheet."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * zoom
    side = max(side, 64.0)
    wx0 = int(max(0, min(W - 1, cx - side / 2.0)))
    wy0 = int(max(0, min(H - 1, cy - side / 2.0)))
    wx1 = int(min(W, max(wx0 + 8, cx + side / 2.0)))
    wy1 = int(min(H, max(wy0 + 8, cy + side / 2.0)))
    return wx0, wy0, wx1, wy1


@torch.no_grad()
def run(model, proc, image, boxes, dev, zoom):
    """zoom<=0 -> one forward for the whole sheet; else one forward per box."""
    W, H = image.size
    if zoom <= 0:
        inp = proc(images=image, input_boxes=[[list(map(float, b)) for b in boxes]],
                   return_tensors="pt").to(dev)
        out = model(pixel_values=inp["pixel_values"], input_boxes=inp["input_boxes"],
                    multimask_output=False)
        ms = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
        return [ms[k].numpy().astype(bool) if ms[k].numpy().ndim == 2
                else ms[k].numpy()[0].astype(bool) for k in range(len(boxes))]
    res = []
    for b in boxes:
        wx0, wy0, wx1, wy1 = window_for(b, W, H, zoom)
        sub = image.crop((wx0, wy0, wx1, wy1))
        lb = [[float(b[0] - wx0), float(b[1] - wy0),
               float(b[2] - wx0), float(b[3] - wy0)]]
        inp = proc(images=sub, input_boxes=[lb], return_tensors="pt").to(dev)
        out = model(pixel_values=inp["pixel_values"], input_boxes=inp["input_boxes"],
                    multimask_output=False)
        m = proc.post_process_masks(out.pred_masks.cpu(),
                                    inp["original_sizes"].cpu())[0][0].numpy()
        if m.ndim == 3:
            m = m[0]
        full = np.zeros((H, W), bool)
        full[wy0:wy1, wx0:wx1] = m.astype(bool)
        res.append(full)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="local .pth/.safetensors, Hub repo id, or omit for zero-shot")
    ap.add_argument("--data", default="data/sam3_ft")
    ap.add_argument("--zooms", type=float, nargs="+", default=[0, 1.5, 2.0, 3.0])
    ap.add_argument("--jitter", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = os.environ.get("HF_TOKEN")
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3", token=tok)
    model = Sam3TrackerModel.from_pretrained("facebook/sam3", token=tok).to(dev)
    if args.checkpoint:
        sd, _ = _load_decoder_state(args.checkpoint)
        model.mask_decoder.load_state_dict(sd)
        model.mask_decoder.to(dev)
    model.eval()

    val = [json.loads(l) for l in open(f"{args.data}/val.jsonl")]
    print(f"checkpoint: {args.checkpoint or 'zero-shot'}   "
          f"val {len(val)} sheets / {sum(len(s['instances']) for s in val)} instances\n")
    print(f"{'zoom':>6} {'mask mean/med':>16} {'poly mean/med':>16} {'>=0.8':>7} "
          f"{'big sheets':>11} {'s/inst':>7}")
    print("-" * 72)

    for z in args.zooms:
        rng = np.random.default_rng(args.seed)
        mi, pi, big = [], [], []
        t0 = time.time(); n = 0
        for s in val:
            im = Image.open(s["image"]).convert("RGB")
            W, H = im.size
            boxes = [jitter(i["bbox_xyxy"], W, H, args.jitter, rng) for i in s["instances"]]
            masks = run(model, proc, im, boxes, dev, z)
            n += len(boxes)
            for k, inst in enumerate(s["instances"]):
                gt = render_instance_mask([inst["outer_polygon"]], H, W).astype(bool)
                m = masks[k]
                mi.append(float((m & gt).sum()) / max(1, int((m | gt).sum())))
                p = regularize(m)
                if p is None or len(p) < 3:
                    pi.append(0.0)
                else:
                    pm = np.zeros((H, W), np.uint8)
                    cv2.fillPoly(pm, [np.round(p).astype(np.int32)], 1)
                    pm = pm.astype(bool)
                    pi.append(float((pm & gt).sum()) / max(1, int((pm | gt).sum())))
                # the sheets the ceiling argument is actually about
                if max(W, H) >= 1500:
                    big.append(pi[-1])
        mi, pi = np.array(mi), np.array(pi)
        dt = (time.time() - t0) / max(1, n)
        lab = "full" if z <= 0 else f"{z:g}x"
        print(f"{lab:>6} {mi.mean():7.4f}/{np.median(mi):.4f} "
              f"{pi.mean():7.4f}/{np.median(pi):.4f} {(pi>=.8).mean():6.1%} "
              f"{np.mean(big) if big else float('nan'):11.4f} {dt:7.3f}")
    print(f"\n'big sheets' = mean polygon IoU on instances from sheets >=1500px "
          f"(n={len(big)}), where the 1008 squash bites hardest.")


if __name__ == "__main__":
    main()
