#!/usr/bin/env python3
"""Price CLICKS against BOXES as the labelling gesture.

A box is a drag: press, move, release, and it has to enclose the region. A click
is one event and needs no accuracy beyond landing inside. If a click gets the
same polygon, the labelling gesture gets cheaper for free -- and on a family of
15 regions that difference compounds.

The decoder in `abshetty/floz-sam3-labelassist` was fine-tuned on BOX prompts
only, so the first question is whether it transfers to point prompts at all.
SAM 3 itself accepts both (`input_points`/`input_labels`).

Clicks are sampled from the region's INTERIOR by distance transform, and from
the region with its holes removed -- a labeller selecting a wall clicks on wall,
never through a window. Extra clicks are placed in the parts the current
prediction misses, which is what a person does when correcting.

    PYTHONPATH=. python3 scripts/sam3_point_probe.py \
        --checkpoint data/runs/sam3_nodihedral_s7/best.pth --clicks 1 2 3
"""
import argparse, json, os, sys
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask
from scripts.regularize_polygon import regularize
from scripts.sam3_region_model import _load_decoder_state
from scripts.train_sam3_boxseg import jitter, interior_points


def correction_point(pred, truth, rng):
    """One click in the largest area the prediction currently gets wrong.

    Returns (point, label): label 1 to add a missed area, 0 to remove a false
    one -- the two corrections a labeller actually makes.
    """
    miss = truth & ~pred
    extra = pred & ~truth
    tgt, lab = (miss, 1) if miss.sum() >= extra.sum() else (extra, 0)
    if tgt.sum() < 50:
        return None, None
    n, l, st, _ = cv2.connectedComponentsWithStats(tgt.astype(np.uint8), 8)
    if n <= 1:
        return None, None
    big = 1 + int(np.argmax([st[i][4] for i in range(1, n)]))
    p = interior_points(l == big, 1, rng)
    return (p[0], lab) if p else (None, None)


@torch.no_grad()
def predict(model, proc, image, dev, points=None, labels=None, box=None,
            multimask=False):
    kw = {}
    if points is not None:
        kw["input_points"] = [[points]]
        kw["input_labels"] = [[labels]]
    if box is not None:
        kw["input_boxes"] = [[list(map(float, box))]]
    inp = proc(images=image, return_tensors="pt", **kw).to(dev)
    out = model(pixel_values=inp["pixel_values"],
                input_points=inp.get("input_points"),
                input_labels=inp.get("input_labels"),
                input_boxes=inp.get("input_boxes"),
                multimask_output=multimask)
    m = proc.post_process_masks(out.pred_masks.cpu(),
                                inp["original_sizes"].cpu())[0][0].numpy()
    if m.ndim == 3:
        sc = out.iou_scores[0, 0].detach().cpu().numpy().reshape(-1)
        m = m[int(np.argmax(sc[:len(m)]))]
    return m.astype(bool)


def poly_iou(mask, truth, H, W):
    p = regularize(mask)
    if p is None or len(p) < 3:
        return 0.0
    pm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(pm, [np.round(p).astype(np.int32)], 1)
    pm = pm.astype(bool)
    return float((pm & truth).sum()) / max(1, int((pm | truth).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="omit for zero-shot")
    ap.add_argument("--data", default="data/sam3_ft/val.jsonl")
    ap.add_argument("--clicks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jitter", type=float, default=0.08)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = os.environ.get("HF_TOKEN")
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3", token=tok)
    model = Sam3TrackerModel.from_pretrained("facebook/sam3", token=tok).to(dev)
    ft = bool(args.checkpoint)
    if ft:
        sd, _ = _load_decoder_state(args.checkpoint)
        model.mask_decoder.load_state_dict(sd)
        model.mask_decoder.to(dev)
    model.eval()

    sheets = [json.loads(l) for l in open(args.data)]
    print(f"decoder: {args.checkpoint or 'stock zero-shot'}   "
          f"val {len(sheets)} sheets\n")
    print(f"{'gesture':<22}{'mask mean':>10}{'poly mean':>11}{'>=0.8':>8}")
    print("-" * 51)

    # ---- box control -----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    bm, bp = [], []
    for s in sheets:
        im = Image.open(s["image"]).convert("RGB")
        W, H = im.size
        for inst in s["instances"]:
            gt = render_instance_mask([inst["outer_polygon"]], H, W).astype(bool)
            b = jitter(inst["bbox_xyxy"], W, H, args.jitter, rng)
            m = predict(model, proc, im, dev, box=b, multimask=not ft)
            bm.append(float((m & gt).sum()) / max(1, int((m | gt).sum())))
            bp.append(poly_iou(m, gt, H, W))
    bm, bp = np.array(bm), np.array(bp)
    print(f"{'box (1 drag)':<22}{bm.mean():10.4f}{bp.mean():11.4f}{(bp>=.8).mean():8.1%}")

    # ---- clicks ----------------------------------------------------------
    for nc in args.clicks:
        rng = np.random.default_rng(args.seed)
        cm, cp = [], []
        for s in sheets:
            im = Image.open(s["image"]).convert("RGB")
            W, H = im.size
            for inst in s["instances"]:
                gt = render_instance_mask([inst["outer_polygon"]], H, W).astype(bool)
                # click on material, never through a window
                solid = render_instance_mask(
                    [inst["outer_polygon"]] + inst.get("hole_polygons", []),
                    H, W).astype(bool)
                pts = interior_points(solid if solid.any() else gt, 1, rng)
                if not pts:
                    cm.append(0.0); cp.append(0.0); continue
                labs = [1]
                m = predict(model, proc, im, dev, points=pts, labels=labs,
                            multimask=not ft)
                for _ in range(nc - 1):     # corrective clicks, as a person makes
                    p, l = correction_point(m, gt, rng)
                    if p is None:
                        break
                    pts, labs = pts + [p], labs + [l]
                    m = predict(model, proc, im, dev, points=pts, labels=labs,
                                multimask=False)
                cm.append(float((m & gt).sum()) / max(1, int((m | gt).sum())))
                cp.append(poly_iou(m, gt, H, W))
        cm, cp = np.array(cm), np.array(cp)
        print(f"{f'{nc} click' + ('s' if nc > 1 else '') + ' (corrective)':<22}"
              f"{cm.mean():10.4f}{cp.mean():11.4f}{(cp>=.8).mean():8.1%}")


if __name__ == "__main__":
    main()
