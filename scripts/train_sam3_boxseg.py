#!/usr/bin/env python3
"""Fine-tune SAM 3's mask decoder for box-prompted region segmentation on plans.

Trains the labelling-assist model: a hand-drawn box over a material region ->
the mask of that region. The 454M vision encoder is FROZEN and only the 4.2M
mask decoder is trained, which is the right scale for 767 training instances and
lets the encoder run under no_grad.

Target is the OUTER ring (holes filled); windows are cut afterwards as separate
`remove` polygons, matching the Roboflow pipeline.

Boxes are jittered every epoch because the deployed prompt is hand-drawn, not a
tight ground-truth box.

    PYTHONPATH=. python3 scripts/train_sam3_boxseg.py --epochs 12 --out data/runs/sam3_ft
"""
import argparse, json, os, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask
from scripts.regularize_polygon import regularize

MASK_RES = 288


def load_split(path):
    return [json.loads(l) for l in open(path)]


def jitter(box, W, H, frac, rng):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    j = [x0 + rng.uniform(-frac, frac) * bw, y0 + rng.uniform(-frac, frac) * bh,
         x1 + rng.uniform(-frac, frac) * bw, y1 + rng.uniform(-frac, frac) * bh]
    return [max(0.0, j[0]), max(0.0, j[1]), min(W - 1.0, j[2]), min(H - 1.0, j[3])]


def targets_for(scene, res=MASK_RES):
    H, W = scene["height"], scene["width"]
    out = []
    for inst in scene["instances"]:
        m = render_instance_mask([inst["outer_polygon"]], H, W).astype(np.uint8)
        out.append(cv2.resize(m, (res, res), interpolation=cv2.INTER_AREA))
    return np.stack(out).astype(np.float32)


def dice_bce(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum((-2, -1)) + eps
    den = p.sum((-2, -1)) + target.sum((-2, -1)) + eps
    return bce + (1.0 - num / den).mean()


@torch.no_grad()
def evaluate(model, proc, scenes, dev, jitter_frac, seed=7, multimask=False):
    rng = np.random.default_rng(seed)
    mask_ious, poly_ious, verts = [], [], []
    for s in scenes:
        im = Image.open(s["image"]).convert("RGB")
        W, H = im.size
        boxes = [jitter(i["bbox_xyxy"], W, H, jitter_frac, rng) for i in s["instances"]]
        inp = proc(images=im, input_boxes=[boxes], return_tensors="pt").to(dev)
        out = model(pixel_values=inp["pixel_values"], input_boxes=inp["input_boxes"],
                    multimask_output=multimask)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
        for k, inst in enumerate(s["instances"]):
            gt = render_instance_mask([inst["outer_polygon"]], H, W).astype(bool)
            mk = masks[k].numpy()
            mk = mk if mk.ndim == 2 else mk[int(np.argmax(
                out.iou_scores[0, k].detach().cpu().numpy().reshape(-1)))]
            mk = mk.astype(bool)
            mask_ious.append(float((mk & gt).sum()) / max(1, int((mk | gt).sum())))
            p = regularize(mk)
            if p is None or len(p) < 3:
                poly_ious.append(0.0); continue
            pm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(pm, [np.round(p).astype(np.int32)], 1)
            pm = pm.astype(bool)
            poly_ious.append(float((pm & gt).sum()) / max(1, int((pm | gt).sum())))
            verts.append(len(p))
    return (np.array(mask_ious), np.array(poly_ious), np.array(verts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sam3_ft")
    ap.add_argument("--out", default="data/runs/sam3_ft")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--jitter", type=float, default=0.08)
    ap.add_argument("--max-boxes", type=int, default=12, help="cap boxes per forward (memory)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dev = "cuda"
    tok = os.environ.get("HF_TOKEN")
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3", token=tok)
    model = Sam3TrackerModel.from_pretrained("facebook/sam3", token=tok).to(dev)

    for p in model.vision_encoder.parameters():
        p.requires_grad_(False)
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in train_params)
    print(f"trainable: {n_tr/1e6:.2f}M of {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    train = load_split(f"{args.data}/train.jsonl")
    val = load_split(f"{args.data}/val.jsonl")
    print(f"train {len(train)} images / {sum(len(s['instances']) for s in train)} instances | "
          f"val {len(val)} images / {sum(len(s['instances']) for s in val)} instances")

    model.eval()
    mi, pi, vv = evaluate(model, proc, val, dev, args.jitter, seed=args.seed)
    print(f"\nepoch  -1 (zero-shot): mask {mi.mean():.4f}/{np.median(mi):.4f}  "
          f"poly {pi.mean():.4f}/{np.median(pi):.4f}  >=0.8 {(pi>=.8).mean():.1%}  verts {np.median(vv):.0f}")
    best = float(np.median(pi)); os.makedirs(args.out, exist_ok=True)
    hist = [{"epoch": -1, "poly_mean": float(pi.mean()), "poly_median": float(np.median(pi)),
             "mask_mean": float(mi.mean()), "ge80": float((pi >= .8).mean())}]

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    for ep in range(args.epochs):
        model.train(); model.vision_encoder.eval()
        order = rng.permutation(len(train)); tot, nb = 0.0, 0
        t0 = time.time()
        for si in order:
            s = train[si]
            im = Image.open(s["image"]).convert("RGB")
            W, H = im.size
            idx = np.arange(len(s["instances"]))
            if len(idx) > args.max_boxes:
                idx = rng.choice(idx, args.max_boxes, replace=False)
            boxes = [jitter(s["instances"][k]["bbox_xyxy"], W, H, args.jitter, rng) for k in idx]
            tgt = torch.from_numpy(targets_for(
                {"height": H, "width": W,
                 "instances": [s["instances"][k] for k in idx]})).to(dev)
            inp = proc(images=im, input_boxes=[boxes], return_tensors="pt").to(dev)
            with torch.no_grad():
                emb = model.get_image_embeddings(inp["pixel_values"])
            out = model(image_embeddings=[e.detach() for e in emb],
                        input_boxes=inp["input_boxes"], multimask_output=False)
            logits = out.pred_masks[0, :, 0]                     # [N, 288, 288]
            loss = dice_bce(logits.float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        sched.step()
        model.eval()
        mi, pi, vv = evaluate(model, proc, val, dev, args.jitter, seed=args.seed)
        print(f"epoch {ep:3d}: loss {tot/max(nb,1):.4f}  mask {mi.mean():.4f}/{np.median(mi):.4f}  "
              f"poly {pi.mean():.4f}/{np.median(pi):.4f}  >=0.8 {(pi>=.8).mean():.1%}  "
              f"verts {np.median(vv):.0f}  {time.time()-t0:.0f}s", flush=True)
        hist.append({"epoch": ep, "loss": tot/max(nb,1), "poly_mean": float(pi.mean()),
                     "poly_median": float(np.median(pi)), "mask_mean": float(mi.mean()),
                     "ge80": float((pi >= .8).mean())})
        if np.median(pi) > best:
            best = float(np.median(pi))
            torch.save({"mask_decoder": model.mask_decoder.state_dict(),
                        "epoch": ep, "poly_median": best}, f"{args.out}/best.pth")
            print(f"   -> saved best (poly median {best:.4f})")
        json.dump(hist, open(f"{args.out}/history.json", "w"), indent=1)
    print(f"\nbest val polygon IoU (median): {best:.4f}")


if __name__ == "__main__":
    main()
