#!/usr/bin/env python3
"""Inference entry point for the box-prompted region model.

This is what the labelling web app calls. One object, two methods:

    from scripts.sam3_region_model import RegionModel
    rm = RegionModel()                                          # published default
    rm = RegionModel(checkpoint="abshetty/floz-sam3-labelassist")  # any Hub repo
    rm = RegionModel(checkpoint="data/runs/sam3_ft/best.pth")    # a local run
    rm = RegionModel(checkpoint=None)                           # stock, zero-shot
    polys = rm.polygons(pil_image, [[x0, y0, x1, y1], ...])     # -> list of (N,2) arrays

`polygons` returns regularized polygons in the ORIGINAL image pixel frame, ready
to write into COCO `segmentation`. `masks` returns the raw boolean masks if the
caller wants to do its own post-processing.

`checkpoint` resolves in this order: a path that exists on disk (`.pth` or
`.safetensors`), otherwise a HuggingFace Hub repo id, which is downloaded and
cached. That is what makes this model portable -- the decoder that produced the
recorded numbers lives on the Hub, not in a git-ignored `data/runs/` on one VM.
Pass `checkpoint=None` explicitly for stock zero-shot SAM 3.

`facebook/sam3` is gated, so `HF_TOKEN` must belong to an account that accepted
its terms; the frozen encoder is always pulled from there.

**`crop_zoom` is the largest accuracy lever here and needs no retraining.** The
encoder is fixed at 1008 square, so a region on a 3168px sheet reaches it
squashed; running the encoder on a window around the prompt box instead spends
that 1008 on the region. On the 29-sheet val split, fine-tuned decoder:

    crop_zoom   poly mean   >=0.8      cost
    None        0.8678      86.7%      1 forward per SHEET  (embedding cacheable)
    1.5         0.8851      88.8%      1 forward per BOX
    2           0.8819      90.8%      1 forward per BOX
    3           0.8754      91.8%      1 forward per BOX

It defaults to **off** because it forfeits the per-sheet embedding cache, which
is what makes the CPU deployment viable -- on a GPU box prefer `crop_zoom=2`.
It is strongly NEGATIVE zero-shot (>=0.8 71.4% -> 56.1%), so it is refused
unless a fine-tuned decoder is loaded, and it pays most on a decoder trained
with the crop augmentation (`scripts/sam3_augment.py`).

The model runs on CPU if no GPU is present (slower, but the app is expected to
run on a non-GPU box); pass `device` to force one.
"""
import json
import os
import numpy as np, torch
from PIL import Image

from scripts.regularize_polygon import regularize

#: Published decoder. Portable default so a fresh VM needs no local artifacts.
DEFAULT_CHECKPOINT = "abshetty/floz-sam3-labelassist"

_UNSET = object()


def _load_decoder_state(checkpoint):
    """Resolve a local file or a Hub repo id -> (state_dict, metadata)."""
    path, meta = checkpoint, {}
    if not os.path.exists(checkpoint):
        from huggingface_hub import hf_hub_download
        tok = os.environ.get("HF_TOKEN")
        path = hf_hub_download(checkpoint, "mask_decoder.safetensors", token=tok)
        try:
            meta = json.load(open(hf_hub_download(checkpoint, "config.json",
                                                  token=tok)))
        except Exception:
            pass
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path), meta
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["mask_decoder"] if "mask_decoder" in ck else ck
    return sd, {k: v for k, v in ck.items() if k != "mask_decoder"}


class RegionModel:
    def __init__(self, checkpoint=_UNSET, device=None, model_id="facebook/sam3",
                 eps_frac=0.010, crop_zoom=None):
        # _UNSET (the default) -> the published decoder. An explicit None is a
        # deliberate request for stock zero-shot SAM 3.
        if checkpoint is _UNSET:
            checkpoint = DEFAULT_CHECKPOINT
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = os.environ.get("HF_TOKEN")
        self.proc = Sam3TrackerProcessor.from_pretrained(model_id, token=tok)
        self.model = Sam3TrackerModel.from_pretrained(model_id, token=tok).to(self.device)
        self.eps_frac = eps_frac
        self.crop_zoom = crop_zoom
        self.finetuned = False
        self.info = {}
        if checkpoint:
            sd, self.info = _load_decoder_state(checkpoint)
            missing, unexpected = self.model.mask_decoder.load_state_dict(
                sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"decoder mismatch loading {checkpoint}: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected keys")
            self.model.mask_decoder.to(self.device)
            self.finetuned = True
        self.model.eval()

    @torch.no_grad()
    def _masks_cropped(self, image, boxes_xyxy, zoom):
        """One encoder forward per box, on a window `zoom`x the box's long side.

        The encoder is fixed at 1008 square, so a region on a 3168px sheet
        reaches it squashed 3x. Cropping first is the only way to spend that
        1008 on the region instead of the sheet, and it needs no retraining.
        """
        W, H = image.size
        out = []
        for b in boxes_xyxy:
            x0, y0, x1, y1 = map(float, b)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            side = max(max(x1 - x0, y1 - y0) * zoom, 64.0)
            wx0 = int(max(0, min(W - 1, cx - side / 2.0)))
            wy0 = int(max(0, min(H - 1, cy - side / 2.0)))
            wx1 = int(min(W, max(wx0 + 8, cx + side / 2.0)))
            wy1 = int(min(H, max(wy0 + 8, cy + side / 2.0)))
            sub = image.crop((wx0, wy0, wx1, wy1))
            lb = [[x0 - wx0, y0 - wy0, x1 - wx0, y1 - wy0]]
            inp = self.proc(images=sub, input_boxes=[lb],
                            return_tensors="pt").to(self.device)
            o = self.model(pixel_values=inp["pixel_values"],
                           input_boxes=inp["input_boxes"],
                           multimask_output=not self.finetuned)
            m = self.proc.post_process_masks(
                o.pred_masks.cpu(), inp["original_sizes"].cpu())[0][0].numpy()
            if m.ndim == 3:
                sc = o.iou_scores[0, 0].detach().cpu().numpy().reshape(-1)
                m = m[int(np.argmax(sc[:len(m)]))]
            full = np.zeros((H, W), bool)
            full[wy0:wy1, wx0:wx1] = m.astype(bool)
            out.append(full)
        return out

    @torch.no_grad()
    def masks(self, image, boxes_xyxy, crop_zoom=_UNSET):
        """image: PIL.Image or path. boxes: list of [x0,y0,x1,y1] in image pixels."""
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        image = image.convert("RGB")
        if not boxes_xyxy:
            return []
        zoom = self.crop_zoom if crop_zoom is _UNSET else crop_zoom
        if zoom:
            if not self.finetuned:
                # Measured: cropping a stock decoder takes >=0.8 from 71.4% to
                # 56.1%. It only pays on a decoder fine-tuned for this task.
                raise ValueError(
                    "crop_zoom needs a fine-tuned decoder; it is strongly "
                    "negative zero-shot (>=0.8 71.4% -> 56.1%)")
            return self._masks_cropped(image, boxes_xyxy, float(zoom))
        inp = self.proc(images=image,
                        input_boxes=[[list(map(float, b)) for b in boxes_xyxy]],
                        return_tensors="pt").to(self.device)
        out = self.model(pixel_values=inp["pixel_values"],
                         input_boxes=inp["input_boxes"],
                         multimask_output=not self.finetuned)
        ms = self.proc.post_process_masks(out.pred_masks.cpu(),
                                          inp["original_sizes"].cpu())[0]
        res = []
        for k in range(len(boxes_xyxy)):
            m = ms[k].numpy()
            if m.ndim == 3:      # zero-shot returns 3 candidates; take the model's pick
                sc = out.iou_scores[0, k].detach().cpu().numpy().reshape(-1)
                m = m[int(np.argmax(sc[:len(m)]))]
            res.append(m.astype(bool))
        return res

    def polygons(self, image, boxes_xyxy, crop_zoom=_UNSET):
        out = []
        for m in self.masks(image, boxes_xyxy, crop_zoom=crop_zoom):
            p = regularize(m, eps_frac=self.eps_frac)
            out.append(None if p is None or len(p) < 3 else p)
        return out
