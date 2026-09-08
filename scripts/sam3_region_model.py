#!/usr/bin/env python3
"""Inference entry point for the box-prompted region model.

This is what the labelling web app calls. One object, two methods:

    from scripts.sam3_region_model import RegionModel
    rm = RegionModel(checkpoint="data/runs/sam3_ft/best.pth")   # None = zero-shot
    polys = rm.polygons(pil_image, [[x0, y0, x1, y1], ...])     # -> list of (N,2) arrays

`polygons` returns regularized polygons in the ORIGINAL image pixel frame, ready
to write into COCO `segmentation`. `masks` returns the raw boolean masks if the
caller wants to do its own post-processing.

The model runs on CPU if no GPU is present (slower, but the app is expected to
run on a non-GPU box); pass `device` to force one.
"""
import os
import numpy as np, torch
from PIL import Image

from scripts.regularize_polygon import regularize


class RegionModel:
    def __init__(self, checkpoint=None, device=None, model_id="facebook/sam3",
                 eps_frac=0.010):
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = os.environ.get("HF_TOKEN")
        self.proc = Sam3TrackerProcessor.from_pretrained(model_id, token=tok)
        self.model = Sam3TrackerModel.from_pretrained(model_id, token=tok).to(self.device)
        self.eps_frac = eps_frac
        self.finetuned = False
        if checkpoint:
            sd = torch.load(checkpoint, map_location="cpu")
            self.model.mask_decoder.load_state_dict(sd["mask_decoder"])
            self.finetuned = True
            self.info = {k: v for k, v in sd.items() if k != "mask_decoder"}
        self.model.eval()

    @torch.no_grad()
    def masks(self, image, boxes_xyxy):
        """image: PIL.Image or path. boxes: list of [x0,y0,x1,y1] in image pixels."""
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        image = image.convert("RGB")
        if not boxes_xyxy:
            return []
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

    def polygons(self, image, boxes_xyxy):
        out = []
        for m in self.masks(image, boxes_xyxy):
            p = regularize(m, eps_frac=self.eps_frac)
            out.append(None if p is None or len(p) < 3 else p)
        return out
