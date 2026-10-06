#!/usr/bin/env python3
"""Do backbone features flag unrealistic reference boxes? (2026-10-06, prototype)

Hand labels give no reference region: the sampler takes any box inside a labelled piece, so boxes land on
room tags, fixtures, windows, railings. Idea: a box a user would draw shows the piece's own material, so
its features sit near the piece's typical feature. Per question: the box's mean feature (one backbone
level, cells overlapping the box) vs the median feature of the piece's interior cells; distance =
1 - cosine. Scored against a hand list of unrealistic boxes (AUC, and the share flagged at a few
thresholds), on a local pool's evaluator output.

    PYTHONPATH=. python3 scripts/ref_region_probe.py --checkpoint CK --pool POOL --metrics M.json \
        --bad data/evaluations/gemini30/unrealistic_boxes_v1.json [--level 1] [--size 2048]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from refmask2former.dataset import render_instance_mask
from scripts.evaluate_refunet_selection import _normalize_chw, load_refunet


def piece_stats(feat, mask, box, scale):
    """feat: C x h x w (L2-normalised per cell); mask: native bool; box native xywh."""
    C, h, w = feat.shape
    H0, W0 = mask.shape
    m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.999
    if m.sum() < 1:
        m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5
    if m.sum() < 1:
        return None
    cells = feat[:, m]                                  # C x n
    proto = np.median(cells, axis=1)
    proto /= np.linalg.norm(proto) + 1e-8
    x, y, bw, bh = box
    sx, sy = w / W0, h / H0
    x0, x1 = int(np.floor(x * sx)), max(int(np.floor(x * sx)) + 1, int(np.ceil((x + bw) * sx)))
    y0, y1 = int(np.floor(y * sy)), max(int(np.floor(y * sy)) + 1, int(np.ceil((y + bh) * sy)))
    b = feat[:, y0:y1, x0:x1].reshape(C, -1).mean(1)
    b /= np.linalg.norm(b) + 1e-8
    cell_d = 1 - proto @ cells                          # spread of the piece's own cells
    return float(1 - b @ proto), float(np.percentile(cell_d, 50)), float(np.percentile(cell_d, 90))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--bad", required=True)
    ap.add_argument("--level", type=int, default=1, help="feature level: 0..3 = stride 4/8/16/32")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(32)
    model, _ = load_refunet(a.checkpoint, torch.device("cpu"))
    model.eval()
    pool = Path(a.pool)
    files = sorted((pool / "annotations").glob("*.json"))
    rows = json.load(open(a.metrics))["selections"]
    bad = {tuple(k) for k in json.load(open(a.bad))["dropped"]}
    by = {}
    for r in rows:
        by.setdefault(r["image_index"], []).append(r)
    out = []
    for i, f in enumerate(files):
        if i not in by:
            continue
        ann = json.loads(f.read_text())
        img = cv2.cvtColor(cv2.imread(str(pool / "images" / ann["image"]["file_name"])), cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]
        s = a.size / max(H0, W0)
        im = cv2.resize(img, (round(W0 * s), round(H0 * s)), interpolation=cv2.INTER_LINEAR)
        with torch.inference_mode():
            feat = model.features(_normalize_chw(im).unsqueeze(0))[a.level][0].float().numpy()
        feat /= np.linalg.norm(feat, axis=0, keepdims=True) + 1e-8
        anns = ann["annotations"]
        for r in by[i]:
            q = r["reference_instance"]
            mask = render_instance_mask(anns[q]["segmentation"], H0, W0).astype(bool)
            st = piece_stats(feat, mask, r["reference_box_native"], s)
            if st is None:
                continue
            out.append({"image_index": i, "reference_instance": q, "d_box": st[0], "d_p50": st[1],
                        "d_p90": st[2], "bad": (i, q) in bad, "box": r["reference_box_native"]})
        print(f"sheet {i} done", flush=True)
    y = np.array([o["bad"] for o in out])
    for key, score in (("d_box", np.array([o["d_box"] for o in out])),
                       ("d_box / d_p90", np.array([o["d_box"] / max(o["d_p90"], 1e-6) for o in out]))):
        pos, neg = score[y], score[~y]
        auc = (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
        print(f"{key}: AUC {auc:.3f}  (bad {len(pos)}, ok {len(neg)})")
        for t in np.percentile(score, [70, 80, 85, 90]):
            fl = score >= t
            print(f"   flag >= {t:.3f}: flagged {fl.sum():3d}  recall {fl[y].mean():.2f}  precision {y[fl].mean():.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
