"""Wall-clock per query for the product variants (2026-11-11). One user query = image features + decode at each scale (the
features are not shared between queries here: worst case, a fresh sheet per query) + the CPU post-processing.
  GPU: model.features + forward_cached at each size, bf16 autocast as the evaluator, batch 1, median of 10 after 3 warm-ups,
       on HF14 sheets of three sizes (small / median / large).
  CPU: the post-processing of each product variant on saved maps (one question per sheet, all HF14 sheets).
Usage: PYTHONPATH=. python3 scripts/bench_latency.py CKPT MAPS_DIR"""
import sys, time, io, json, statistics
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, ".")
import numpy as np, cv2, torch
from PIL import Image
from evaluate_refunet_selection import load_refunet, _normalize_on
from refmask2former import load_parquet_records

ck, maps = sys.argv[1], Path(sys.argv[2])
dev = torch.device("cuda")
model = load_refunet(ck, dev)
model = model[0] if isinstance(model, tuple) else model
model.eval()
recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
sizes = {}
for i in (12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24):
    im = np.asarray(Image.open(io.BytesIO(recs[i]["image"]["bytes"] if isinstance(recs[i]["image"], dict) else recs[i]["image"])).convert("RGB"))
    sizes[i] = im
order = sorted(sizes, key=lambda i: max(sizes[i].shape[:2]))
pick = [order[0], order[len(order) // 2], order[-1]]


def gpu_time(img0, S):
    h0, w0 = img0.shape[:2]; sc = S / max(h0, w0)
    img = cv2.resize(img0, (round(w0 * sc), round(h0 * sc)), interpolation=cv2.INTER_LINEAR)
    t = _normalize_on(img, dev).unsqueeze(0)
    ref = torch.randn(1, 3, 224, 224, device=dev)
    ts = []
    for k in range(13):
        torch.cuda.synchronize(); a = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.features(t)
            model.forward_cached(f, t.shape[-2:], ref, None, image=t).sigmoid().float().cpu()
        torch.cuda.synchronize()
        if k >= 3:
            ts.append(time.perf_counter() - a)
    return statistics.median(ts)


print("GPU, seconds per query (features + decode), sheets", [f"{i}:{max(sizes[i].shape[:2])}px" for i in pick])
G = {}
for S in (2048, 3072, 4096, 5120):
    G[S] = [gpu_time(sizes[i], S) for i in pick]
    print(f"  {S}: " + "  ".join(f"{x:.3f}" for x in G[S]))
for name, ss in (("v1 (4096+2048)", (4096, 2048)), ("v2/v3 (5120+4096+3072)", (5120, 4096, 3072))):
    print(f"  {name}: " + "  ".join(f"{sum(G[S][j] for S in ss):.3f}" for j in range(3)))

# CPU post-processing per variant
import common
from stack import colour_cut, CUR
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, offset_polys, snap_to_cells, probs_to_polygons, LOW_VERTEX
common.WITH_BOX[0] = True; common.MAPS = maps
T = {"v1 ms+cc_lv": [], "v2": [], "v3": []}
for job in common.jobs("hf14fix4"):
    idx, img, anyann, qs = common.load(job); h, w = img.shape[:2]
    ref, cat, tgt, p4, box = qs[0]
    s = p4.shape[1] / w
    P2 = np.load(maps / "hf14_2048" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
    a = time.perf_counter()
    P = 0.7 * cv2.resize(p4, (w, h)).astype(np.float32) / 255 + 0.3 * cv2.resize(P2, (w, h)).astype(np.float32) / 255
    m = colour_cut(img, fill_gaps(_fill_holes(P > 0.35, 4096 / s / s), P), box)
    probs_to_polygons(np.where(m, 255, 0).astype(np.uint8), m.shape, **LOW_VERTEX)
    T["v1 ms+cc_lv"].append(time.perf_counter() - a)
    for v in ("v2", "v3"):
        a = time.perf_counter()
        P = cv2.resize(p4, (w, h)).astype(np.float32) / 255 / 3
        for z in (5120, 3072):
            P += cv2.resize(np.load(maps / f"hf14_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"], (w, h)).astype(np.float32) / 255 / 3
        m = colour_cut(img, fill_gaps(_fill_holes(P > 0.35, 4096 / s / s), P), box)
        if v == "v3":
            m = snap_to_cells(m, img, s)
        offset_polys(mask_to_polygons(m, 0.005, 6.0), 1.0)
        T[v].append(time.perf_counter() - a)
print("CPU post-processing, seconds per query: median (max) over the HF14 sheets")
for k, v in T.items():
    print(f"  {k}: {statistics.median(v):.3f} ({max(v):.3f})")
