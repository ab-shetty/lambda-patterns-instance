"""Line signature (2026-10-22): is the reference's LINE pattern present -- direction, spacing, density -- wherever the lines
fall? Per 32 px window (stride 8, @2048 working scale) of the stroke map (ink v2): FFT magnitude binned into 8 directions x 6
spacing bands (DC dropped, log, L2-normalised) + stroke density. Similarity to the reference = cosine of the spectra (the
reference descriptor = mean over windows inside its box), times a density term min(d, d_ref) / max(d, d_ref).
Descriptive here: distribution of the similarity over the mask's correct vs wrong pixels, per split."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from common import ink_map_v2
from methods import fill_holes
common.WITH_BOX[0] = True
T = 0.35
WIN, STR = 32, 8
yy, xx = np.mgrid[-WIN // 2:WIN // 2, -WIN // 2:WIN // 2]
RAD = np.hypot(yy, xx); ANG = (np.degrees(np.arctan2(yy, xx)) % 180)
RB = np.digitize(RAD, [1.5, 3, 5, 8, 11, 16]) - 1        # 6 spacing bands (cycles per window), DC (r < 1.5) = -1
AB = np.minimum((ANG / 22.5).astype(int), 7)
BIN = np.where((RB >= 0) & (RB < 6), RB * 8 + AB, -1)


def descriptors(stroke):
    H, W = stroke.shape
    ys = np.arange(0, H - WIN + 1, STR); xs = np.arange(0, W - WIN + 1, STR)
    win = np.lib.stride_tricks.sliding_window_view(stroke, (WIN, WIN))[::STR, ::STR]
    gh, gw = win.shape[:2]
    f = np.abs(np.fft.fftshift(np.fft.fft2(win.reshape(-1, WIN, WIN) - win.reshape(-1, WIN, WIN).mean((1, 2), keepdims=True)), axes=(1, 2)))
    f = f.reshape(len(f), -1); b = BIN.ravel(); ok = b >= 0
    spec = np.zeros((len(f), 48), np.float32)
    np.add.at(spec.T, b[ok], f[:, ok].T)
    spec = np.log1p(spec); spec /= np.linalg.norm(spec, axis=1, keepdims=True) + 1e-6
    dens = win.reshape(len(win.reshape(-1, WIN, WIN)), -1).mean(1)
    return spec.reshape(gh, gw, 48), dens.reshape(gh, gw), ys, xs


def sim_map(stroke, box):
    spec, dens, ys, xs = descriptors(stroke)
    x, y, bw, bh = box
    gy = (np.arange(len(ys)) * STR + WIN // 2); gx = (np.arange(len(xs)) * STR + WIN // 2)
    iy = (gy >= y) & (gy < y + bh); ix = (gx >= x) & (gx < x + bw)
    if not iy.any() or not ix.any():
        return None, None
    rs = spec[np.ix_(iy, ix)].reshape(-1, 48).mean(0); rs /= np.linalg.norm(rs) + 1e-6
    rd = dens[np.ix_(iy, ix)].mean()
    s = (spec @ rs) * (np.minimum(dens, rd) / np.maximum(np.maximum(dens, rd), 1e-6))
    H, W = stroke.shape
    full = cv2.resize(s.astype(np.float32), (len(xs) * STR, len(ys) * STR), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((H, W), np.float32)
    out[WIN // 2:WIN // 2 + full.shape[0], WIN // 2:WIN // 2 + full.shape[1]] = full[:H - WIN // 2, :W - WIN // 2]
    return out, float(np.median(s[np.ix_(iy, ix)]))


def sheet(job):
    idx, img, anyann, qs = common.load(job)
    h0, w0 = img.shape[:2]; s = 2048 / max(h0, w0); W, H = round(w0 * s), round(h0 * s)
    stroke = ink_map_v2(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)).astype(np.float32)
    rows = []
    for ref, cat, tgt, p, box in qs:
        P = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        t = cv2.resize(tgt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        sp = p.shape[1] / W
        M = fill_holes(P > T, int(4096 / (sp * sp)))
        S, rs = sim_map(stroke, [int(round(v * s)) for v in box])
        if S is None or rs is None or rs <= 0:
            continue
        r = S / rs
        rows.append({"sheet": idx, "q": ref, "tp": np.histogram(r[M & t], np.linspace(0, 1.5, 16))[0].tolist(),
                     "fp": np.histogram(r[M & ~t], np.linspace(0, 1.5, 16))[0].tolist(), "label": int(t.sum())})
    return rows


if __name__ == "__main__":
    split = sys.argv[1]
    R = [r for job in common.jobs(split) for r in sheet(job)]
    json.dump(R, open(f"scripts/pp_probe/rows_linesig_{split}.json", "w"))
    tp = np.sum([r["tp"] for r in R], 0); fp = np.sum([r["fp"] for r in R], 0); L = sum(r["label"] for r in R)
    print(f"{split}: similarity / reference's, share of label area (correct | wrong) in the mask:")
    e = np.linspace(0, 1.5, 16)
    for i in range(len(tp)):
        print(f"  {e[i]:.1f}-{e[i+1]:.1f}: correct {tp[i]/L:.4f}  wrong {fp[i]/L:.4f}")
