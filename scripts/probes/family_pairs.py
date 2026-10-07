"""How do labelled families on one sheet differ? Same measuring stick for Gemini and synthetic pools.

Per family: line direction and spacing (FFT peak of patches inside the family), fill colour (CIELAB of
the light pixels), and 'lineness' (peak strength). Per pair of families on a sheet: which of
direction / spacing / fill colour differ. Output: share of pairs that differ in exactly one attribute
-- the near-miss pairs the user sees the model merge -- per pool and sheet mode.

usage: python3 family_pairs.py <pool_dir> [<pool_dir> ...] [--json out.json]
"""
import json, glob, os, sys, itertools, numpy as np, cv2
sys.path.insert(0, '.')
from refmask2former.dataset import render_instance_mask

LONG = 2048          # sheets normalised to this long side
P = 96               # patch side
DIR_TOL, SP_TOL, DE_TOL = 15.0, 1.3, 10.0
TX_TOL = 0.2   # probe sheets: identical / direction / spacing / colour pairs <= 0.05, lap vs shingle 0.36, lap vs brick 0.44
# known blind spot: board vs board-and-batten (paired lines) reads as identical


def fam_desc(img, lab, mask, rng):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    dt[:P, :] = 0; dt[-P:, :] = 0; dt[:, :P] = 0; dt[:, -P:] = 0
    ys, xs = np.nonzero(dt > P / 2 + 2)
    if len(ys) == 0:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = np.outer(np.hanning(P), np.hanning(P)).astype(np.float32)
    yy, xx = np.mgrid[-P // 2:P // 2, -P // 2:P // 2]
    rad = np.hypot(yy, xx)
    dirs, sps, strengths, hists, radial = [], [], [], [], []
    for k in rng.choice(len(ys), size=min(8, len(ys)), replace=False):
        cy, cx = ys[k], xs[k]
        p = g[cy - P // 2:cy + P // 2, cx - P // 2:cx + P // 2]
        if p.shape != (P, P) or p.std() < 3:
            continue                      # blank patch: no pattern here
        F = np.abs(np.fft.fftshift(np.fft.fft2((p - p.mean()) * win)))
        F[rad < 3] = 0                    # DC and very long periods
        F[rad > P // 2 - 1] = 0
        iy, ix = np.unravel_index(np.argmax(F), F.shape)
        fy, fx = iy - P // 2, ix - P // 2
        r = np.hypot(fy, fx)
        # FFT peak is the line NORMAL; line direction = normal + 90, mod 180 (0 = horizontal lines)
        ang = (np.degrees(np.arctan2(fy, fx)) + 90) % 180
        dirs.append(ang); sps.append(P / r); strengths.append(F.max() / (F[F > 0].mean() + 1e-6))
        gx = cv2.Sobel(p, cv2.CV_32F, 1, 0); gy = cv2.Sobel(p, cv2.CV_32F, 0, 1)
        mag = np.hypot(gx, gy); th = (np.degrees(np.arctan2(gy, gx)) + 90) % 180   # edge direction
        hh = np.histogram(th, bins=18, range=(0, 180), weights=mag)[0]; hists.append(hh / (hh.sum() + 1e-6))
        # radial profile relative to the main peak: harmonics / second frequencies (paired battens, coursing)
        rp = np.array([F[(rad >= r * k - 0.75) & (rad < r * k + 0.75)].sum() for k in (0.5, 1, 1.5, 2, 3)])
        radial.append(rp / (rp[1] + 1e-6))
    if not dirs:
        return None
    a = np.radians(np.array(dirs) * 2)
    d = (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180
    hist = np.mean(hists, axis=0)
    # rotate so the main line direction sits in bin 0: texture compared independent of direction
    hist = np.roll(hist, -int(round(d / 10)) % 18)
    L = lab[..., 0][mask]
    dark = mask & (lab[..., 0] <= np.percentile(L, 10))
    ink = np.median(lab[dark].reshape(-1, 3), axis=0)
    light = mask & (lab[..., 0] >= np.percentile(L, 60))
    fill = np.median(lab[light].reshape(-1, 3), axis=0)
    return {'dir': float(d), 'sp': float(np.median(sps)), 'line': float(np.median(strengths)),
            'fill': [float(v) for v in fill], 'ink': [float(v) for v in ink], 'hist': [float(v) for v in hist],
            'radial': [float(v) for v in np.median(radial, axis=0)], 'dir_spread': float(np.degrees(np.sqrt(-2 * np.log(max(1e-6, np.hypot(np.sin(a).mean(), np.cos(a).mean()))))) / 2)}


def lab_of(img):
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l[..., 0] *= 100 / 255; l[..., 1:] -= 128
    return l


def compare(a, b):
    dd = abs(a['dir'] - b['dir']); dd = min(dd, 180 - dd)
    sr = max(a['sp'], b['sp']) / min(a['sp'], b['sp'])
    de = float(np.linalg.norm(np.array(a['fill']) - np.array(b['fill'])))
    di = float(np.linalg.norm(np.array(a['ink']) - np.array(b['ink'])))
    ha, hb = np.array(a['hist']), np.array(b['hist'])
    sm = lambda h: (np.roll(h, 1) + 2 * h + np.roll(h, -1)) / 4          # 10-deg bins: tolerate one-bin misalignment
    tx = float(min(np.abs(sm(ha) - np.roll(sm(hb), k)).sum() for k in (-1, 0, 1)))
    rd = float(np.abs(np.log((np.array(a['radial']) + 0.02) / (np.array(b['radial']) + 0.02))).max())
    diff = {'direction': dd > DIR_TOL, 'spacing': sr > SP_TOL, 'colour': de > DE_TOL or di > 2 * DE_TOL,
            'texture': tx > TX_TOL}
    return diff, dd, sr, de, di, tx, rd


def pool_pairs(pool, limit=None):
    out = []
    files = sorted(glob.glob(f'{pool}/annotations/*.json'))[:limit]
    for f in files:
        a = json.load(open(f))
        img0 = cv2.imread(f"{pool}/images/{a['image']['file_name']}")
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]; s = LONG / max(h0, w0)
        img = cv2.resize(img0, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]; lab = lab_of(img)
        fams = {}
        for x in a['annotations']:
            m = render_instance_mask(x['segmentation'], h0, w0).astype(np.uint8)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            fams[x['category_name']] = fams.get(x['category_name'], np.zeros((h, w), bool)) | m
        rng = np.random.default_rng(7)
        desc = {c: fam_desc(img, lab, m, rng) for c, m in sorted(fams.items())}
        desc = {c: d for c, d in desc.items() if d}
        for c1, c2 in itertools.combinations(sorted(desc), 2):
            diff, dd, sr, de, di, tx, rd = compare(desc[c1], desc[c2])
            out.append({'sheet': os.path.basename(f), 'mode': a.get('mode', '?'), 'pair': [c1, c2],
                        'differs': [k for k, v in diff.items() if v], 'ddir': round(dd, 1), 'sp_ratio': round(sr, 2),
                        'dE': round(de, 1), 'dInk': round(di, 1), 'tex': round(tx, 3), 'rad': round(rd, 2), 'desc': [desc[c1], desc[c2]]})
    return out


def summarise(name, rows):
    n = len(rows)
    if not n:
        print(name, 'no pairs'); return
    cnt = {}
    for r in rows:
        k = '+'.join(r['differs']) if r['differs'] else 'none (look alike)'
        cnt[k] = cnt.get(k, 0) + 1
    one = sum(1 for r in rows if len(r['differs']) == 1)
    print(f'{name}: {n} pairs on {len({r["sheet"] for r in rows})} sheets; exactly one attribute differs: {one} ({100 * one / n:.0f}%)')
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f'   {k:32s} {v:4d}  {100 * v / n:4.0f}%')


if __name__ == '__main__':
    args = sys.argv[1:]; js = None
    if '--json' in args:
        i = args.index('--json'); js = args[i + 1]; args = args[:i] + args[i + 2:]
    allrows = {}
    for p in args:
        rows = pool_pairs(p)
        allrows[p] = rows
        summarise(p, rows)
        modes = sorted({r['mode'] for r in rows})
        if len(modes) > 1:
            for m in modes:
                summarise(f'  mode {m}', [r for r in rows if r['mode'] == m])
    if js:
        json.dump(allrows, open(js, 'w'))
