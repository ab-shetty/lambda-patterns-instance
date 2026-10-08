"""Target pixels the new model drops (control kept them) vs pixels both keep: how far is their local colour (CIELAB of a
15 px neighbourhood mean, i.e. fill + ink blended) from the reference box's mean colour? Usage: drop_tone.py <split>"""
import json, glob, sys, io, numpy as np, cv2
sys.path.insert(0, '.')
from refmask2former.dataset import render_instance_mask
split = sys.argv[1]
if split == 'gemini30':
    POOL = 'data/eval_pools/gemini30_s20261006'; files = sorted(glob.glob(f'{POOL}/annotations/*.json'))
    B = 'data/hf_eval/screen1008'
    ARMS = {'r32': f'{B}/revitfailswissr32-2k-e8/evaluations/revitfailswissr32roi_s7_i2048_gemini30',
            'near': f'{B}/revitnear2k-e8/evaluations/revitnearroi_s7_i2048_gemini30'}
    def load(i):
        a = json.load(open(files[i])); return cv2.imread(f"{POOL}/images/{a['image']['file_name']}"), a['annotations']
else:
    import pyarrow.parquet as pq
    T = pq.read_table('data/hf_eval/real-world-test/test-00000.parquet').to_pylist()
    E = 'data/evaluations/screen1008'
    ARMS = {'r32': f'{E}/revitfailswissr32-2k-e8_val4', 'near': f'{E}/revitnear2k-e8_val4'}
    def load(i):
        r = T[i]; img = cv2.imdecode(np.frombuffer(r['image'], np.uint8), cv2.IMREAD_COLOR)
        A = r['annotations']; return img, (json.loads(A) if isinstance(A, str) else A)
sel = {a: {(r['image_index'], r['reference_instance']): r for r in json.load(open(p + '.json'))['selections']} for a, p in ARMS.items()}
keys = sorted(set(sel['r32']) & set(sel['near']))
dk, dd, dl_keep, dl_drop, n_drop, n_keep = [], [], [], [], 0, 0
per_sheet = {}
cache = {}
for (i, q) in keys:
    if i not in cache:
        img, anns = load(i); H0, W0 = img.shape[:2]
        cache = {i: (img, anns, [render_instance_mask(x['segmentation'], H0, W0).astype(np.uint8) for x in anns])}
    img, anns, ms = cache[i]
    cats = [x.get('category_name', 'pattern') for x in anns]
    pr = {a: np.load(f'{p}_probs/{i:03d}_{q:03d}.npz')['prob'] > 89 for a, p in ARMS.items()}
    h, w = pr['r32'].shape
    im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.blur(im, (15, 15)), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255; lab[..., 1:] -= 128
    t = np.zeros((h, w), bool)
    for m, c in zip(ms, cats):
        if c == cats[q]: t |= cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    x, y, bw, bh = sel['r32'][(i, q)]['reference_box_native']; s = w / img.shape[1]
    ref = lab[int(y * s):int((y + bh) * s) + 1, int(x * s):int((x + bw) * s) + 1].reshape(-1, 3).mean(0)
    inner = cv2.distanceTransform(t.astype(np.uint8), cv2.DIST_L2, 3) > 12   # interior: no edge blending
    keep = inner & pr["r32"] & pr["near"]; drop = inner & pr["r32"] & ~pr["near"]
    if drop.sum() > 50 and keep.sum() > 50:
        dE = np.linalg.norm(lab - ref, axis=2)
        dk.append(np.median(dE[keep])); dd.append(np.median(dE[drop]))
        dl_keep.append(np.median(lab[..., 0][keep] - ref[0])); dl_drop.append(np.median(lab[..., 0][drop] - ref[0]))
        per_sheet.setdefault(i, []).append((dE[drop].mean() - dE[keep].mean()))
    n_drop += drop.sum(); n_keep += keep.sum()
dk, dd = np.array(dk), np.array(dd)
print(f'{split}: questions with >50 dropped px: {len(dk)}; dropped share of control-kept target px: {n_drop / (n_drop + n_keep):.3f}')
print(f'  median colour distance to the reference box: kept {np.median(dk):.1f}  dropped {np.median(dd):.1f}  '
      f'(dropped farther on {np.mean(dd > dk):.0%} of questions)')
print(f'  median lightness vs reference: kept {np.median(dl_keep):+.1f}  dropped {np.median(dl_drop):+.1f}')
for i, v in sorted(per_sheet.items()):
    print(f'   sheet {i}: {len(v)} q, dropped minus kept dE {np.mean(v):+.1f}')
