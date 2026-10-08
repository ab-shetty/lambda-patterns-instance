"""Gemini30: per piece of the asked family (other than the reference piece), recall under each 2k model, vs how the
piece differs from the reference piece (direction / spacing / fill / ink / texture descriptors)."""
import json, glob, sys, collections, numpy as np, cv2
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/probes')
from family_pairs import fam_desc, lab_of, compare, LONG
from refmask2former.dataset import render_instance_mask
POOL = 'data/eval_pools/gemini30_s20261006'; files = sorted(glob.glob(f'{POOL}/annotations/*.json'))
B = 'data/hf_eval/screen1008'
ARMS = {'r32': f'{B}/revitfailswissr32-2k-e8/evaluations/revitfailswissr32roi_s7_i2048_gemini30',
        'noelev': f'{B}/revitnearnoelev2k-e8/evaluations/revitnearnoelevroi_s7_i2048_gemini30',
        'near': f'{B}/revitnear2k-e8/evaluations/revitnearroi_s7_i2048_gemini30'}
rows = {a: {(r['image_index'], r['reference_instance']) for r in json.load(open(p + '.json'))['selections']} for a, p in ARMS.items()}
keys = sorted(set.intersection(*rows.values()))
out = []
cache = {}
for (i, q) in keys:
    if i not in cache:
        a = json.load(open(files[i])); H0, W0 = a['image']['height'], a['image']['width']
        img0 = cv2.imread(f"{POOL}/images/{a['image']['file_name']}"); s = LONG / max(H0, W0)
        img = cv2.resize(img0, (round(W0 * s), round(H0 * s)), interpolation=cv2.INTER_AREA); lab = lab_of(img)
        ms = [render_instance_mask(x['segmentation'], H0, W0).astype(np.uint8) for x in a['annotations']]
        ds = [fam_desc(img, lab, cv2.resize(m, img.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool), np.random.default_rng(5)) for m in ms]
        cache = {i: (a, ms, ds)}
    a, ms, ds = cache[i]
    cats = [x['category_name'] for x in a['annotations']]
    probs = {k: np.load(f'{p}_probs/{i:03d}_{q:03d}.npz')['prob'] > 89 for k, p in ARMS.items()}
    h, w = probs['r32'].shape
    for j, m in enumerate(ms):
        if j == q or cats[j] != cats[q]:
            continue
        mm = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mm.sum() < 50:
            continue
        rec = {k: float(p[mm].mean()) for k, p in probs.items()}
        diff = None
        if ds[q] and ds[j]:
            d, dd, sr, de, di, tx = compare(ds[q], ds[j])[:6]
            diff = [k for k, v in d.items() if v]
        out.append({'sheet': i, 'q': q, 'piece': j, 'area': int(mm.sum()), 'recall': rec, 'differs': diff})
json.dump(out, open('data/evaluations/piece_recall.json', 'w'))
g = collections.defaultdict(list)
for o in out:
    k = 'unmeasured' if o['differs'] is None else ('+'.join(o['differs']) if o['differs'] else 'same look')
    g[k].append(o)
print(f"{'piece vs reference piece':28s} {'n':>4s}  recall r32  noelev  near   (area-weighted)")
for k, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
    wts = np.array([o['area'] for o in v], float)
    r = {a: np.average([o['recall'][a] for o in v], weights=wts) for a in ARMS}
    print(f"{k:28s} {len(v):4d}  {r['r32']:.3f}  {r['noelev']:.3f}  {r['near']:.3f}")
