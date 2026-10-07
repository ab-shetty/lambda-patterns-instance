# run from the repo root: PYTHONPATH=.:scripts/probes python3 scripts/probes/within_family.py <pool> ...
"""Within one labelled family: do its pieces change direction or spacing? (A family whose pieces turn teaches the
model that direction is not a cue; one whose spacing jumps teaches spacing is not a cue.) Same descriptor as
family_pairs.py, per piece big enough to measure."""
import json, glob, sys, collections, numpy as np, cv2
sys.path.insert(0, '.')
from family_pairs import fam_desc, lab_of, LONG, DIR_TOL, SP_TOL
from refmask2former.dataset import render_instance_mask

def run(pool):
    res = collections.defaultdict(lambda: collections.Counter())
    for f in sorted(glob.glob(f'{pool}/annotations/*.json')):
        a = json.load(open(f)); img0 = cv2.imread(f"{pool}/images/{a['image']['file_name']}")
        if img0 is None: continue
        h0, w0 = img0.shape[:2]; s = LONG / max(h0, w0)
        img = cv2.resize(img0, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA); h, w = img.shape[:2]; lab = lab_of(img)
        per = collections.defaultdict(list); rng = np.random.default_rng(3)
        for x in a['annotations']:
            m = cv2.resize(render_instance_mask(x['segmentation'], h0, w0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            d = fam_desc(img, lab, m, rng)
            if d and d['line'] > 8: per[x['category_name']].append(d)     # measurable, clearly periodic pieces
        mode = a.get('mode', '?')
        for c, ds in per.items():
            if len(ds) < 2: continue
            dirs = [d['dir'] for d in ds]; sps = [d['sp'] for d in ds]
            turn = max(min(abs(a1 - a2), 180 - abs(a1 - a2)) for a1 in dirs for a2 in dirs) > DIR_TOL
            jump = max(sps) / min(sps) > SP_TOL
            res[mode]['families'] += 1; res[mode]['pieces turn'] += turn; res[mode]['spacing jumps'] += jump
    return res

for pool in sys.argv[1:]:
    for mode, c in sorted(run(pool).items()):
        n = c['families']
        if n: print(f'{pool.split("/")[-1][:34]:34s} {mode:10s} families with 2+ measurable pieces: {n:4d}; pieces turn: {100 * c["pieces turn"] / n:3.0f}%; spacing jumps > {SP_TOL}x: {100 * c["spacing jumps"] / n:3.0f}%')
