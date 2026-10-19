"""Leave-one-sheet-out pick over a rows file: for each held-out sheet, choose the variant (or "ref") with the best per-sheet
mean on the other sheets, score it on the held-out sheet. Usage: loso.py rows.json [family-prefix ...]"""
import sys, json
import numpy as np
rows = json.load(open(sys.argv[1])); fams = sys.argv[2:] or [""]
keys = [k for k in rows[0] if k not in ("i", "q") and not k.endswith("_nv")]
by = {}
for r in rows: by.setdefault(r["i"], []).append(r)
S = {i: {k: np.mean([r[k] for r in rs]) for k in keys} for i, rs in by.items()}
for f in fams:
    ks = ["ref"] + [k for k in keys if k != "ref" and k.startswith(f)]
    gain, picks = [], []
    for i in S:
        tr = [j for j in S if j != i]
        best = max(ks, key=lambda k: np.mean([S[j][k] for j in tr]))
        picks.append(best); gain.append(S[i][best] - S[i]["ref"])
    full = max(ks, key=lambda k: np.mean([S[j][k] for j in S]))
    print(f"{f or 'all':8s} LOSO per-sheet gain {np.mean(gain):+.4f} (up {sum(g > .005 for g in gain)} / down {sum(g < -.005 for g in gain)}, "
          f"worst {min(gain):+.3f}); full-data pick {full} ({np.mean([S[j][full] - S[j]['ref'] for j in S]):+.4f}); picks {sorted(set(picks))}")
