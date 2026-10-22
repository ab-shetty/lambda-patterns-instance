"""colourcut.py on another model's maps (2026-10-22).
    python3 scripts/pp_probe/colourcut_maps.py MAPS_ROOT SPLIT OUT.json
MAPS_ROOT holds <split>_4096/probs/*.npz (evaluate_refunet_selection.py --save-probs). Writes per-question rows
(base = lin_fill, cc4 / cc6 = the colour cut) and prints the means."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import common
import colourcut

if __name__ == "__main__":
    root, split, out = sys.argv[1:4]
    common.MAPS = Path(root)
    rows = [r for job in common.jobs(split) for r in colourcut.sheet(job)]
    json.dump(rows, open(out, "w"))
    keys = [k for k in rows[0] if k.startswith("cc")]
    print(split, len(rows), "q; base", round(np.mean([r["base"] for r in rows]), 4),
          " ".join(f"{k} {np.mean([r[k] for r in rows]):.4f} ({np.mean([r[k] - r['base'] for r in rows]):+.4f}; "
                   f"{sum(r[k] > r['base'] + 0.005 for r in rows)}/{sum(r[k] < r['base'] - 0.005 for r in rows)})" for k in keys))
