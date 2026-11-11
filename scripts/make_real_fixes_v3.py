#!/usr/bin/env python3
"""Label fixes v3 for the real eval sheets (2026-11-09), from the user's edits in Roboflow `floz-eval28-reference`.

The HF eval labels (abshetty/floz-synth-v5 real-world-test) are not edited: instance order fixes every question's box, so the
edits are applied at scoring time (evaluate_refunet_selection.py --label-fixes, pp_probe split valfix3) as
  {"category", "polygon"}   pixels added to that family's target (Roboflow has them, HF does not)
  {"category", "subtract"}  pixels taken out of it (HF has them, Roboflow does not)
  {"category", "ignore"}    either answer is acceptable: out of prediction and target, and a question whose box is mostly
                            inside it is dropped. Roboflow's `ignore` class has no family, so it is "*" (every family on the
                            sheet): bordering is no guide (val 9 / 10's porch roof borders the walls but is ambiguous for the
                            roof), and for the families it is not ambiguous for the model does not select it anyway.
Roboflow is read through the REST API (no export version needed; the agent proxy authenticates):
    python3 scripts/make_real_fixes_v3.py [--api-dir data/roboflow/eval28-api] [--out eval_labels/real_fixes_v3.json]
v2's HF14 fixes (sheets 14, 18) are carried over unchanged, so v3 = v2 + the val edits.
--roboflow-only (v4, 2026-11-11): v2 now lives in Roboflow too (scripts/push_v2_to_roboflow.py), so every sheet comes from
Roboflow alone -- the user's later edits on 14 / 18 count (v4: 18's right roof is an ignore region instead of pattern3b):
    python3 scripts/make_real_fixes_v3.py --roboflow-only --refetch --out eval_labels/real_fixes_v4.json"""
import argparse, glob, json, os, subprocess, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, cv2
from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask

PROJ = "https://api.roboflow.com/perceive-ai/floz-eval28-reference"


def fetch(api_dir):
    os.makedirs(api_dir, exist_ok=True)
    q = f"?api_key={os.environ['ROBOFLOW_API_KEY']}" if os.environ.get("ROBOFLOW_API_KEY") else ""   # no agent proxy on a bare VM
    s = f"{api_dir}/search.json"
    subprocess.run(["curl", "-sS", "-m", "120", "-X", "POST", f"{PROJ}/search{q}", "-H", "Content-Type: application/json",
                    "-d", '{"limit":100,"fields":["id","name"]}', "-o", s], check=True)
    for r in json.load(open(s))["results"]:
        subprocess.run(["curl", "-sS", "-m", "60", f"{PROJ}/images/{r['id']}{q}", "-o", f"{api_dir}/img_{r['id']}.json"], check=True)


def poly_of(bx):
    if "points" in bx:
        return np.array(bx["points"], np.float64)
    cx, cy, w, h = (float(bx[k]) for k in ("x", "y", "width", "height"))      # a rectangle tool box (centre, size)
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def outlines(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0, :].tolist() for c in cs if cv2.contourArea(c) >= 50]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-dir", default="data/roboflow/eval28-api")
    ap.add_argument("--out", default="eval_labels/real_fixes_v3.json")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--roboflow-only", action="store_true", help="no v2 carry-over: Roboflow is the single source (v4)")
    a = ap.parse_args()
    if a.refetch or not glob.glob(f"{a.api_dir}/img_*.json"):
        fetch(a.api_dir)
    recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
    v2 = {} if a.roboflow_only else json.load(open("eval_labels/hf14_fixes_v2.json"))
    out = {"_doc": __doc__.split("\n\n")[0] + " Regenerate with scripts/make_real_fixes_v3.py.",
           **{k: v for k, v in v2.items() if not k.startswith("_")}}
    for f in sorted(glob.glob(f"{a.api_dir}/img_*.json")):
        an = json.load(open(f))["image"]["annotation"]
        i = int(an["key"].split("_")[1].split(".")[0])
        r = recs[i]
        h, w = int(an["height"]), int(an["width"])
        anns = r["annotations"]; anns = json.loads(anns) if isinstance(anns, str) else anns
        hf = {}
        for x in anns:
            hf[x["category_name"]] = hf.get(x["category_name"], np.zeros((h, w), bool)) | \
                render_instance_mask(x["segmentation"], h, w).astype(bool)
        pos, rem, ign = {}, np.zeros((h, w), np.uint8), []
        for bx in an["boxes"]:
            pts = np.round(poly_of(bx)).astype(np.int32)
            if bx["label"] == "remove":
                cv2.fillPoly(rem, [pts], 1)
            elif bx["label"] == "ignore":
                ign.append(pts)
            else:
                cv2.fillPoly(pos.setdefault(bx["label"], np.zeros((h, w), np.uint8)), [pts], 1)
        fx = []
        for c in sorted(set(pos) | set(hf)):
            p = pos.get(c, np.zeros((h, w), np.uint8)).astype(bool) & ~rem.astype(bool)
            q = hf.get(c, np.zeros((h, w), bool))
            fx += [{"category": c, "polygon": o} for o in outlines(p & ~q)]
            fx += [{"category": c, "subtract": o} for o in outlines(q & ~p)]
        for pts in ign:
            fx.append({"category": "*", "ignore": pts.tolist()})
        # an ignore area the label also took out of the family is ignore, not subtract (subtract would be redundant)
        if fx:
            print(i, ", ".join(f"{list(k for k in d if k != 'category')[0]} {d['category']} ({len(list(d.values())[1])} pts)" for d in fx))
            out.setdefault(str(i), [])
            out[str(i)] = [d for d in out[str(i)] if "reassign" in d or str(i) in v2] if str(i) in v2 else []
            out[str(i)] += [d for d in fx if str(i) not in v2]
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
