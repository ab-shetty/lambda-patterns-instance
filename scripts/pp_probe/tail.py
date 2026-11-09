"""Where the tail's vertices are (2026-11-09). Per question, at a given eps: vertices on the main piece's outer ring, on its holes,
on other pieces (outer + holes); hole count and the sizes of holes (as a share of the answer). Usage:
  python3 scripts/pp_probe/tail.py SPLIT MAPS EPS [SHARD NSHARDS] -> scripts/pp_probe/rows_tail_<model>_<split>_e<eps>[_<shard>].json
  python3 scripts/pp_probe/tail.py --summary MODEL EPS"""
import sys, json, glob
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from pathlib import Path


def summary(model, eps):
    for split in ("gemini30", "val", "hf14"):
        rows = sum((json.load(open(f)) for f in sorted(glob.glob(f"scripts/pp_probe/rows_tail_{model}_{split}_e{eps}*.json"))), [])
        rows = [r for r in rows if r["label"] > 0]
        if not rows:
            continue
        lab = np.array([r["label"] for r in rows], float)
        tot = np.array([r["main_outer"] + r["main_holes"] + r["other"] for r in rows], float)
        x = tot / lab
        tail = x >= np.percentile(x, 75)
        def share(k, sel):
            return 100 * np.array([r[k] for r in rows], float)[sel].sum() / tot[sel].sum()
        print(f"{split} eps {eps}: {len(rows)} q | vertices on main outer / main holes / other pieces:"
              f" all {share('main_outer', ~np.zeros_like(tail)):.0f}/{share('main_holes', ~np.zeros_like(tail)):.0f}/{share('other', ~np.zeros_like(tail)):.0f}%"
              f"  worst quarter (>= {np.percentile(x, 75):.1f}x) {share('main_outer', tail):.0f}/{share('main_holes', tail):.0f}/{share('other', tail):.0f}%")
        hs = np.concatenate([r["hole_frac"] for r in rows if r["hole_frac"]] or [np.zeros(0)])
        hv = np.concatenate([r["hole_v"] for r in rows if r["hole_v"]] or [np.zeros(0)])
        lh = np.array([r["label_holes"] for r in rows])
        print(f"   holes: ours {len(hs)} over {len(rows)} q (labels {lh.sum()}); hole area as share of answer: "
              f"median {np.median(hs) if len(hs) else 0:.4f}; holes < 0.1% of answer: {np.mean(hs < 0.001) * 100 if len(hs) else 0:.0f}% of holes, "
              f"{hv[hs < 0.001].sum() / max(1, hv.sum()) * 100:.0f}% of hole vertices; < 1%: {np.mean(hs < 0.01) * 100 if len(hs) else 0:.0f}% / "
              f"{hv[hs < 0.01].sum() / max(1, hv.sum()) * 100:.0f}%")
        for r, xx in sorted(zip(rows, x), key=lambda t: -t[1])[:5]:
            print(f"   worst: sheet {r['sheet']} q {r['q']} {xx:.1f}x  label {r['label']} | main outer {r['main_outer']} holes {r['main_holes']}"
                  f" ({len(r['hole_frac'])}) other {r['other']} ({r['n_other']} pieces)")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2], sys.argv[3]); sys.exit()
    import common
    from curve import polys_from, label_vertices, B
    common.WITH_BOX[0] = True
    split, maps, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])
    shard, nsh = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
    common.MAPS = Path(maps)
    rows = []
    for k, job in enumerate(common.jobs(split)):
        if k % nsh != shard:
            continue
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            s = p4.shape[1] / w
            P = cv2.resize(p4.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_LINEAR)
            polys = polys_from(P, s, {**B, "eps": eps})
            anns = job[1]["annotations"]; anns = json.loads(anns) if isinstance(anns, str) else anns
            r = {"sheet": idx, "q": ref, "label": label_vertices(job[1], cat),
                 "label_holes": sum(1 for a in anns if a.get("category_name", a.get("category")) == "remove"),
                 "main_outer": 0, "main_holes": 0, "other": 0, "n_other": 0, "hole_frac": [], "hole_v": []}
            if polys:
                areas = [cv2.contourArea(np.array(p["outer"], np.int32)) for p in polys]
                tot = sum(areas) or 1
                j = int(np.argmax(areas)); mp = polys[j]
                r["main_outer"] = len(mp["outer"]); r["main_holes"] = sum(len(x) for x in mp["holes"])
                r["hole_frac"] = [cv2.contourArea(np.array(x, np.int32)) / tot for x in mp["holes"]]
                r["hole_v"] = [len(x) for x in mp["holes"]]
                for i, p in enumerate(polys):
                    if i != j:
                        r["other"] += len(p["outer"]) + sum(len(x) for x in p["holes"]); r["n_other"] += 1
            rows.append(r)
    tag = Path(maps).name
    json.dump(rows, open(f"scripts/pp_probe/rows_tail_{tag}_{split}_e{sys.argv[3]}{'_%d' % shard if nsh > 1 else ''}.json", "w"))
    print("done", split, tag, shard, len(rows))
