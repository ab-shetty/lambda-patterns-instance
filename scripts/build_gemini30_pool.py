#!/usr/bin/env python3
"""Build data/eval_pools/gemini30_s20261006 (the Gemini30 eval set) from the Roboflow exports, as
pool_manifests/gemini30_s20261006.json describes it:

  r2 sheets  <- floz-gen-gemini-r2 v1
  r3 sheets  <- floz-gen-gemini-r3 v1; #048 also takes the pieces r3 v3 added (the user's 2026-10-07
                fix), APPENDED after its v1 instances so every other question keeps its automatic box
  r4 sheets  <- floz-gen-gemini-r4 v2 (v2 changed only #091)

Annotation files are named as in the manifest (v1 hashes; a sheet whose hash changed is matched by its
floz_gen_NNN prefix); images get the round prefix. Raw exports are read from data/roboflow/<name>-raw
(run_build_realmix.sh downloads them); missing ones are downloaded here.

  python3 scripts/build_gemini30_pool.py [--out data/eval_pools/gemini30_s20261006]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

RF = "data/roboflow"
SRC = {"r2": ("floz-gen-gemini-r2", 1, "floz-gen-gemini-r2"),
       "r3": ("floz-gen-gemini-r3", 1, "floz-gen-gemini-r3"),
       "r4": ("floz-gen-gemini-r4", 2, "floz-gen-gemini-r4v2")}
APPEND = {"r3_floz_gen_048": ("floz-gen-gemini-r3", 3, "floz-gen-gemini-r3v3")}


def download(proj, ver, name):
    loc = f"{RF}/{name}-raw"
    if glob.glob(f"{loc}/*/_annotations.coco.json"):
        return loc
    if os.environ.get("ROBOFLOW_API_KEY"):
        from roboflow import Roboflow
        Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace("perceive-ai").project(proj).version(ver) \
            .download("coco-segmentation", location=loc, overwrite=True)
    else:   # cloud sessions: the agent proxy authenticates the REST API (passing a key there is refused)
        import time
        import urllib.request
        import zipfile
        url = f"https://api.roboflow.com/perceive-ai/{proj}/{ver}/coco-segmentation"
        for _ in range(30):
            d = json.load(urllib.request.urlopen(url))
            if "export" in d:
                break
            time.sleep(10)
        os.makedirs(loc, exist_ok=True)
        z = os.path.join(loc, "export.zip")
        urllib.request.urlretrieve(d["export"]["link"], z)
        zipfile.ZipFile(z).extractall(loc)
        os.remove(z)
    return loc


def convert(raw, tmp):
    """Every split of a raw export -> {floz_gen_NNN_png: (annotation dict, image path)}."""
    out = {}
    for coco in sorted(glob.glob(f"{raw}/*/_annotations.coco.json")):
        split = os.path.basename(os.path.dirname(coco))
        dst = os.path.join(tmp, os.path.basename(raw) + "_" + split)
        subprocess.run([sys.executable, "scripts/roboflow_to_local.py", "--coco", coco,
                        "--img-dir", os.path.dirname(coco), "--out", dst], check=True, stdout=subprocess.DEVNULL)
        for f in glob.glob(f"{dst}/annotations/*.json"):
            a = json.load(open(f))
            out[os.path.basename(f).split(".rf.")[0]] = (a, f"{dst}/images/{a['image']['file_name']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="pool_manifests/gemini30_s20261006.json")
    ap.add_argument("--out", default="data/eval_pools/gemini30_s20261006")
    args = ap.parse_args()
    sheets = json.load(open(args.manifest))["sheets"]
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; remove it to rebuild")
    with tempfile.TemporaryDirectory() as tmp:
        conv = {r: convert(download(*SRC[r]), tmp) for r in SRC}
        extra = {k: convert(download(*v), tmp) for k, v in APPEND.items()}
        os.makedirs(f"{args.out}/annotations")
        os.makedirs(f"{args.out}/images")
        for s in sheets:
            r, key = s[:2], s[3:].split(".rf.")[0]
            ann, img = conv[r][key]
            ann = json.loads(json.dumps(ann))
            fix = f"{r}_{key}"[:len("r3_floz_gen_048")]
            if fix in extra:
                new, _ = extra[fix][key]
                seen = {json.dumps(a["segmentation"]) for a in ann["annotations"]}
                added = [a for a in new["annotations"] if json.dumps(a["segmentation"]) not in seen]
                ann["annotations"] += added
                print(f"{s}: {len(added)} piece(s) appended from {APPEND[fix][2]}")
            fn = f"{r}_{ann['image']['file_name']}"
            shutil.copy(img, f"{args.out}/images/{fn}")
            ann["image"]["file_name"] = fn
            json.dump(ann, open(f"{args.out}/annotations/{s}", "w"))
    print(f"{args.out}: {len(sheets)} sheets")


if __name__ == "__main__":
    main()
