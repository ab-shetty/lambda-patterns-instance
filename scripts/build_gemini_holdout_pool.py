#!/usr/bin/env python3
"""The hand-labelled Gemini sheets of rounds 2-4 MINUS the 30 Gemini30 eval sheets (2026-10-21).

Gemini30 (pool_manifests/gemini30_s20261006.json) is a random 30 of the r2 + r3 + r4 labelled sheets, so any training
pool that holds those rounds leaks the eval set. This writes the other sheets (139 of 169 with r3 v3 / r4 v2) as a
local pool, excluding by round + sheet number (floz_gen_NNN), whatever the export's hash suffix.

    python3 scripts/build_gemini_holdout_pool.py [--out data/roboflow/floz-gen-gemini-r234-no30-clean]
Reads data/roboflow/floz-gen-gemini-{r2,r3v3,r4v2}-clean (run_gemini_holdout.sh downloads and converts them).
"""
import argparse
import json
import os
import re
import shutil

RF = "data/roboflow"
SRC = {"r2": f"{RF}/floz-gen-gemini-r2-clean", "r3": f"{RF}/floz-gen-gemini-r3v3-clean",
       "r4": f"{RF}/floz-gen-gemini-r4v2-clean"}
MANIFEST = "pool_manifests/gemini30_s20261006.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{RF}/floz-gen-gemini-r234-no30-clean")
    a = ap.parse_args()
    held = set()
    for f in json.load(open(MANIFEST))["sheets"]:
        m = re.match(r"(r\d)_(floz_gen_\d+)", f)
        held.add((m.group(1), m.group(2)))
    assert len(held) == 30, len(held)
    os.makedirs(f"{a.out}/images", exist_ok=True)
    os.makedirs(f"{a.out}/annotations", exist_ok=True)
    kept, dropped = 0, 0
    for rnd, src0 in SRC.items():
        # a merged export, or its train / valid / test parts (r3 v3 is split)
        srcs = [src0] if os.path.isdir(f"{src0}/annotations") else \
            [f"{src0[:-len('-clean')]}-{sp}-clean" for sp in ("train", "valid", "test")
             if os.path.isdir(f"{src0[:-len('-clean')]}-{sp}-clean/annotations")]
        for src, x in [(d, x) for d in srcs for x in sorted(os.listdir(f"{d}/annotations"))]:
            sheet = re.match(r"(floz_gen_\d+)", x).group(1)
            if (rnd, sheet) in held:
                dropped += 1
                continue
            A = json.load(open(f"{src}/annotations/{x}"))
            img = A["image"]["file_name"]
            A["image"]["file_name"] = f"{rnd}_{img}"
            shutil.copy(f"{src}/images/{img}", f"{a.out}/images/{rnd}_{img}")
            json.dump(A, open(f"{a.out}/annotations/{rnd}_{x}", "w"))
            kept += 1
    assert dropped == 30, f"expected to drop the 30 Gemini30 sheets, dropped {dropped}"
    print(f"{a.out}: {kept} sheets (dropped the 30 Gemini30 sheets)")


if __name__ == "__main__":
    main()
