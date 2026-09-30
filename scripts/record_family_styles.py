"""Run generate_synthetic_fc.py and also record every family's material, spacing and base
colour, which the annotations do not carry (they give only the family role).

    REC_DIR=out/styles python3 scripts/record_family_styles.py <generate_synthetic_fc.py args>

Writes REC_DIR/<pid>.jsonl, one line per sheet (floor plans: one line for the hardscape, one
for the interior floors), keyed by image id:
    elevation: {"id", "mode", "shaded", "styles": {family: {kind, base, params}}}
    freeform:  {"id", "mode", "hard": {kind, base, params}} / {"id", "mode", "floors": [...]}
Same arguments and seed give byte-identical images and annotations to a plain run.
Elevation styles are recorded after --distinct-looks (so it must be on, the default).
Spacing: revit_render.SPACING names each kind's spacing param (ft).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REC = os.environ["REC_DIR"]
os.makedirs(REC, exist_ok=True)

import generate_synthetic_fc as FCm  # noqa: E402
import revit_plans as RP  # noqa: E402
import revit_render as RR  # noqa: E402

CUR = [None]


def _out(obj):
    with open(os.path.join(REC, f"{os.getpid()}.jsonl"), "a") as f:
        f.write(json.dumps(obj) + "\n")


def _sty(s):
    return {"kind": s.kind, "base": list(s.base),
            "params": {k: v for k, v in s.params.items() if isinstance(v, (int, float))}}


_compose = RR.compose
def compose(image_id, seed, mw):
    CUR[0] = image_id
    return _compose(image_id, seed, mw)


_distinct_looks = RR.distinct_looks
def distinct_looks(styles, shaded, app, fam_seed, S_guess):
    _distinct_looks(styles, shaded, app, fam_seed, S_guess)
    _out({"id": CUR[0], "mode": "elevation", "shaded": bool(shaded),
          "styles": {k: _sty(v) for k, v in styles.items()}})


_hard_style = RP._hard_style
def hard_style(kind, r, lw, seed, colour):
    s = _hard_style(kind, r, lw, seed, colour)
    _out({"id": CUR[0], "mode": "freeform", "hard": _sty(s)})
    return s


_separate_all = RP._separate_all
def separate_all(mats):
    done = _separate_all(mats)
    _out({"id": CUR[0], "mode": "freeform",
          "floors": [{"kind": k, "base": list(b),
                      "params": {kk: vv for kk, vv in p.items() if isinstance(vv, (int, float)) and kk != "seed"}}
                     for k, p, b in done]})
    return done


RR.compose, RR.distinct_looks = compose, distinct_looks
RP._hard_style, RP._separate_all = hard_style, separate_all

if __name__ == "__main__":
    FCm.main()
