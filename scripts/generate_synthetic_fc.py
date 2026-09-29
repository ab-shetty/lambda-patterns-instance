"""FreeCAD-geometry pilot: v6 sheets whose elevations come from a fused 3D model.

Everything a sheet looks like -- materials and hatching, colour, ink, text,
annotation, labels, crop -- is generate_synthetic_v6.py with the given flags.
Only elevation geometry changes: each v6 house (same blocks, pitches,
overhangs, chimney) is built as 3D solids in FreeCAD (scripts/fc_massing.py),
fused, and every view is the set of faces actually visible from that side
(planar depth test between overlapping faces). Wings and garages now meet the
main roof properly, hip ends foreshorten correctly, rakes and fascias are the
real roof slab edges, and side views show the roofs of blocks behind.
Roof plans and floor plans are v6's own.

  python scripts/generate_synthetic_fc.py --out data/synthetic/fc_probe300 --n 300 --start 200000 --r8
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_synthetic_v6 as G  # noqa: E402

FREECADCMD = os.environ.get("FREECADCMD", "/opt/fc/bin/freecadcmd")
HERE = os.path.dirname(os.path.abspath(__file__))
R8 = dict(val_details=True, material_mix=True, val_fills=True, res_degrade=0.15, neutral_palette=True,
          fill_scale=True, real_labelling=True)
TOWARD = {"front": (0, -1), "rear": (0, 1), "left": (-1, 0), "right": (1, 0)}   # plan dir to the viewer

_FACES = {}
_CUR = [None]


# ----------------------------------------------------------------------------
# houses: reproduce compose()'s rng prefix so the FreeCAD model is the house
# compose() will draw
# ----------------------------------------------------------------------------
def house_for(image_id, seed, mw):
    rng = random.Random(seed * 1_000_003 + image_id)
    rng.choices(list(mw.keys()), weights=list(mw.values()))
    G.make_appearance(rng)
    return G.make_house(rng)


SHAPED = [False]      # --shaped: notched / U / chamfered / bayed houses (3D), set in main()


def _rect_block(b, x0, y0, x1, y1, parent):
    nb = dict(b, x0=x0, y0=y0, w=x1 - x0, d=y1 - y0, parent=parent)
    nb.pop("poly", None)
    if b["roof"] in ("gable", "hip"):
        nb["ridge"] = "x" if (x1 - x0) >= (y1 - y0) else "y"
    return nb


def _poly_block(b, pts, sloped, parent, **kw):
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    q = orient(Polygon(pts), 1.0)
    cc = [tuple(round(v, 4) for v in p) for p in list(q.exterior.coords)[:-1]]
    x0, y0, x1, y1 = q.bounds
    nb = dict(b, x0=x0, y0=y0, w=x1 - x0, d=y1 - y0, parent=parent, poly=cc, **kw)
    n = len(cc)
    nb["sloped"] = [i for i in range(n) if sloped(cc[i], cc[(i + 1) % n])]
    return nb


def shape_blocks(blocks, house, r):
    """--shaped: turn the main block into an L (notch), a U (rear courtyard) or a chamfered
    polygon, and add a 45-degree bay. Pieces keep `parent` (the v6 block they came from)
    so elevation zoning still uses the right Block."""
    from shapely.geometry import box as sbox
    main, rest = blocks[0], blocks[1:]
    x0, y0, x1, y1 = main["x0"], main["y0"], main["x0"] + main["w"], main["y0"] + main["d"]
    others = [sbox(b["x0"], b["y0"], b["x0"] + b["w"], b["y0"] + b["d"]).buffer(0.5) for b in rest]

    def free(g):
        return not any(g.intersects(o) for o in others)
    pieces = [dict(main, parent=0)]
    op = r.random()
    w, d = x1 - x0, y1 - y0
    if op < 0.30 and w >= 28 and d >= 24:                         # L: notch a corner
        u, v = r.uniform(0.25, 0.42) * w, r.uniform(0.3, 0.48) * d
        opts = []
        for right in (True, False):
            for front in (True, False):
                nx = (x1 - u, x1) if right else (x0, x0 + u)
                ny = (y0, y0 + v) if front else (y1 - v, y1)
                if free(sbox(nx[0], ny[0], nx[1], ny[1])):
                    opts.append((right, front, nx, ny))
        if opts:
            right, front, nx, ny = r.choice(opts)
            A = (x0, y0, x1 - u, y1) if right else (x0 + u, y0, x1, y1)
            B = (nx[0], y0 + v, nx[1], y1) if front else (nx[0], y0, nx[1], y1 - v)
            pieces = [_rect_block(main, *A, 0), _rect_block(main, *B, 0)]
    elif op < 0.42 and w >= 34 and d >= 26:                       # U: courtyard into the rear
        cw, cd = r.uniform(0.3, 0.45) * w, r.uniform(0.35, 0.5) * d
        cx = x0 + (w - cw) / 2 + r.uniform(-0.1, 0.1) * w
        if free(sbox(cx, y1 - cd, cx + cw, y1)):
            pieces = [_rect_block(main, x0, y0, x1, y1 - cd, 0), _rect_block(main, x0, y1 - cd, cx, y1, 0),
                      _rect_block(main, cx + cw, y1 - cd, x1, y1, 0)]
    elif op < 0.65 and main["roof"] in ("hip", "gable", "flat"):  # chamfered corners
        corners = [(x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)]
        cut = [c for c in corners if free(sbox(min(c[0], c[0] + 6 * c[2]), min(c[1], c[1] + 6 * c[3]),
                                                max(c[0], c[0] + 6 * c[2]), max(c[1], c[1] + 6 * c[3])))]
        r.shuffle(cut)
        cut = cut[:r.choice([1, 1, 2])]
        pts = []
        for (cx, cy, sx, sy) in corners:
            if (cx, cy, sx, sy) in cut:
                k = r.uniform(3, 6)
                a = (cx, cy + sy * k) if (sx * sy) > 0 else (cx + sx * k, cy)
                b = (cx + sx * k, cy) if (sx * sy) > 0 else (cx, cy + sy * k)
                pts += [a, b]
            else:
                pts.append((cx, cy))
        if cut:
            ridge_x = main["ridge"] == "x"

            def sl(p, q):
                if main["roof"] == "hip":
                    return True
                dx, dy = abs(q[0] - p[0]), abs(q[1] - p[1])
                if dx > 1e-6 and dy > 1e-6:
                    return True                                      # chamfer edges carry a small hip
                return (dy < 1e-6) if ridge_x else (dx < 1e-6)       # eave edges run parallel to the ridge
            pieces = [_poly_block(main, pts, sl, 0)]
    # 45-degree bay on the front or rear wall of the largest piece
    if r.random() < 0.25 and main["roof"] != "flat":
        P = max(pieces, key=lambda b: b["w"] * b["d"])
        px0, px1 = P["x0"], P["x0"] + P["w"]
        wy = r.choice([P["y0"], P["y0"] + P["d"]])
        out = -1 if wy == P["y0"] else 1
        bw, bd = r.uniform(7, 12), r.uniform(2.5, 4)
        bd = min(bd, (bw - 2.0) / 2)                                # keep the trapezoid's outer face >= 2 ft
        if px1 - px0 > bw + 6:
            a = r.uniform(px0 + 2, px1 - 2 - bw)
            pts = [(a, wy), (a + bd, wy + out * bd), (a + bw - bd, wy + out * bd), (a + bw, wy)]
            if free(sbox(a, min(wy, wy + out * bd), a + bw, max(wy, wy + out * bd))):
                Hb = min(main["H"], house["fh"] + house["plate"])

                def sl_bay(p, q):
                    return not (abs(p[1] - wy) < 1e-3 and abs(q[1] - wy) < 1e-3)   # not the house-side edge
                pieces.append(_poly_block(main, pts, sl_bay, 0, H=Hb, roof="hip", ov=0.6))
    for i, b in enumerate(rest):
        b.setdefault("parent", i + 1)
    return pieces + rest


def massing_spec(image_id, seed, house):
    r = random.Random(seed * 1_000_003 + image_id * 151 + 7)
    blocks = []
    for b in house["blocks"]:
        H = b.stories * house["fh"] + house["plate"]
        # wings of a flat-roofed house inherit pitch 0 (v6 draws them as a flat line)
        pitch = b.pitch if b.roof == "flat" or b.pitch >= 0.1 else r.uniform(0.25, 0.45)
        blocks.append({"x0": b.x0, "y0": b.y0, "w": b.w, "d": b.d, "H": H, "roof": b.roof, "ridge": b.ridge,
                       "pitch": pitch, "ov": b.ov, "parapet": r.uniform(1.0, 2.5)})
    if SHAPED[0]:
        # own rng stream: the rest of the spec (chimney) is unchanged by the flag
        blocks = shape_blocks(blocks, house, random.Random(seed * 1_000_003 + image_id * 211 + 3))
    spec = {"id": image_id, "blocks": blocks}
    if house["chimney"]:
        m = blocks[0]
        cw, cd = r.uniform(2.0, 3.5), r.uniform(1.8, 3.0)
        along_x = m["ridge"] == "x"
        span = m["d"] if along_x else m["w"]
        rise = {"gable": m["pitch"] * span / 2, "hip": m["pitch"] * span / 2, "shed": m["pitch"] * span}.get(m["roof"], 0)
        if along_x:
            cx = r.choice([m["x0"] + r.uniform(2, 6), m["x0"] + m["w"] - r.uniform(2, 6) - cw])
            cy = m["y0"] + m["d"] / 2 + r.uniform(-0.25, 0.1) * m["d"]
        else:
            cy = r.choice([m["y0"] + r.uniform(2, 6), m["y0"] + m["d"] - r.uniform(2, 6) - cd])
            cx = m["x0"] + m["w"] / 2 + r.uniform(-0.25, 0.1) * m["w"]
        spec["chimney"] = {"x0": cx, "y0": cy, "w": cw, "d": cd, "z0": m["H"] - 2,
                           "z1": m["H"] + rise + r.uniform(1.5, 3.5)}
    return spec


def run_freecad(specs):
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.json"), os.path.join(td, "out.json")
        json.dump(specs, open(src, "w"))
        code = f"import sys; sys.argv=['x', {src!r}, {dst!r}]; exec(open({os.path.join(HERE, 'fc_massing.py')!r}).read())"
        p = subprocess.run([FREECADCMD, "-c", code], capture_output=True, text=True)
        if not os.path.exists(dst):
            raise RuntimeError("freecadcmd failed:\n" + p.stdout[-2000:] + p.stderr[-2000:])
        print([l for l in p.stdout.splitlines() if l.startswith("fc_massing")][-1], flush=True)
        return json.load(open(dst))


# ----------------------------------------------------------------------------
# projection + visibility
# ----------------------------------------------------------------------------
def _uvt(view, x, y, z):
    u, t = G.view_map(view)(x, y)
    return u, -z, t


def visible_faces(faces, view, specs_blocks, chim):
    tx, ty = TOWARD[view]
    cand = []
    for f in faces:
        n = f["n"]
        dot = n[0] * tx + n[1] * ty
        if dot < 0.02:
            continue
        rings = [[_uvt(view, *p) for p in lp] for lp in f["loops"]]
        try:
            poly = Polygon([(u, v) for u, v, _ in rings[0]], [[(u, v) for u, v, _ in r] for r in rings[1:]]).buffer(0)
        except Exception:
            continue
        if poly.is_empty or poly.area < 1e-3:
            continue
        P = np.array([p for r in rings for p in r])
        A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        plane = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
        cand.append({"poly": poly, "plane": plane, "n": n, "c": f["c"], "frontal": dot > 0.98})

    def depth(fc, pt):
        return fc["plane"][0] * pt.x + fc["plane"][1] * pt.y + fc["plane"][2]

    out = []
    for i, F in enumerate(cand):
        occ = []
        for j, Gf in enumerate(cand):
            if i == j or not F["poly"].intersects(Gf["poly"]):
                continue
            ov = F["poly"].intersection(Gf["poly"])
            if ov.area < 1e-4:
                continue
            pt = ov.representative_point()
            if depth(Gf, pt) < depth(F, pt) - 1e-4:
                occ.append(Gf["poly"])
        vis = F["poly"].difference(unary_union(occ)) if occ else F["poly"]
        if vis.area < 1e-3:
            continue
        F = dict(F, vis=vis)
        F["kind"] = classify(F, specs_blocks, chim)
        out.append(F)
    return out


def classify(F, blocks, chim):
    n, (cx, cy, cz) = F["n"], F["c"]
    if abs(n[2]) > 0.05:
        return ("roof", None)
    if chim and chim["x0"] - 0.05 <= cx <= chim["x0"] + chim["w"] + 0.05 and \
            chim["y0"] - 0.05 <= cy <= chim["y0"] + chim["d"] + 0.05 and cz > chim["z0"]:
        return ("chimney", None)
    for bi, b in enumerate(blocks):
        if b.get("poly"):
            from shapely.geometry import Point, Polygon
            if Polygon(b["poly"]).buffer(0.05).contains(Point(cx, cy)):
                return ("wall", b.get("parent", bi))
        elif b["x0"] - 0.05 <= cx <= b["x0"] + b["w"] + 0.05 and b["y0"] - 0.05 <= cy <= b["y0"] + b["d"] + 0.05:
            return ("wall", b.get("parent", bi))
    return ("trim", None)           # rake boards / fascia: slab edges outside every footprint


# ----------------------------------------------------------------------------
# the v6 elevation dict, from visible faces
# ----------------------------------------------------------------------------
def build_elevation_fc(house, view, rng):
    spec, faces = _FACES[_CUR[0]]
    blocks, chim = spec["blocks"], spec.get("chimney")
    vf = visible_faces(faces, view, blocks, chim)
    fh = house["fh"]
    surfaces, trims, openings = [], [], []
    # view-space order of v6's Block objects matches spec["blocks"]
    walls_by_block = {}
    for F in vf:
        kind, bi = F["kind"]
        if kind == "roof":
            for p in G.polys_of(F["vis"]):
                surfaces.append(("roof", p, bi if bi is not None else 0, "roof"))
            # fascia band along the eave (lowest, horizontal edge of the plane), seen from the eave side
            if F["frontal"] is False and abs(F["n"][2]) < 0.97:
                x0, y0, x1, y1 = F["poly"].bounds
                eave = box(x0, y1 - 0.02, x1, y1 + rng.uniform(0.35, 0.7)).intersection(F["vis"].buffer(0.8))
                if not eave.is_empty and eave.area > 0.2:
                    trims.append(eave)
        elif kind == "chimney":
            for p in G.polys_of(F["vis"]):
                surfaces.append(("chimney", p, 0, "chimney"))
        elif kind == "trim":
            trims.append(F["vis"])
        else:
            walls_by_block.setdefault(bi, []).append(F)
    if chim:
        cs = [p for (f, p, _, _) in surfaces if f == "chimney"]
        if cs:
            b = unary_union(cs).bounds
            trims.append(box(b[0] - 0.3, b[1], b[2] + 0.3, b[1] + 0.5))
    acc = house["accent"]
    for bi, Fs in walls_by_block.items():
        B = house["blocks"][bi]
        wall = unary_union([F["vis"] for F in Fs])
        if wall.is_empty:
            continue
        u0, y_top_all, u1, _ = wall.bounds
        H = B.stories * fh + house["plate"]
        y_top = -H
        belt_polys = []
        if house["belt"] and B.stories >= 2:
            for s in range(1, B.stories):
                yb = -s * fh
                belt_polys.append(G.rect(u0 - 0.2, yb - house["belt_h"] / 2, u1 + 0.2, yb + house["belt_h"] / 2))

        def zone_for(placement):
            if placement == "gable" and wall.bounds[1] < y_top - 1.0:
                belt_polys.append(G.rect(u0 - 0.2, y_top - 0.1, u1 + 0.2, y_top + rng.uniform(0.4, 0.8)))
                return G.rect(u0 - 1, y_top, u1 + 1, y_top - 100)
            if placement == "wainscot":
                belt_polys.append(G.rect(u0 - 0.2, -house["wainscot_h"] - 0.3, u1 + 0.2, -house["wainscot_h"]))
                return G.rect(u0 - 1, 0.5, u1 + 1, -house["wainscot_h"])
            if placement == "upper" and B.stories >= 2:
                if not house["belt"]:
                    belt_polys.append(G.rect(u0 - 0.2, -fh - 0.3, u1 + 0.2, -fh + 0.3))
                return G.rect(u0 - 1, -fh, u1 + 1, -200)
            if placement == "block" and B.kind != "main":
                return G.rect(u0 - 1, 1, u1 + 1, -200)
            return None

        accent_zone = zone_for(acc) if acc else None
        accent2_zone = zone_for(house["accent2"]) if house.get("accent2") else None
        found = G.rect(u0 - 0.1, 0, u1 + 0.1, -house["foundation"]) if house["foundation"] > 0 else None
        belts = unary_union(belt_polys).intersection(wall.buffer(0.2)) if belt_polys else None
        main_zone = wall
        if found is not None:
            main_zone = main_zone.difference(found)
        if belts is not None:
            main_zone = main_zone.difference(belts)
        for zname, zone in (("accent", accent_zone), ("accent2", accent2_zone)):
            if zone is not None:
                for p in G.polys_of(main_zone.intersection(zone)):
                    surfaces.append((zname, p, bi, "wall"))
                main_zone = main_zone.difference(zone)
        for p in G.polys_of(main_zone):
            surfaces.append(("main", p, bi, "wall"))
        if found is not None:
            for p in G.polys_of(wall.intersection(found)):
                surfaces.append(("foundation", p, bi, "found"))
        if belts is not None:
            trims += G.polys_of(belts)
        openings += place_openings(house, B, bi, view, wall, Fs, rng)
    # porch on the front: nearest of all (v6's 2D porch)
    porch = None
    if house["porch"] and view == "front" and 0 in walls_by_block:
        mb = house["blocks"][0]
        u0, u1 = mb.x0, mb.x1
        pw = rng.uniform(8, min(24, mb.w - 4))
        pu0 = min(max(u0 + house["front_door_u"] * mb.w - pw / 2, u0 + 1), u1 - pw - 1)
        top = -(fh - rng.uniform(0.2, 1.0))
        proof = G.rect(pu0 - 1, top, pu0 + pw + 1, top - rng.uniform(1.5, 3.5))
        porch = {"roof": proof, "posts": [G.rect(pu0 + 0.3, 0, pu0 + 0.9, top), G.rect(pu0 + pw - 0.9, 0, pu0 + pw - 0.3, top)],
                 "fascia": G.rect(pu0 - 1, top, pu0 + pw + 1, top + 0.5)}
        cut = proof.union(porch["fascia"])
        surfaces = [(f, q, bi, k) for (f, p, bi, k) in surfaces for q in G.polys_of(p.difference(cut))]
        surfaces.append(("roof", proof, -1, "porch"))
        trims.append(porch["fascia"])
        trims += porch["posts"]
    trims = [q for t in trims for q in G.polys_of(t) if q.area > 0.05]
    allp = [p for (_, p, _, _) in surfaces] + trims
    return {"surfaces": surfaces, "trims": trims, "openings": openings, "porch": porch,
            "extent": unary_union(allp).bounds, "vf": vf}


def place_openings(house, B, bi, view, wall, Fs, rng):
    fh = house["fh"]
    out = []
    inner = wall.buffer(-0.6)
    # frontal wall faces only (each is one planar facade)
    faces = [F for F in Fs if F["frontal"]]
    if not faces:
        return out
    fu0, _, fu1, _ = unary_union([F["vis"] for F in faces]).bounds
    width = fu1 - fu0
    front_main = view == "front" and B.kind == "main"
    garage_face = view == "front" and B.kind == "garage"

    def ok(p, margin=0.0):
        return p.within(inner) and not any(p.buffer(0.8).intersects(o["poly"]) for o in out)

    if house.get("units") and B.kind == "main" and view in ("front", "rear"):
        n = house["units"]
        uw = width / n
        tmpl = house["unit_template"]
        head0 = -rng.uniform(6.8, 7.2)
        for i in range(n):
            base = fu0 + i * uw
            for kind, off, sw in tmpl["ground"]:
                if kind == "door" and view == "front":
                    p, t = G.rect(base + off, 0, base + off + sw, -6.8), "door"
                elif kind == "garage" and view == "front":
                    p, t = G.rect(base + off, 0, base + off + sw, -7.0), "garage"
                else:
                    ww = sw if kind == "win" else 3.0
                    p, t = G.rect(base + off, head0, base + off + ww, head0 + 4.0), "window"
                if p.within(wall.buffer(0.01)):
                    out.append({"poly": p, "type": t, "block": bi})
            for st in range(1, B.stories):
                for kind, off, sw in tmpl["upper"]:
                    head = -st * fh - 7.0
                    p = G.rect(base + off, head, base + off + sw, head + 4.0)
                    if ok(p):
                        out.append({"poly": p, "type": "window", "block": bi})
        return out
    if front_main:
        dw = 3.0 if rng.random() < 0.7 else 6.0
        du = min(max(fu0 + house["front_door_u"] * width, fu0 + 2.5), fu1 - 2.5 - dw)
        p = G.rect(du, 0, du + dw, -6.8)
        if p.within(wall.buffer(0.01)):
            out.append({"poly": p, "type": "door", "block": bi})
    if garage_face and width > 11:
        gdw = min(8.0 if width < 20 or rng.random() < 0.4 else 16.0, width - 3)
        gu = fu0 + (width - gdw) / 2 + rng.uniform(-1, 1)
        p = G.rect(gu, 0, gu + gdw, -7.0)
        if p.within(wall.buffer(0.01)):
            out.append({"poly": p, "type": "garage", "block": bi})
    for s in range(B.stories):
        if B.kind == "garage" and s == 0 and garage_face:
            continue
        n = max(0, int(width / rng.uniform(6, 11)))
        if B.kind == "garage":
            n = min(n, 2)
        for i in range(n):
            cu = fu0 + (i + 0.5) * width / n + rng.uniform(-1.0, 1.0)
            ww = rng.choice([2.5, 3.0, 3.0, 4.0, 5.0, 6.0])
            wh = rng.choice([3.0, 3.5, 4.0, 4.5, 5.0])
            head = -s * fh - rng.uniform(6.6, 7.4)
            if s == 0 and house["accent"] == "wainscot" and head + wh > -house["wainscot_h"] + 0.3:
                wh = min(wh, -house["wainscot_h"] - head - 0.3)
                if wh < 2.0:
                    continue
            p = G.rect(cu - ww / 2, head, cu + ww / 2, head + wh)
            if ok(p):
                out.append({"poly": p, "type": "window", "block": bi})
    # gable-end attic window where the facade rises into a gable
    x0, y0, x1, _ = wall.bounds
    H = B.stories * fh + house["plate"]
    if y0 < -H - 3.5 and rng.random() < 0.4:
        cu = (x0 + x1) / 2
        p = G.rect(cu - 1.2, -H - 1.0, cu + 1.2, -H - 3.2)
        if ok(p):
            out.append({"poly": p, "type": "window", "block": bi})
    return out


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def _init(out, seed, mw, flags, faces, revit=False, revit_plans=False):
    G._init(out, seed, mw, **flags)
    _FACES.update(faces)
    G.build_elevation = build_elevation_fc
    if revit:
        import revit_render
        revit_render.install(G, sys.modules[__name__], plans=revit_plans)


def _job(image_id):
    if image_id not in _FACES:
        return (image_id, False, "no massing")
    _CUR[0] = image_id
    return G._job(image_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--mode-weights", default="86,7,7")
    ap.add_argument("--r8", action="store_true", help="the r8 flag set (best probe pool so far)")
    ap.add_argument("--tight-crop", action="store_true")
    ap.add_argument("--real-labelling", action="store_true",
                    help="v6's --real-labelling alone (walls labelled, mostly not trim/chimney/foundation)")
    ap.add_argument("--revit", action="store_true",
                    help="elevations drawn Revit-style from the 3D model: cast shadows, line-weight "
                         "hierarchy, level datums, view titles, restrained materials")
    ap.add_argument("--revit-plans", action="store_true",
                    help="with --revit: roof plans (from the 3D model's roof faces) and floor plans drawn "
                         "Revit-style too (scripts/revit_plans.py); off = v6 plans, byte-identical")
    ap.add_argument("--shaped", action="store_true",
                    help="3D houses beyond boxes: L (notch), U (rear courtyard), chamfered corners, 45-degree bays; "
                         "roofs over convex polygons as the lower envelope of edge slope planes (fc_massing.py). "
                         "Off = byte-identical")
    args = ap.parse_args()
    SHAPED[0] = args.shaped
    if args.revit_plans and not args.revit:
        ap.error("--revit-plans needs --revit")
    e, r, f = [float(v) for v in args.mode_weights.split(",")]
    mw = {"elevation": e, "roof_plan": r, "freeform": f}
    flags = dict(R8) if args.r8 else {}
    if args.tight_crop:
        flags["tight_crop_"] = True
    if args.real_labelling:
        flags["real_labelling"] = True
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    ids = list(range(args.start, args.start + args.n))
    t0 = time.time()
    specs = [massing_spec(i, args.seed, house_for(i, args.seed, mw)) for i in ids]
    raw = run_freecad(specs)
    faces = {s["id"]: (s, raw[str(s["id"])]) for s in specs if isinstance(raw.get(str(s["id"])), list)}
    print(f"massing {len(faces)}/{len(ids)} in {time.time() - t0:.0f}s", flush=True)
    ok, modes = 0, {}
    with Pool(args.workers, initializer=_init, initargs=(args.out, args.seed, mw, flags, faces, args.revit, args.revit_plans)) as pool:
        for i, (iid, good, info) in enumerate(pool.imap_unordered(_job, ids, chunksize=2)):
            if good:
                ok += 1
                modes[info] = modes.get(info, 0) + 1
            if (i + 1) % 50 == 0 or i + 1 == len(ids):
                print(f"{i + 1}/{len(ids)} ok={ok} {time.time() - t0:.0f}s {modes}", flush=True)
    with open(os.path.join(args.out, "generation_manifest.json"), "w") as fo:
        json.dump({"generator": "scripts/generate_synthetic_fc.py", "n": args.n, "seed": args.seed,
                   "start": args.start, "ok": ok, "modes": modes, "mode_weights": mw, "flags": flags, "revit": args.revit,
                   "revit_plans": args.revit_plans, "shaped": args.shaped}, fo, indent=2)


if __name__ == "__main__":
    main()
