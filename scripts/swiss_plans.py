#!/usr/bin/env python3
"""Real apartment layouts from Swiss Dwellings v3 (Archilyse AG, CC BY 4.0) for the Revit-style
floor-plan renderer (scripts/revit_plans.py, `--plan-source swiss`).

Swiss Dwellings: Standfest, M. et al., "Swiss Dwellings: a large dataset of apartment models
including aggregated geolocation-based simulation results covering viewshed, natural light,
traffic noise, centrality and geometric analysis", Zenodo 7788422 (v3.0.0), CC BY 4.0.
https://zenodo.org/records/7788422 -- attribution must travel with any data or model trained on it.

Why: the procedural layouts (`plan_layout`) were the last thing blind judges kept calling out
(wall stubs, colliding door swings, odd rooms; plan_v3_judging.md). These are real plans: real
walls with real thicknesses, doors, windows, kitchens, sinks, toilets, tubs, showers, stairs.

Preprocess once (streams the CSV out of the zip, nothing extracted):

    python3 scripts/swiss_plans.py --zip data/reference/swiss_dwellings/sd.zip \
        --out data/reference/swiss_dwellings/layouts.pkl.gz

Each kept apartment is rotated so its walls are axis-aligned (the renderer's doors and windows
are axis-aligned), converted to feet, and stored as WKB. Kept: residential, >= 4 rooms, 45-260 m2
of rooms, >= 92% of wall length within 2 deg of an axis, one footprint; exact duplicates (the
same unit repeated up a building) are dropped.

At render time `layout(rec, r)` returns the dict `plan_layout` returns (rooms / doors / windows /
openings / walls / walls_full / foot / t_ext / t_int) plus the balconies, railings and fitted
pieces (kitchen runs, sinks, toilets, tubs, showers, stairs) as furniture pieces.
"""
import argparse
import collections
import csv
import gzip
import io
import math
import pickle
import sys
import zipfile

import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

FT = 3.280839895
CSV_NAME = "swiss-dwellings-v3.0.0/geometries.csv"
ROOM_TYPES = {"ROOM", "BEDROOM", "LIVING_ROOM", "LIVING_DINING", "DINING", "KITCHEN", "KITCHEN_DINING",
              "BATHROOM", "CORRIDOR", "STOREROOM", "OFFICE", "STAIRCASE", "WINTERGARDEN", "NOT_DEFINED",
              "ROOM_DINING", "GUESTROOM", "LIBRARY", "LOBBY", "ENTRANCE", "WASH_AND_DRY", "CHILDRENS_ROOM"}
OUT_TYPES = {"BALCONY", "LOGGIA", "TERRACE", "GARDEN", "PATIO", "OUTDOOR_VOID"}
SOLID_TYPES = {"SHAFT", "VOID", "ELEVATOR"}


def _polys(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [q for q in getattr(g, "geoms", []) if isinstance(q, Polygon) and not q.is_empty]


def _dominant_angle(walls):
    """Wall direction mod 90 (deg), length-weighted circular mean on 4*theta."""
    sx = sy = 0.0
    for w in walls:
        c = list(w.exterior.coords)
        for a, b in zip(c[:-1], c[1:]):
            L = math.dist(a, b)
            if L < 1e-6:
                continue
            t = math.atan2(b[1] - a[1], b[0] - a[0])
            sx += L * math.cos(4 * t)
            sy += L * math.sin(4 * t)
    return math.degrees(math.atan2(sy, sx) / 4)


def _ortho_frac(walls, tol=2.0):
    tot = ok = 0.0
    for w in walls:
        c = list(w.exterior.coords)
        for a, b in zip(c[:-1], c[1:]):
            L = math.dist(a, b)
            if L < 1e-6:
                continue
            d = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 90
            tot += L
            if min(d, 90 - d) <= tol:
                ok += L
    return ok / max(tot, 1e-9)


def convert(rows):
    """One apartment's rows -> record in feet, walls axis-aligned; None if it doesn't qualify."""
    if any(r["unit_usage"] != "RESIDENTIAL" for r in rows):
        return None
    g = collections.defaultdict(list)
    for r in rows:
        try:
            geom = shapely.from_wkt(r["geometry"])
        except Exception:
            return None
        for q in _polys(geom):
            g[(r["entity_type"], r["entity_subtype"])].append(q)
    walls = g[("separator", "WALL")] + g[("separator", "COLUMN")]
    rooms = [(k[1], q) for k, qs in g.items() if k[0] == "area" and k[1] in ROOM_TYPES for q in qs]
    if len(rooms) < 4 or not walls:
        return None
    room_m2 = sum(q.area for _, q in rooms)
    if not 45 <= room_m2 <= 260:
        return None
    ang = _dominant_angle(walls)
    allg = unary_union(walls + [q for _, q in rooms])
    c = allg.centroid

    def T(q):
        q = affinity.rotate(q, -ang, origin=c)
        return affinity.scale(affinity.translate(q, -c.x, -c.y), FT, -FT, origin=(0, 0))   # y down, as the sheet
    walls = [T(q) for q in walls]
    if _ortho_frac(walls) < 0.92:
        return None
    rec = {"walls": [shapely.to_wkb(q) for q in walls],
           "rooms": [(t, shapely.to_wkb(T(q))) for t, q in rooms],
           "out": [(k[1], shapely.to_wkb(T(q))) for k, qs in g.items() if k[0] == "area" and k[1] in OUT_TYPES for q in qs],
           "solid": [(k[1], shapely.to_wkb(T(q))) for k, qs in g.items() if k[0] == "area" and k[1] in SOLID_TYPES
                     for q in qs],
           "railings": [shapely.to_wkb(T(q)) for q in g[("separator", "RAILING")]],
           "openings": [(k[1], shapely.to_wkb(T(q))) for k, qs in g.items() if k[0] == "opening" for q in qs],
           "features": [(k[1], shapely.to_wkb(T(q))) for k, qs in g.items() if k[0] == "feature" for q in qs],
           "room_m2": room_m2, "ids": {k: rows[0][k] for k in ("apartment_id", "site_id", "building_id", "floor_id")}}
    foot = unary_union([shapely.from_wkb(b) for b in rec["walls"]] + [shapely.from_wkb(b) for _, b in rec["rooms"]] +
                       [shapely.from_wkb(b) for _, b in rec["solid"]]).buffer(0.15, join_style=2).buffer(-0.15, join_style=2)
    ps = _polys(foot)
    if not ps or max(q.area for q in ps) < 0.95 * sum(q.area for q in ps):
        return None             # the unit is split (a maisonette's two levels, a detached store)
    return rec


def _sig(rec):
    """Same unit repeated up a building: equal room types + areas + opening count."""
    return (tuple(sorted((t, round(shapely.from_wkb(b).area)) for t, b in rec["rooms"])), len(rec["openings"]))


def preprocess(zip_path, out_path, limit=0):
    csv.field_size_limit(1 << 30)
    groups = collections.OrderedDict()
    with zipfile.ZipFile(zip_path) as z, z.open(CSV_NAME) as fb:
        for n, r in enumerate(csv.DictReader(io.TextIOWrapper(fb, encoding="utf-8"))):
            groups.setdefault(r["apartment_id"], []).append(r)
            if n % 500_000 == 0:
                print(f"{n:,} rows, {len(groups):,} apartments", flush=True)
    print(f"{len(groups):,} apartments", flush=True)
    out, seen, why = [], set(), collections.Counter()
    for k, (aid, rows) in enumerate(groups.items()):
        rec = convert(rows)
        if rec is None:
            why["filtered"] += 1
            continue
        s = _sig(rec)
        if s in seen:
            why["duplicate"] += 1
            continue
        seen.add(s)
        out.append(rec)
        if limit and len(out) >= limit:
            break
        if k % 5000 == 0:
            print(f"  {k:,} converted, {len(out):,} kept", flush=True)
    with gzip.open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"kept {len(out):,}; {dict(why)} -> {out_path}")


# ---------------------------------------------------------------------------------------------
# render time
# ---------------------------------------------------------------------------------------------
_CACHE = {}


def load(path):
    if path not in _CACHE:
        with gzip.open(path, "rb") as f:
            _CACHE[path] = pickle.load(f)
    return _CACHE[path]


def _axis_rect(q):
    """Opening polygon -> (o, pos, c, w, thickness) on an axis-aligned wall."""
    x0, y0, x1, y1 = q.bounds
    if x1 - x0 >= y1 - y0:
        return "h", (y0 + y1) / 2, (x0 + x1) / 2, x1 - x0, y1 - y0
    return "v", (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0, x1 - x0


def _thickness(q):
    m = q.minimum_rotated_rectangle
    c = list(m.exterior.coords)
    if len(c) < 4:
        return None
    a, b = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    return min(a, b) if q.area > 0.8 * m.area else None


def _names(rooms, entrance, feats, r, opn=()):
    """Swiss area types -> the US names the furniture / finish code keys on. `opn`: pairs of areas
    with no wall between them; a ROOM open to the kitchen or a corridor is the living room."""
    names = {}
    hub = {k for k, (t, q) in enumerate(rooms) if t in ("KITCHEN", "KITCHEN_DINING", "CORRIDOR", "LIVING_ROOM",
                                                         "LIVING_DINING", "DINING", "ROOM_DINING")}
    for i, j in opn:
        for a, b in ((i, j), (j, i)):
            if rooms[a][0] == "ROOM" and b in hub and a not in names:
                names[a] = r.choice(["LIVING ROOM", "LIVING", "GREAT ROOM"])
    bed = [k for k, (t, q) in enumerate(rooms) if t in ("ROOM", "BEDROOM", "GUESTROOM", "CHILDRENS_ROOM") and k not in names]
    bed.sort(key=lambda k: -rooms[k][1].area)
    baths = sorted([k for k, (t, q) in enumerate(rooms) if t == "BATHROOM"], key=lambda k: -rooms[k][1].area)
    nb = 1
    for i, k in enumerate(bed):
        if i == 0 and r.random() < 0.6:
            names[k] = "PRIMARY BEDROOM"
        elif rooms[k][1].area < 85 and r.random() < 0.5:
            names[k] = r.choice(["OFFICE", "STUDY", "DEN"])
        else:
            nb += 1
            names[k] = f"BEDROOM {nb}" if i or r.random() < 0.5 else "BEDROOM"
    for i, k in enumerate(baths):
        wet = any(t in ("BATHTUB", "SHOWER") and q.intersects(rooms[k][1]) for t, q in feats)
        names[k] = "BATH" if i == 0 else (f"BATH {i + 1}" if wet else r.choice(["POWDER", "1/2 BATH"]))
    m = {"LIVING_ROOM": ["LIVING ROOM", "LIVING", "GREAT ROOM"], "LIVING_DINING": ["LIVING / DINING", "GREAT ROOM"],
         "DINING": ["DINING"], "ROOM_DINING": ["DINING"], "KITCHEN": ["KITCHEN"], "KITCHEN_DINING": ["KITCHEN / DINING"],
         "STOREROOM": ["STORAGE", "CLOSET", "PANTRY"], "OFFICE": ["OFFICE"], "STAIRCASE": ["STAIR"],
         "WINTERGARDEN": ["SUNROOM"], "WASH_AND_DRY": ["LAUNDRY"], "LIBRARY": ["STUDY"], "NOT_DEFINED": ["STORAGE"],
         "LOBBY": ["ENTRY"], "ENTRANCE": ["ENTRY"]}
    for k, (t, q) in enumerate(rooms):
        if k in names:
            continue
        if t == "CORRIDOR":
            near = entrance is not None and q.distance(entrance) < 1.5
            names[k] = r.choice(["ENTRY", "FOYER"]) if near and r.random() < 0.7 else r.choice(["HALL", "HALLWAY", "CORRIDOR"])
        else:
            names[k] = r.choice(m.get(t, ["STORAGE"]))
    seen, out = collections.Counter(), []
    for k in range(len(rooms)):
        nm = names[k]
        seen[nm] += 1
        out.append(nm if seen[nm] == 1 or nm.startswith(("BEDROOM", "BATH")) else f"{nm} {seen[nm]}")
    return out


def layout(rec, r, zone):
    """Record -> plan_layout's dict (+ 'balconies', 'railings', 'fixtures', 'front_x')."""
    W_ = [shapely.from_wkb(b) for b in rec["walls"]]
    rooms_raw = [(t, shapely.from_wkb(b)) for t, b in rec["rooms"]]
    ops = [(t, shapely.from_wkb(b)) for t, b in rec["openings"]]
    solid = [shapely.from_wkb(b) for _, b in rec["solid"]]
    walls_full = unary_union(W_ + solid)
    foot_raw = unary_union(W_ + [q for _, q in rooms_raw] + solid).buffer(0.15, join_style=2).buffer(-0.15, join_style=2)
    foot = Polygon(max(_polys(foot_raw), key=lambda q: q.area).exterior)
    edge = foot.exterior
    th_ext = [t for w in W_ if (t := _thickness(w)) and w.distance(edge) < 0.2]
    th_int = [t for w in W_ if (t := _thickness(w)) and w.distance(edge) >= 0.2]
    t_ext = sorted(th_ext)[len(th_ext) // 2] if th_ext else 1.0
    t_int = sorted(th_int)[len(th_int) // 2] if th_int else 0.5
    entrance = next((q for t, q in ops if t == "ENTRANCE_DOOR"), None)
    feats = [(t, shapely.from_wkb(b)) for t, b in rec["features"]]
    rq = [max(_polys(q.buffer(0)), key=lambda p: p.area) for _, q in rooms_raw]
    # open plan: Swiss areas with no wall between them sit a hair apart (~0.05 ft); each area grows
    # into the wall-free gaps around it (never into a wall or another area), so one finish runs on
    # across the open side and no blank hairline splits the floor
    opn = [(i, j) for i in range(len(rq)) for j in range(i + 1, len(rq)) if rq[i].distance(rq[j]) < 0.15]
    names = _names(rooms_raw, entrance, feats, r, opn)
    free = foot.difference(walls_full)
    for k in range(len(rq)):
        others = unary_union([g for j, g in enumerate(rq) if j != k])
        add = rq[k].buffer(0.35, join_style=2).intersection(free).difference(others)
        rq[k] = max(_polys(unary_union([rq[k], add]).buffer(0)), key=lambda p: p.area)
    rooms = []
    for (t, _), q, nm in zip(rooms_raw, rq, names):
        rooms.append({"cell": q.buffer(t_int / 2, join_style=2), "inner": q, "kind": "room", "name": nm,
                      "zone": zone(nm), "swiss": t})
    for _ in range(2):      # a corridor open to a room takes that room's finish (largest neighbour)
        for i, j in opn:
            for a, b in ((i, j), (j, i)):
                if rooms[a]["swiss"] == "CORRIDOR" and rooms[b]["swiss"] != "CORRIDOR" or \
                        rooms[a]["swiss"] == rooms[b]["swiss"] == "CORRIDOR" and rooms[b]["inner"].area > rooms[a]["inner"].area:
                    nb_ = [rooms[c]["inner"].area for p_ in opn for c in p_ if a in p_ and c != a]
                    if rooms[b]["inner"].area >= max(nb_):
                        rooms[a]["zone"] = rooms[b]["zone"]
    balc = [shapely.from_wkb(b) for _, b in rec["out"]]
    doors, windows, openings, cut = [], [], [], []
    for t, q in ops:
        o, pos, c, w, th = _axis_rect(q)
        if w < 1.0:
            continue
        rect = box(c - w / 2, pos - th / 2 - 0.1, c + w / 2, pos + th / 2 + 0.1) if o == "h" else \
            box(pos - th / 2 - 0.1, c - w / 2, pos + th / 2 + 0.1, c + w / 2)
        cut.append(rect)
        if t == "WINDOW":
            windows.append((o, pos, c, w))
            continue
        d_ = th / 2 + 0.35          # the rooms either side of the door, probed just past the wall faces
        sides = [Point(c, pos - d_), Point(c, pos + d_)] if o == "h" else [Point(pos - d_, c), Point(pos + d_, c)]
        near = []
        for pt in sides:
            dk = min((rm["inner"].distance(pt), k) for k, rm in enumerate(rooms))
            if dk[0] < 0.3 and dk[1] not in near:
                near.append(dk[1])
        to_out = any(b.distance(rect) < 0.6 for b in balc)
        if to_out:
            dtype = "slider" if w > 5 and r.random() < 0.7 else "swing"
        elif w > 4.3:
            dtype = r.choice(["double", "double", "open", "pocket"])
        else:
            dtype = r.choices(["swing", "pocket", "barn"], weights=[86, 9, 5])[0]
        doors.append((o, pos, c, w, dtype))
        if len(near) == 2:
            openings.append((near[0], near[1], rect))
    walls = walls_full.difference(unary_union(cut)) if cut else walls_full
    # fitted pieces: drawn with the plan's furniture glyph machinery, as (polygon, kind) pieces
    fix = []
    for t, q in feats:
        fix.extend(_fixture(t, q, W_, r))
    ef = entrance.centroid.x if entrance is not None else foot.centroid.x
    return {"rooms": rooms, "doors": doors, "windows": windows, "openings": openings, "walls": walls,
            "walls_full": walls_full, "foot": foot, "t_ext": t_ext, "t_int": t_int, "balconies": balc,
            "railings": [shapely.from_wkb(b) for b in rec["railings"]], "fixtures": fix, "front_x": ef}


def _frame(q, walls):
    """Fixture rectangle -> (centre, u long-axis unit, v short-axis unit pointing away from the
    nearest wall, half-lengths)."""
    m = q.minimum_rotated_rectangle
    c = list(m.exterior.coords)[:4]
    e1, e2 = (c[1][0] - c[0][0], c[1][1] - c[0][1]), (c[2][0] - c[1][0], c[2][1] - c[1][1])
    l1, l2 = math.hypot(*e1), math.hypot(*e2)
    if l1 < l2:
        e1, e2, l1, l2 = e2, e1, l2, l1
    u = (e1[0] / max(l1, 1e-9), e1[1] / max(l1, 1e-9))
    v = (e2[0] / max(l2, 1e-9), e2[1] / max(l2, 1e-9))
    ctr = m.centroid
    wu = unary_union(walls) if walls else None
    if wu is not None:      # v points away from the wall the piece stands against
        pa = Point(ctr.x + v[0] * l2 / 2, ctr.y + v[1] * l2 / 2)
        pb = Point(ctr.x - v[0] * l2 / 2, ctr.y - v[1] * l2 / 2)
        if pa.distance(wu) < pb.distance(wu):
            v = (-v[0], -v[1])
    return (ctr.x, ctr.y), u, v, l1 / 2, l2 / 2


def _fixture(t, q, walls, r):
    near = [w for w in walls if w.distance(q) < 1.5]
    (cx, cy), u, v, hu, hv = _frame(q, near)

    def P(a, b):            # local (along u, out from the wall along v; wall at b = 0) -> sheet
        return (cx + u[0] * a + v[0] * (b - hv), cy + u[1] * a + v[1] * (b - hv))

    def R(a0, b0, a1, b1, kind="item", rad=0.0):
        g = Polygon([P(a0, b0), P(a1, b0), P(a1, b1), P(a0, b1)])
        if rad > 0 and min(abs(a1 - a0), abs(b1 - b0)) > 2.2 * rad:
            g = g.buffer(-rad).buffer(rad, 8)
        return (g, kind)

    def E(a, b, ra, rb, kind="line"):
        g = affinity.scale(Point(0, 0).buffer(1.0, 24), ra, rb)
        ang = math.degrees(math.atan2(u[1], u[0]))
        g = affinity.rotate(g, ang, origin=(0, 0))
        x, y = P(a, b)
        return (affinity.translate(g, x, y), kind)
    out = []
    D = 2 * hv
    if t == "TOILET":
        out += [R(-0.85, 0, 0.85, 0.75), E(0, 1.55, 0.62, 0.85, "item"), E(0, 1.6, 0.45, 0.62)]
    elif t == "SINK":
        out += [R(-hu, 0, hu, D, "fit")]
        n = 2 if hu > 1.9 else 1
        for k in range(n):
            a = -hu + 2 * hu * (k + 0.5) / n
            out += [E(a, D * 0.52, min(0.7, hu * 0.7 / n), D * 0.3), E(a, D * 0.52, min(0.5, hu * 0.5 / n), D * 0.2),
                    E(a, D * 0.52, 0.07, 0.07)]
    elif t == "BATHTUB":
        out += [R(-hu, 0, hu, D, "fit"), R(-hu + 0.3, 0.3, hu - 0.3, D - 0.3, "line", rad=min(0.6, D * 0.25)),
                E(hu - 0.6, D / 2, 0.1, 0.1)]
    elif t == "SHOWER":
        out += [R(-hu, 0, hu, D, "fit"), (LineString([P(-hu, 0), P(hu, D)]).buffer(0.02), "line"), E(0, D / 2, 0.12, 0.12)]
    elif t == "KITCHEN":
        out += [R(-hu, 0, hu, D, "fit")]
        if hu > 3:          # sink, range, fridge along the run
            s = r.uniform(-hu * 0.5, hu * 0.1)
            out += [R(s - 1.2, 0.35, s + 1.2, D - 0.35, "line", rad=0.15),
                    (LineString([P(s, 0.35), P(s, D - 0.35)]).buffer(0.02), "line")]
            fr = r.random() < 0.6                                   # fridge at the end of the run
            hi = hu - (3.2 if fr else 0.1) - 2.0
            slots = [g for g in (s + 1.4, s - 3.4, -hu + 0.1, hi) if -hu + 0.1 <= g <= hi]
            g0 = slots[0] if slots else None
            for da, db in ((0.55, 0.6), (1.45, 0.6), (0.55, 1.5), (1.45, 1.5)) if g0 is not None else ():
                out += [E(g0 + da, min(db, D - 0.4), 0.33, 0.33), E(g0 + da, min(db, D - 0.4), 0.18, 0.18)]
            if fr:
                out += [R(hu - 3.0, 0, hu, D + 0.3, "fit"), R(hu - 3.0, D + 0.05, hu, D + 0.3, "line")]
        if r.random() < 0.5:                                    # uppers, projected
            out.append((LineString([P(-hu, D * 0.55), P(hu, D * 0.55)]).buffer(0.02), "line"))
    elif t == "STAIRS":
        out += [(q, "fit")]
        x0, y0, x1, y1 = q.bounds
        n = int(max(x1 - x0, y1 - y0) / 0.9)
        horiz = (x1 - x0) >= (y1 - y0)
        for k in range(1, n):
            f = k / n
            ln = LineString([(x0 + f * (x1 - x0), y0), (x0 + f * (x1 - x0), y1)]) if horiz else \
                LineString([(x0, y0 + f * (y1 - y0)), (x1, y0 + f * (y1 - y0))])
            out.append((ln.intersection(q).buffer(0.02), "line"))
    elif t == "ELEVATOR":
        x0, y0, x1, y1 = q.bounds
        out += [(q, "fit"), (LineString([(x0, y0), (x1, y1)]).buffer(0.02), "line"),
                (LineString([(x0, y1), (x1, y0)]).buffer(0.02), "line")]
    else:                   # built-in cupboards, washing machines: fixed to the wall too
        out += [(q, "fit")]
    return [p for p in out if p is not None and not p[0].is_empty]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="data/reference/swiss_dwellings/sd.zip")
    ap.add_argument("--out", default="data/reference/swiss_dwellings/layouts.pkl.gz")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    preprocess(a.zip, a.out, a.limit)
