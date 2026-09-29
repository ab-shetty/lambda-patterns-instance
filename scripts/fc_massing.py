"""FreeCAD stage of the FreeCAD-geometry synth pilot (run under freecadcmd).

Reads a JSON list of houses (blocks in plan feet, z up), builds each as 3D
solids -- wall boxes with gable infill, roof slabs / hip solids, chimney --
fuses them, and writes every planar face of the fused solid (outer loop,
holes, normal) back out as JSON. Visibility, zoning and drawing happen in
scripts/generate_synthetic_fc.py.

  freecadcmd -c "import sys; sys.argv=['x','in.json','out.json']; exec(open('scripts/fc_massing.py').read())"
"""
import json
import sys

import FreeCAD as App  # noqa: F401  (initialises the Part module)
import Part
from FreeCAD import Vector as V

SLAB = 0.55          # roof slab thickness, ft (rakes / fascia read as trim)


def _xyz(ridge, a, c, z):
    return V(a, c, z) if ridge == "x" else V(c, a, z)


def prism(ridge, profile, a0, a1):
    """Extrude a closed (c, z) profile along the ridge axis from a0 to a1."""
    pts = [_xyz(ridge, a0, c, z) for c, z in profile]
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    d = _xyz(ridge, a1 - a0, 0, 0)
    return _outward(face.extrude(d))


def _outward(sol):
    """Profiles wound either way and the x/y swap for y-ridges can yield
    inside-out solids, which boolean fuse rejects."""
    if sol.ShapeType != "Solid":
        sol = Part.Solid(Part.Shell(sol.Faces))
    if sol.Volume < 0:
        sol.reverse()
    return sol


def solid_from_faces(loops):
    faces = [Part.Face(Part.makePolygon([V(*p) for p in lp] + [V(*lp[0])])) for lp in loops]
    return _outward(Part.Solid(Part.Shell(faces)))


def block_solids(b):
    x0, y0, w, d, H = b["x0"], b["y0"], b["w"], b["d"], b["H"]
    x1, y1 = x0 + w, y0 + d
    roof, ridge, pitch, ov = b["roof"], b["ridge"], b["pitch"], b["ov"]
    # a = along the ridge, c = across it (the slope direction)
    a0, a1, c0, c1 = (x0, x1, y0, y1) if ridge == "x" else (y0, y1, x0, x1)
    span = c1 - c0
    out = [Part.makeBox(w, d, H + (b.get("parapet", 0.0) if roof == "flat" else 0.0), V(x0, y0, 0))]
    if roof == "gable":
        rise = pitch * span / 2
        cm = (c0 + c1) / 2
        out.append(prism(ridge, [(c0, H - 0.01), (c1, H - 0.01), (cm, H + rise - 0.01)], a0, a1))
        ze = H - ov * pitch
        top = [(c0 - ov, ze), (cm, H + rise), (c1 + ov, ze)]
        prof = top + [(c1 + ov, ze - SLAB), (cm, H + rise - SLAB), (c0 - ov, ze - SLAB)]
        out.append(prism(ridge, prof, a0 - ov, a1 + ov))
    elif roof == "shed":
        rise = pitch * span
        out.append(prism(ridge, [(c0, H - 0.01), (c1, H - 0.01), (c1, H + rise - 0.01)], a0, a1))
        zl, zh = H - ov * pitch, H + rise + ov * pitch
        prof = [(c0 - ov, zl), (c1 + ov, zh), (c1 + ov, zh - SLAB), (c0 - ov, zl - SLAB)]
        out.append(prism(ridge, prof, a0 - ov, a1 + ov))
    elif roof == "hip":
        ze = H - ov * pitch
        X0, X1, Y0, Y1 = x0 - ov, x1 + ov, y0 - ov, y1 + ov
        run = (min(X1 - X0, Y1 - Y0)) / 2
        zr = ze + pitch * run
        if X1 - X0 >= Y1 - Y0:
            ym = (Y0 + Y1) / 2
            r0, r1 = (X0 + run, ym), (X1 - run, ym)
        else:
            xm = (X0 + X1) / 2
            r0, r1 = (xm, Y0 + run), (xm, Y1 - run)
        A, B, C, D = (X0, Y0, ze), (X1, Y0, ze), (X1, Y1, ze), (X0, Y1, ze)
        R0, R1 = (r0[0], r0[1], zr), (r1[0], r1[1], zr)
        if abs(R0[0] - R1[0]) + abs(R0[1] - R1[1]) < 1e-6:          # pyramid
            loops = [[A, B, R0], [B, C, R0], [C, D, R0], [D, A, R0], [A, D, C, B]]
        elif X1 - X0 >= Y1 - Y0:
            loops = [[A, B, R1, R0], [B, C, R1], [C, D, R0, R1], [D, A, R0], [A, D, C, B]]
        else:
            loops = [[A, B, R0], [B, C, R1, R0], [C, D, R1], [D, A, R0, R1], [A, D, C, B]]
        out.append(solid_from_faces(loops))
    return out


# ---- convex polygon blocks (--shaped: chamfered corners, 45-degree bays)
# For a convex footprint, a hip roof is the lower envelope of the edges' slope planes,
# so the roof is the footprint prism intersected with one half-space per sloped edge.
# Gable edges contribute no plane (their triangle rises flush with the wall); a shed
# keeps one plane.
def _halfspace_below(E0, e, n_in, tan_t, B=400.0):
    U = V(e[0], e[1], 0)
    U.normalize()
    Vv = V(n_in[0], n_in[1], tan_t)
    Vv.normalize()
    W = U.cross(Vv)
    if W.z > 0:                           # box must lie BELOW the plane: flip W (and U to stay right-handed)
        W, U = -W, -U
    O = E0 - U * B - Vv * B
    m = App.Matrix(U.x, Vv.x, W.x, O.x, U.y, Vv.y, W.y, O.y, U.z, Vv.z, W.z, O.z, 0, 0, 0, 1)
    box = Part.makeBox(2 * B, 2 * B, 2 * B)
    box.Placement = App.Placement(m)          # rigid placement keeps the faces planar (transformGeometry -> BSpline)
    return box


def _ccw(pts):
    a = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    return pts if a > 0 else pts[::-1]


def _prism(pts, z0, z1):
    wire = Part.makePolygon([V(x, y, z0) for x, y in pts] + [V(pts[0][0], pts[0][1], z0)])
    return _outward(Part.Face(wire).extrude(V(0, 0, z1 - z0)))


def _offset_pts(pts, d):
    """Offset a convex CCW polygon outward by d (edge-parallel)."""
    n = len(pts)
    lines = []
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        nx, ny = (y1 - y0) / L, -(x1 - x0) / L         # outward for CCW
        lines.append(((x0 + nx * d, y0 + ny * d), (x1 - x0, y1 - y0)))
    out = []
    for i in range(n):
        (p, d1), (q, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        t = ((q[0] - p[0]) * d2[1] - (q[1] - p[1]) * d2[0]) / den
        out.append((p[0] + d1[0] * t, p[1] + d1[1] * t))
    return out


def poly_block_solids(b):
    pts = _ccw([tuple(p) for p in b["poly"]])
    H, roof, pitch, ov = b["H"], b["roof"], b["pitch"], b["ov"]
    n = len(pts)
    top = H + 60.0
    if roof == "flat":
        return [_prism(pts, 0.0, H + b.get("parapet", 0.0))]
    out = [_prism(pts, 0.0, H)]
    sloped = b["sloped"]                       # indices of edges carrying a roof plane
    attic = _prism(pts, H - 0.01, top)
    slab_top = _prism(_offset_pts(pts, ov), H - ov * pitch - SLAB - 0.5, top)
    slab_bot = slab_top.copy()
    for i in sloped:
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        e = (x1 - x0, y1 - y0)
        L = (e[0] ** 2 + e[1] ** 2) ** 0.5
        n_in = (-e[1] / L, e[0] / L)           # inward for CCW
        E0 = V(x0, y0, H)
        attic = attic.common(_halfspace_below(E0, e, n_in, pitch))
        slab_top = slab_top.common(_halfspace_below(E0, e, n_in, pitch))
        slab_bot = slab_bot.common(_halfspace_below(E0 - V(0, 0, SLAB), e, n_in, pitch))
    # booleans return compounds; the fuse in build() needs plain solids
    out += [sd for sd in attic.Solids if sd.Volume > 1e-3]
    out += [sd for sd in slab_top.cut(slab_bot).Solids if sd.Volume > 1e-3]
    return out


def faces_of(shape):
    faces = []
    for f in shape.Faces:
        if not isinstance(f.Surface, Part.Plane):
            continue
        try:
            n = f.normalAt(*f.Surface.parameter(f.CenterOfMass))
        except Exception:
            continue
        # outward normal: a point just off the face along n must be outside the solid
        n.normalize()
        # probe from a point ON the face (L-shaped faces have their centroid off the face)
        pts, tris = f.tessellate(0.2)
        if not tris:
            continue
        big = max(tris, key=lambda t: ((pts[t[1]] - pts[t[0]]).cross(pts[t[2]] - pts[t[0]])).Length)
        on = (pts[big[0]] + pts[big[1]] + pts[big[2]]) * (1.0 / 3)
        probe = on + n * 0.02
        if shape.isInside(probe, 1e-4, False):
            n = -n
        loops = []
        for wi, w in enumerate([f.OuterWire] + [w for w in f.Wires if not w.isSame(f.OuterWire)]):
            lp = [[round(v.Point.x, 4), round(v.Point.y, 4), round(v.Point.z, 4)] for v in w.OrderedVertexes]
            if len(lp) >= 3:
                loops.append(lp)
        c = f.CenterOfMass
        faces.append({"loops": loops, "n": [round(n.x, 5), round(n.y, 5), round(n.z, 5)],
                      "c": [round(c.x, 4), round(c.y, 4), round(c.z, 4)], "area": round(f.Area, 3)})
    return faces


def build(house):
    solids = []
    for b in house["blocks"]:
        solids += poly_block_solids(b) if b.get("poly") else block_solids(b)
    ch = house.get("chimney")
    if ch:
        solids.append(Part.makeBox(ch["w"], ch["d"], ch["z1"] - ch["z0"], V(ch["x0"], ch["y0"], ch["z0"])))
    shape = solids[0].fuse(solids[1:]) if len(solids) > 1 else solids[0]
    shape = shape.removeSplitter()
    return faces_of(shape)


def main(src, dst):
    houses = json.load(open(src))
    out = {}
    for h in houses:
        try:
            out[str(h["id"])] = build(h)
        except Exception as exc:          # one bad house must not kill the batch
            out[str(h["id"])] = {"error": str(exc)}
    json.dump(out, open(dst, "w"))
    print("fc_massing: %d houses, %d failed" % (len(out), sum(isinstance(v, dict) for v in out.values())))


if len(sys.argv) >= 3:
    main(sys.argv[-2], sys.argv[-1])
