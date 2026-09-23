#!/usr/bin/env python3
"""Takahashi single-stage cell in true 3D with the inlet tube entering
through the SIDE wall (from -z, perpendicular to the cell axis), near the
back wall -- mirroring cases/fluor-cell-3d, whose gas inlet cap sits on the
housing's bottom face near the back. Controlled test of whether that
inlet orientation alone produces the asymmetric exterior plume seen there.

Same dimensions as gen_takahashi_3d.py (12.7mm ID x 53mm cell, 5mm
aperture through a 0.5mm plate, 4mm ID x 20mm inlet tube, solid outer body
of radius RTOP_GEOM). A tube-into-cylinder intersection can't be
triangulated watertight by simple revolution, so this builds the solid
with gmsh's OpenCASCADE kernel (outer body minus gas region) and meshes
its boundary.

gmsh's pip wheel needs libGLU, absent on ORCD nodes; one was installed to
~/opt/glu-env via conda-forge -- run with
  LD_LIBRARY_PATH=~/opt/glu-env/lib python gen_takahashi_3d_side.py
"""
import math
import os
from collections import Counter

import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))

# mm (converted to m on output)
RCELL = 6.35
CELL_LENGTH = 53.0
RB = 2.0
TUBE_LENGTH = 20.0
RAP = 2.5
PLATE_THICK = 0.5
RTOP_GEOM = 55 * 0.13
XOUT = -20.3            # back of outer body, same as the reference
X_INLET = 8.0           # tube axis position along the cell
Z_CAP = -(RCELL + TUBE_LENGTH)
R_SLEEVE = RB + 1.0     # solid wall around the tube, outside the main body
SLEEVE_BELOW_CAP = 0.3  # same solid thickness behind the cap as the reference (XOUT - XLO)

XEXIT = CELL_LENGTH + PLATE_THICK

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("takahashi_side")
occ = gmsh.model.occ

body = occ.addCylinder(XOUT, 0, 0, XEXIT - XOUT, 0, 0, RTOP_GEOM)
sleeve = occ.addCylinder(X_INLET, 0, Z_CAP - SLEEVE_BELOW_CAP, 0, 0, -Z_CAP + SLEEVE_BELOW_CAP, R_SLEEVE)
outer, _ = occ.fuse([(3, body)], [(3, sleeve)])

cell = occ.addCylinder(0, 0, 0, CELL_LENGTH, 0, 0, RCELL)
aperture = occ.addCylinder(CELL_LENGTH - 0.1, 0, 0, PLATE_THICK + 1.1, 0, 0, RAP)
bore = occ.addCylinder(X_INLET, 0, Z_CAP, 0, 0, -Z_CAP, RB)
gas, _ = occ.fuse([(3, cell)], [(3, aperture), (3, bore)])

solid, _ = occ.cut(outer, gas)
occ.synchronize()
assert len(solid) == 1, solid
vol = solid[0][1]

cap_tags = []
for dim, tag in gmsh.model.getBoundary([(3, vol)], oriented=False):
    cx, cy, cz = occ.getCenterOfMass(dim, tag)
    if abs(cz - Z_CAP) < 1e-6 and abs(occ.getMass(dim, tag) - math.pi * RB * RB) < 1e-3:
        cap_tags.append(tag)
assert len(cap_tags) == 1, cap_tags

gmsh.option.setNumber("Mesh.MeshSizeMax", 1.2)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.15)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
gmsh.model.mesh.generate(2)

node_tags, coords, _ = gmsh.model.mesh.getNodes()
xyz = {int(t): (coords[3*i], coords[3*i+1], coords[3*i+2]) for i, t in enumerate(node_tags)}

triangles = []  # (type, n1, n2, n3) with gmsh node tags
for dim, tag in gmsh.model.getBoundary([(3, vol)], oriented=False):
    ttype = 2 if abs(tag) in cap_tags else 1
    etypes, _, enodes = gmsh.model.mesh.getElements(2, abs(tag))
    for et, en in zip(etypes, enodes):
        assert et == 2, "expected linear triangles"
        for k in range(0, len(en), 3):
            triangles.append((ttype, int(en[k]), int(en[k+1]), int(en[k+2])))
gmsh.finalize()

# gmsh's oriented-boundary signs came back inconsistent for a few surfaces
# of this boolean solid, so orientation is made consistent here instead:
# walk triangle-to-triangle across shared edges, flipping any neighbor that
# traverses the shared edge in the same direction (on a closed manifold,
# neighbors must traverse it in opposite directions).
edge_tris = {}
for i, (_, a, b, c) in enumerate(triangles):
    for u, v in ((a, b), (b, c), (c, a)):
        edge_tris.setdefault(tuple(sorted((u, v))), []).append(i)
seen = [False] * len(triangles)
for start in range(len(triangles)):
    if seen[start]:
        continue
    seen[start] = True
    stack = [start]
    while stack:
        i = stack.pop()
        _, a, b, c = triangles[i]
        for u, v in ((a, b), (b, c), (c, a)):
            for j in edge_tris[tuple(sorted((u, v)))]:
                if seen[j]:
                    continue
                t, p, q, r = triangles[j]
                if (u, v) in ((p, q), (q, r), (r, p)):
                    triangles[j] = (t, p, r, q)
                seen[j] = True
                stack.append(j)


def signed_volume(tris):
    s = 0.0
    for _, a, b, c in tris:
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = xyz[a], xyz[b], xyz[c]
        s += x1 * (y2 * z3 - z2 * y3) - y1 * (x2 * z3 - z2 * x3) + z1 * (x2 * y3 - y2 * x3)
    return s / 6.0


# SPARTA wants normals pointing OUT of the solid; for a consistently wound
# closed surface that's equivalent to positive signed volume.
if signed_volume(triangles) < 0:
    triangles = [(t, a, c, b) for t, a, b, c in triangles]

directed = Counter()
undirected = Counter()
for _, a, b, c in triangles:
    for u, v in ((a, b), (b, c), (c, a)):
        directed[(u, v)] += 1
        undirected[tuple(sorted((u, v)))] += 1
free = sum(1 for n in undirected.values() if n == 1)
nonmanifold = sum(1 for n in undirected.values() if n > 2)
inconsistent = sum(1 for n in directed.values() if n > 1)
print(f"free edges {free}, non-manifold {nonmanifold}, inconsistently wound {inconsistent}, "
      f"signed volume {signed_volume(triangles):.1f} mm^3")
assert free == 0 and nonmanifold == 0 and inconsistent == 0

used = sorted({n for t in triangles for n in t[1:]})
renum = {n: i + 1 for i, n in enumerate(used)}
out_path = os.path.join(HERE, "cell_takahashi_3d_side.surf")
with open(out_path, "w", newline="\n") as f:
    f.write("# GENERATED by gen_takahashi_3d_side.py -- Takahashi cell, true 3D, inlet tube\n")
    f.write("# entering from -z through the side wall at x=%.1fmm. type 1 = wall, type 2 = inlet cap.\n\n" % X_INLET)
    f.write("%d points\n%d triangles\n\nPoints\n\n" % (len(used), len(triangles)))
    for n in used:
        x, y, z = xyz[n]
        f.write("%d %.9g %.9g %.9g\n" % (renum[n], x * 1e-3, y * 1e-3, z * 1e-3))
    f.write("\nTriangles\n\n")
    for i, (t, a, b, c) in enumerate(triangles, start=1):
        f.write("%d %d %d %d %d\n" % (i, t, renum[a], renum[b], renum[c]))
print("wrote", out_path, "points", len(used), "triangles", len(triangles),
      "cap", sum(1 for t in triangles if t[0] == 2))
