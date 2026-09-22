"""
Build the final SPARTA 3D triangulated surf file from the CAD geometry.

Pipeline:
  1. Re-run the screw-hole plugging (same logic as defeature.py).
  2. ALSO seal the outer 0.5mm tip of the 1.588mm buffer-gas-inlet hole
     (in solid #6) with a plug, using the same proven plug/fuse machinery.
     This converts it from an open through-hole into a capped blind stub,
     matching the emit/surf-on-sealed-wall convention used by both existing
     cases in the repo (cases/b5-lean, cases/takahashi-3d).
  3. Triangulate all 7 (now-plugged) solids directly -- no CAD boolean
     merge needed: a solid with through-holes is already a valid closed
     manifold (verified watertight via check_watertight.py), so the union
     of all 7 parts' own closed meshes is exactly the surface SPARTA needs.
  4. Filter degenerate (near-zero-area) triangles left over from tessellating
     the tiny conical drill-tip voids at blind screw holes.
  5. Classify each triangle's type by the diameter of the CAD face it came
     from:
       type 1 = generic wall (everything else, 4K diffuse-reflecting)
       type 2 = buffer gas inlet cap (the new sealing plug's flat face)
       type 3 = molecule inlet channel wall (2.54mm, open to exterior)
       type 4 = molecule outlet channel wall (3.175mm, open to exterior)
  6. Convert mm -> m (SPARTA units si) and write Points/Triangles sections
     in the exact format used by cases/takahashi-3d/cell_takahashi_3d.surf.

Requires cadquery (OCCT bindings) -- a dev-machine dependency, not needed
on the HPC side. Run this to REGENERATE cell_fluor3d.surf from the source
STEP file; SPARTA itself only ever reads the plain-text .surf output.
Run from this directory: python gen_fluor3d.py
"""
import os
import sys
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Solid
from OCP.BRepTools import BRepTools
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_PATH = os.path.join(HERE, "fluorescence_cooling_cell_installed.step")
OUT_SURF = os.path.join(HERE, "cell_fluor3d.surf")

SCREW_DIAMETERS_MM = [1.7780, 2.2606, 2.9464, 3.2639, 5.0800]
DIAM_TOL_MM = 0.02
FUSE_TOL = 1e-3
MM_TO_M = 1e-3

# Optical window (#4) and its retaining frame (#1): confirmed by direct CAD
# inspection that the housing's top face (z=38.1) is ONE continuous planar
# face spanning the full housing footprint, with no pocket under the
# window -- so these two parts have no connection to the gas chamber at
# all (the chamber only bores along X and Y, never reaching z=38.1). They
# are a purely surface-mounted optical/mechanical detail, irrelevant to
# the DSMC gas domain. Their mounting geometry (window sitting ~0.13mm
# above the housing, its top overlapping ~0.38mm into the frame) packs
# multiple thin, closely-stacked surfaces into a sub-mm vertical span --
# smaller than any reasonable grid cell -- which is what was crashing
# SPARTA's cut3d cell-cutter (CH: Edge not part of 2 vertices) even after
# the mesh passed the global watertight check. Excluding them removes that
# crash with zero effect on the actual gas-flow boundary.
EXCLUDED_SOLID_IDXS = {1, 4}

# Buffer gas inlet cap (solid #6, 1.588mm hole, true opening at (34.290,25.400,0.000))
GAS_INLET_SOLID_IDX = 6
GAS_INLET_CENTER_MM = (34.290, 25.400, 0.000)
GAS_INLET_DIR = (0.0, 0.0, 1.0)  # capping inward from the true exterior opening
GAS_INLET_RADIUS_MM = 1.5875 / 2.0
GAS_INLET_CAP_DEPTH_MM = 0.5

# Type-tag diameters (radius*2, mm) matched on the FINAL (post-plug) geometry
MOLECULE_INLET_DIAM_MM = 2.54    # type 3
MOLECULE_OUTLET_DIAM_MM = 3.175  # type 4
CAP_DIAM_MM = 1.5875             # type 2 (only the new plug's flat cap face)
FEATURE_TOL_MM = 0.02

LINEAR_DEFLECTION_MM = 0.05
ANGULAR_DEFLECTION = 0.4
DEGENERATE_AREA_MM2 = 1e-6
VERTEX_ROUND_DP_M = 6  # dedup tolerance once coordinates are in meters (~1 micron)

def closest_match(d, candidates):
    for c in candidates:
        if abs(d - c) <= DIAM_TOL_MM:
            return c
    return None

def get_screw_faces(shape):
    out = []
    e = TopExp_Explorer(shape, TopAbs_FACE)
    while e.More():
        face = TopoDS.Face_s(e.Current())
        surf = BRepAdaptor_Surface(face, True)
        if surf.GetType() == GeomAbs_Cylinder:
            cyl = surf.Cylinder()
            radius = cyl.Radius()
            diameter = radius * 2.0
            if closest_match(diameter, SCREW_DIAMETERS_MM) is not None:
                umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
                axis = cyl.Axis()
                loc, d = axis.Location(), axis.Direction()
                start = gp_Pnt(loc.X() + d.X()*vmin, loc.Y() + d.Y()*vmin, loc.Z() + d.Z()*vmin)
                end = gp_Pnt(loc.X() + d.X()*vmax, loc.Y() + d.Y()*vmax, loc.Z() + d.Z()*vmax)
                out.append((radius, start, end))
        e.Next()
    return out

def axis_key(start, end):
    d = (end.X()-start.X(), end.Y()-start.Y(), end.Z()-start.Z())
    absd = [abs(v) for v in d]
    idx = absd.index(max(absd))
    sign = 1 if d[idx] > 0 else -1
    p = (start.X(), start.Y(), start.Z())
    perp = tuple(round(p[i], 2) for i in range(3) if i != idx)
    return (idx, sign, perp)

def group_screw_holes(faces):
    """Build one plug per physical hole, sized exactly to each face group's
    own cylindrical extent (no attempt to reach any adjacent drill-tip
    cone). This intentionally leaves a small enclosed void at each blind
    hole's conical tip -- see keep_largest_shell() below for how those are
    removed afterward. (Two earlier approaches tried to eliminate the voids
    at the boolean stage instead -- extending each plug to the cone's exact
    apex, or by a flat margin -- but both introduced new, worse artifacts:
    a flat margin risked reaching into unrelated void space like the main
    chamber bore, and even the precise apex-based extension left duplicate/
    non-manifold geometry at the hole's OPENING end as a non-local side
    effect of the more complex simultaneous 32-hole boolean fuse. Stripping
    the resulting shells after a plain fuse is far more robust.)"""
    groups = {}
    for radius, start, end in faces:
        key = axis_key(start, end)
        groups.setdefault(key, []).append((radius, start, end))
    plugs = []
    for (idx, sign, perp), segs in groups.items():
        max_radius = max(r for r, _, _ in segs)
        origin = segs[0][1]
        axis_dir = [0.0, 0.0, 0.0]
        axis_dir[idx] = float(sign)
        d = gp_Dir(*axis_dir)
        def project(pnt):
            return (pnt.X()-origin.X())*d.X() + (pnt.Y()-origin.Y())*d.Y() + (pnt.Z()-origin.Z())*d.Z()
        ts = []
        for _, s, e in segs:
            ts.append(project(s)); ts.append(project(e))
        tmin, tmax = min(ts), max(ts)
        start_pnt = gp_Pnt(origin.X()+d.X()*tmin, origin.Y()+d.Y()*tmin, origin.Z()+d.Z()*tmin)
        plugs.append((max_radius, start_pnt, d, tmax - tmin))
    return plugs

def keep_largest_shell(shape):
    """Discard every shell but the largest (by face count), rebuilt as a
    fresh solid. A plain (unextended) plug fuse leaves each blind screw
    hole's drill-tip cone as its own small, fully enclosed 2-face void
    shell alongside the one real outer shell -- these are individually
    valid B-Rep (BRepCheck-clean) and invisible to a ray-casting classifier,
    but SPARTA's own read_surf watertight check rejects a surf file over
    them regardless ("duplicate edges", one per such void). Since they are
    genuinely enclosed and touch nothing else, just dropping them from the
    solid removes the corresponding tiny closed patch of triangles from the
    final mesh with no effect on the real boundary."""
    shells = []
    e = TopExp_Explorer(shape, TopAbs_SHELL)
    while e.More():
        shells.append(TopoDS.Shell_s(e.Current()))
        e.Next()
    if len(shells) <= 1:
        return shape

    def face_count(shell):
        n = 0
        fe = TopExp_Explorer(shell, TopAbs_FACE)
        while fe.More():
            n += 1
            fe.Next()
        return n

    outer = max(shells, key=face_count)
    new_solid = TopoDS_Solid()
    bb = BRep_Builder()
    bb.MakeSolid(new_solid)
    bb.Add(new_solid, outer)
    return new_solid

def keep_shells_min_faces(shape, min_faces):
    """Like keep_largest_shell, but keeps every shell with at least
    min_faces faces (not just the single largest) -- used after the final
    7-part fuse, where more than one surviving shell can be legitimate
    (e.g. the optical window and its retaining frame, which don't share
    coincident geometry with the main body and so correctly remain their
    own separate shells rather than merging into it)."""
    shells = []
    e = TopExp_Explorer(shape, TopAbs_SHELL)
    while e.More():
        shells.append(TopoDS.Shell_s(e.Current()))
        e.Next()

    def face_count(shell):
        n = 0
        fe = TopExp_Explorer(shell, TopAbs_FACE)
        while fe.More():
            n += 1
            fe.Next()
        return n

    kept = [s for s in shells if face_count(s) >= min_faces]
    dropped = len(shells) - len(kept)
    if dropped:
        print(f"  (dropped {dropped} shell(s) with < {min_faces} faces -- residual fuse artifacts)")
    comp = TopoDS_Compound()
    bb = BRep_Builder()
    bb.MakeCompound(comp)
    for s in kept:
        bb.Add(comp, s)
    return comp

def make_plug(radius, start_pnt, direction, height):
    ax2 = gp_Ax2(start_pnt, direction)
    maker = BRepPrimAPI_MakeCylinder(ax2, radius, height)
    maker.Build()
    return maker.Shape()

def make_compound(shapes):
    comp = TopoDS_Compound()
    bb = BRep_Builder()
    bb.MakeCompound(comp)
    for s in shapes:
        bb.Add(comp, s)
    return comp

def fuse_shapes(base, plugs, do_clean=True):
    if not plugs:
        return base
    tool = make_compound(plugs)
    fuse = BRepAlgoAPI_Fuse(base, tool)
    fuse.SetFuzzyValue(FUSE_TOL)
    fuse.Build()
    if not fuse.IsDone():
        raise RuntimeError("Fuse failed")
    result = fuse.Shape()
    if do_clean:
        result = cq.Shape(result).clean().wrapped
    return result

def triangulate_solid(shape):
    BRepMesh_IncrementalMesh(shape, LINEAR_DEFLECTION_MM, False, ANGULAR_DEFLECTION, True)
    tris = []  # (p1, p2, p3, face)
    e = TopExp_Explorer(shape, TopAbs_FACE)
    while e.More():
        face = TopoDS.Face_s(e.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            e.Next()
            continue
        trsf = loc.Transformation()
        pts = []
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            pts.append((p.X(), p.Y(), p.Z()))
        reversed_face = face.Orientation() != TopAbs_FORWARD
        for i in range(1, tri.NbTriangles() + 1):
            n1, n2, n3 = tri.Triangle(i).Get()
            p1, p2, p3 = pts[n1-1], pts[n2-1], pts[n3-1]
            if reversed_face:
                p2, p3 = p3, p2
            tris.append((p1, p2, p3, face))
        e.Next()
    return tris

from OCP.GeomAbs import GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

CAP_EXPECT_CENTER_MM = (
    GAS_INLET_CENTER_MM[0] + GAS_INLET_DIR[0] * GAS_INLET_CAP_DEPTH_MM,
    GAS_INLET_CENTER_MM[1] + GAS_INLET_DIR[1] * GAS_INLET_CAP_DEPTH_MM,
    GAS_INLET_CENTER_MM[2] + GAS_INLET_DIR[2] * GAS_INLET_CAP_DEPTH_MM,
)
CAP_EXPECT_AREA_MM2 = 3.14159265 * GAS_INLET_RADIUS_MM ** 2

def classify_face(face):
    """Return 2 (gas inlet cap), 3 (molecule inlet), 4 (molecule outlet), or
    1 (generic wall). The cap is identified as a PLANAR, ~circular, small
    face at the plug's known end location -- NOT by cylinder diameter, since
    the cap's cylindrical side wall shares the channel's own 1.588mm
    diameter and would otherwise wrongly tag the whole channel as type 2."""
    surf = BRepAdaptor_Surface(face, True)
    if surf.GetType() == GeomAbs_Cylinder:
        diameter = surf.Cylinder().Radius() * 2.0
        if abs(diameter - MOLECULE_INLET_DIAM_MM) <= FEATURE_TOL_MM:
            return 3
        if abs(diameter - MOLECULE_OUTLET_DIAM_MM) <= FEATURE_TOL_MM:
            return 4
        return 1
    if surf.GetType() == GeomAbs_Plane:
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = props.Mass()
        com = props.CentreOfMass()
        if abs(area - CAP_EXPECT_AREA_MM2) <= 0.5:
            # Tolerance must be well under GAS_INLET_CAP_DEPTH_MM (0.5mm) --
            # the plug has TWO flat end faces exactly that far apart (the
            # true-exterior face at GAS_INLET_CENTER_MM, which must stay
            # type 1, and the intended cap at CAP_EXPECT_CENTER_MM). A loose
            # tolerance here previously matched both, silently tagging half
            # the emit/surf source's area on the wrong (outward-facing) end.
            d = ((com.X()-CAP_EXPECT_CENTER_MM[0])**2 + (com.Y()-CAP_EXPECT_CENTER_MM[1])**2 +
                 (com.Z()-CAP_EXPECT_CENTER_MM[2])**2) ** 0.5
            # Sanity check: this face should be perpendicular to the channel
            # axis (a disk capping it), not e.g. some unrelated planar face
            # that happens to have a similar area by coincidence. Not used
            # for sign/direction -- Geom_Plane's canonical axis direction
            # isn't guaranteed to match face.Orientation()'s sense, so the
            # tight distance check above is what actually disambiguates the
            # two candidate end faces.
            pln = surf.Plane()
            n = pln.Axis().Direction()
            normal_dot = n.X()*GAS_INLET_DIR[0] + n.Y()*GAS_INLET_DIR[1] + n.Z()*GAS_INLET_DIR[2]
            if d <= 0.1 and abs(normal_dot) > 0.9:
                return 2
    return 1

def tri_area(p1, p2, p3):
    ux, uy, uz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
    vx, vy, vz = p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]
    cx, cy, cz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
    return 0.5 * (cx*cx+cy*cy+cz*cz) ** 0.5

def main():
    result = cq.importers.importStep(STEP_PATH)
    solids = result.solids().vals()
    print(f"Loaded {len(solids)} solid(s).")

    # sanity check: solid #6 should be the gas-inlet block (bbox x in [30.48, 38.1])
    check_shape = solids[GAS_INLET_SOLID_IDX].wrapped
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    bb = Bnd_Box()
    BRepBndLib.Add_s(check_shape, bb)
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    assert abs(xmin - 30.48) < 0.5 and abs(xmax - 38.1) < 0.5, \
        f"solid #{GAS_INLET_SOLID_IDX} bbox unexpected: x[{xmin},{xmax}] -- gas-inlet block assumption invalid"
    print(f"Confirmed solid #{GAS_INLET_SOLID_IDX} is the gas-inlet block (x[{xmin:.2f},{xmax:.2f}]).")

    plugged_solids = []
    for si, solid in enumerate(solids):
        if si in EXCLUDED_SOLID_IDXS:
            print(f"  solid #{si}: excluded (optical window/frame, no connection to gas domain)")
            continue
        shape = solid.wrapped
        screw_faces = get_screw_faces(shape)
        hole_plugs = group_screw_holes(screw_faces)
        plugs = [make_plug(r, s, d, h) for r, s, d, h in hole_plugs]

        # clean() (ShapeUpgrade_UnifySameDomain) is needed on solid #0 to
        # avoid shell fragmentation from its 32 simultaneous plugs, but on
        # solid #6 it corrupts an unrelated face where the 1.588mm channel
        # meets the 20.32mm bore (confirmed: clean() alone, with no fuse at
        # all, already breaks that face -- an OCCT limitation on this
        # shallow-angle intersection, not something plugging causes). So
        # skip clean() specifically for solid #6.
        do_clean = si != GAS_INLET_SOLID_IDX
        new_shape = fuse_shapes(shape, plugs, do_clean=do_clean)
        new_shape = keep_largest_shell(new_shape)
        print(f"  solid #{si}: {len(plugs)} hole(s) plugged")

        if si == GAS_INLET_SOLID_IDX:
            cap_start = gp_Pnt(*GAS_INLET_CENTER_MM)
            cap_dir = gp_Dir(*GAS_INLET_DIR)
            cap_plug = make_plug(GAS_INLET_RADIUS_MM, cap_start, cap_dir, GAS_INLET_CAP_DEPTH_MM)
            new_shape = fuse_shapes(new_shape, [cap_plug], do_clean=False)
            new_shape = keep_largest_shell(new_shape)
            print(f"    + gas inlet cap plugged")

        plugged_solids.append(new_shape)

    # Fuse into ONE true watertight body instead of triangulating each of
    # the 7 CAD parts independently and concatenating. The independent-
    # triangulation approach (used until now) is watertight by our own
    # per-solid free-edge check, since each part is internally consistent
    # -- but at every mating boundary between touching parts (housing to
    # each bolted-on plate/cover/inlet block) it leaves TWO separate,
    # oppositely-facing surface patches covering the same physical area: a
    # real double-wall duplicate, invisible to a check that only looks
    # within one solid at a time. That is almost certainly what SPARTA's
    # own read_surf watertight check has been rejecting all along -- the
    # "32 duplicate edges" count stayed IDENTICAL even after the screw-hole
    # void fix above changed solid #0's triangle count by 8x, which rules
    # out the screw holes as the real cause and points at the (until now
    # untouched) mating boundaries instead.
    #
    # Not every part actually touches: the optical window and its
    # retaining frame sit with a genuine small CAD gap/overlap versus the
    # main housing (not flush-mating), so they legitimately remain their
    # own separate shells after the fuse -- that's correct, not a defect,
    # since they don't actually share coincident geometry with anything.
    base = plugged_solids[0]
    tool = make_compound(plugged_solids[1:])
    fuse = BRepAlgoAPI_Fuse(base, tool)
    fuse.SetFuzzyValue(FUSE_TOL)
    fuse.Build()
    if not fuse.IsDone():
        raise RuntimeError("Final assembly fuse failed")
    merged = keep_shells_min_faces(fuse.Shape(), min_faces=3)
    print(f"Fused {len(plugged_solids)} part(s) into one assembly.")

    all_tris = triangulate_solid(merged)
    print(f"Total triangles before filtering: {len(all_tris)}")

    # Classify. Do NOT drop small-area triangles here -- some (e.g. at the
    # tiny conical drill-tip bottoms of blind screw holes) are topologically
    # load-bearing: removing them opens real gaps in the mesh rather than
    # closing harmless artifacts. Truly-degenerate (zero-area) triangles are
    # instead dropped later, after vertex dedup, by their repeated point
    # index -- that's the safe point to filter since it can't orphan edges.
    kept = []
    type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for p1, p2, p3, face in all_tris:
        ttype = classify_face(face)
        type_counts[ttype] += 1
        kept.append((ttype, p1, p2, p3))

    print(f"Kept triangles: {len(kept)}  by type: {type_counts}")

    # Dedup vertices (in meters) and build Points/Triangles
    point_index = {}
    points = []
    def get_idx(p_mm):
        p_m = (p_mm[0]*MM_TO_M, p_mm[1]*MM_TO_M, p_mm[2]*MM_TO_M)
        key = tuple(round(c, VERTEX_ROUND_DP_M) for c in p_m)
        if key not in point_index:
            points.append(p_m)
            point_index[key] = len(points)  # 1-based
        return point_index[key]

    triangles = []
    for ttype, p1, p2, p3 in kept:
        i1, i2, i3 = get_idx(p1), get_idx(p2), get_idx(p3)
        if i1 == i2 or i2 == i3 or i1 == i3:
            continue
        triangles.append((ttype, i1, i2, i3))

    print(f"Final: {len(points)} points, {len(triangles)} triangles")

    with open(OUT_SURF, "w", newline="\n") as f:
        f.write("# GENERATED by gen_fluor3d.py from fluorescence_cooling_cell_installed.step.\n")
        f.write("# Fluorescence buffer-gas cooling cell, true 3D, screw holes plugged, all\n")
        f.write("# touching parts fused into one watertight body.\n")
        f.write("# type 1 = wall, type 2 = buffer gas inlet cap (emit/surf mass-flow source),\n")
        f.write("# type 3 = molecule inlet channel (open, 2.54mm), type 4 = molecule outlet channel (open, 3.175mm).\n")
        f.write("# Units: meters (SPARTA si). Source CAD was in mm; converted here.\n")
        f.write("# NOTE: the optical window and its retaining frame are excluded -- confirmed\n")
        f.write("# by direct CAD inspection they have no connection to the gas chamber (housing\n")
        f.write("# top is solid underneath them), and their sub-mm mounting geometry crashed\n")
        f.write("# SPARTA's cut3d cell cutter. See EXCLUDED_SOLID_IDXS in gen_fluor3d.py.\n\n")
        f.write("%d points\n%d triangles\n\n" % (len(points), len(triangles)))
        f.write("Points\n\n")
        for idx, (x, y, z) in enumerate(points, start=1):
            f.write("%d %.9g %.9g %.9g\n" % (idx, x, y, z))
        f.write("\nTriangles\n\n")
        for idx, (ttype, p1, p2, p3) in enumerate(triangles, start=1):
            f.write("%d %d %d %d %d\n" % (idx, ttype, p1, p2, p3))

    print(f"\nWrote {OUT_SURF}")

if __name__ == "__main__":
    sys.exit(main())
