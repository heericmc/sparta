#!/usr/bin/env python3
"""Seal the optical-window slot in the housing's top face of cell_fluor3d.surf.

gen_fluor3d.py excludes the optical window (#4) and its frame (#1) to avoid
a cut3d crash, on the assumption that the top face is continuous. It is
not: the housing has a 30 x 7mm stadium slot (x 42-72mm, y 15-23mm) from
the top face (z=38.1mm) down into the chamber (ceiling z~34.3mm), which the
window normally covers. Left open, ~70% of the injected neon escaped
through it (see docs/fluor-cell-3d-molecules-findings.md).

This removes the slot's side walls and caps the two openings they leave
(the top-face rim at z=38.1 and the chamber-ceiling rim), i.e. treats the
slot as solid -- the window-sealed cell minus a ~0.9 cm^3 pocket. Cap
triangles are fanned from each rim's centroid and wound opposite to the
neighbouring rim edge, so the result stays consistently oriented.

gen_fluor3d.py needs cadquery (dev machine only); run this on its output:
  python seal_top_slot.py [in.surf] [out.surf]
(defaults: cell_fluor3d.surf -> cell_fluor3d_sealed.surf)
"""
import sys
from collections import Counter, defaultdict

import numpy as np

SLOT_BOX_MM = ((41.8, 72.5), (15.1, 23.0), (33.5, 38.11))
CEILING_MAX_Z_MM = 34.6   # slot walls reach above this; chamber ceiling does not


def read_surf(path):
    lines = open(path).read().split("\n")
    header = [l for l in lines if l.startswith("#")]
    npts = int(next(l for l in lines if l.strip().endswith("points")).split()[0])
    i = lines.index("Points")
    pts = np.array([list(map(float, l.split()[1:4])) for l in lines[i + 2:i + 2 + npts]])
    j = lines.index("Triangles")
    tris = [tuple(map(int, l.split()[1:5])) for l in lines[j + 2:] if l.strip()]
    return header, pts, tris


def rim_loops(tris):
    """Ordered boundary loops of an open triangle set, as directed edges."""
    directed = [(u, v) for _, a, b, c in tris for u, v in ((a, b), (b, c), (c, a))]
    und = Counter(tuple(sorted(e)) for e in directed)
    free = [e for e in directed if und[tuple(sorted(e))] == 1]
    nxt = {u: v for u, v in free}
    assert len(nxt) == len(free), "rim is not a simple loop"
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, u = [], start
        while u not in seen:
            seen.add(u)
            loop.append((u, nxt[u]))
            u = nxt[u]
        loops.append(loop)
    return loops


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "cell_fluor3d.surf"
    dst = sys.argv[2] if len(sys.argv) > 2 else "cell_fluor3d_sealed.surf"
    header, pts, tris = read_surf(src)
    P = pts * 1e3

    (x0, x1), (y0, y1), (z0, z1) = SLOT_BOX_MM
    slot = []
    for k, (t, a, b, c) in enumerate(tris):
        V = P[[a - 1, b - 1, c - 1]]
        n = np.cross(V[1] - V[0], V[2] - V[0])
        if (t == 1 and abs(n[2]) < 0.2 * np.linalg.norm(n) and V[:, 2].max() > CEILING_MAX_Z_MM
                and np.all((V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 1] > y0) & (V[:, 1] < y1)
                           & (V[:, 2] > z0) & (V[:, 2] < z1))):
            slot.append(k)
    if not slot:
        sys.exit("no slot walls found -- already sealed?")
    slot_set = set(slot)
    wall_tris = [tris[k] for k in slot]
    kept = [t for k, t in enumerate(tris) if k not in slot_set]

    # The slot walls' own rim loops, traversed as the walls traverse them;
    # the cap over each rim must traverse every rim edge the same way the
    # walls did (it replaces them), so fan with the walls' direction.
    loops = rim_loops(wall_tris)
    assert len(loops) == 2, f"expected top + bottom rim, got {len(loops)} loops"
    new_pts = [p for p in pts]
    for loop in loops:
        ring = [u for u, _ in loop]
        centroid = pts[[u - 1 for u in ring]].mean(axis=0)
        new_pts.append(centroid)
        cid = len(new_pts)
        for u, v in loop:
            kept.append((1, u, v, cid))
        zmm = pts[[u - 1 for u in ring]][:, 2] * 1e3
        print(f"capped rim: {len(ring)} edges, z {zmm.min():.2f}-{zmm.max():.2f} mm")

    pts = np.array(new_pts)
    directed = Counter((u, v) for _, a, b, c in kept for u, v in ((a, b), (b, c), (c, a)))
    und = Counter()
    for (u, v), m in directed.items():
        und[tuple(sorted((u, v)))] += m
    free = sum(1 for m in und.values() if m == 1)
    nonman = sum(1 for m in und.values() if m > 2)
    incons = sum(1 for m in directed.values() if m > 1)
    vol = 0.0
    for _, a, b, c in kept:
        vol += np.dot(pts[a - 1], np.cross(pts[b - 1], pts[c - 1])) / 6.0
    print(f"removed {len(slot)} slot-wall triangles; free edges {free}, non-manifold {nonman}, "
          f"inconsistent {incons}, signed volume {vol*1e9:.1f} mm^3")
    assert free == 0 and nonman == 0 and incons == 0

    with open(dst, "w", newline="\n") as f:
        for h in header:
            f.write(h + "\n")
        f.write("# SEALED by seal_top_slot.py: optical-window slot in the top face filled.\n\n")
        f.write(f"{len(pts)} points\n{len(kept)} triangles\n\nPoints\n\n")
        for i, (x, y, z) in enumerate(pts, start=1):
            f.write(f"{i} {x:.9g} {y:.9g} {z:.9g}\n")
        f.write("\nTriangles\n\n")
        for i, (t, a, b, c) in enumerate(kept, start=1):
            f.write(f"{i} {t} {a} {b} {c}\n")
    print("wrote", dst)


if __name__ == "__main__":
    main()
