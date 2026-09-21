"""Sanity-check cell_fluor3d.surf: point/triangle counts match the header,
all triangle indices are in range, no degenerate triangles, and the mesh is
watertight (every real edge used by exactly 2 triangles -- free edges mean
a leak/gap; a few non-manifold edges are EXPECTED here where multiple
separate-but-touching CAD solids share a boundary line, e.g. the housing's
own 12 box edges where 2-3 bolted-on plates meet it -- that's normal for a
multi-part assembly, not a defect). Pure stdlib, no dependencies -- meant
to be run on the HPC side too, without needing cadquery/OCCT.

Run from this directory: python check_surf3d.py
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "cell_fluor3d.surf")

def main():
    with open(PATH) as f:
        lines = [l.rstrip("\n") for l in f]

    i = 0
    while not lines[i].strip().endswith("points"):
        i += 1
    n_points = int(lines[i].split()[0]); i += 1
    n_tris = int(lines[i].split()[0]); i += 1
    while lines[i].strip() != "Points":
        i += 1
    i += 1
    while lines[i].strip() == "":
        i += 1

    points = {}
    for _ in range(n_points):
        parts = lines[i].split()
        idx = int(parts[0])
        points[idx] = (float(parts[1]), float(parts[2]), float(parts[3]))
        i += 1
    while lines[i].strip() != "Triangles":
        i += 1
    i += 1
    while lines[i].strip() == "":
        i += 1

    tris = []
    for _ in range(n_tris):
        parts = lines[i].split()
        tris.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])))
        i += 1

    print(f"Header says: {n_points} points, {n_tris} triangles")
    print(f"Parsed: {len(points)} points, {len(tris)} triangles")
    assert len(points) == n_points and len(tris) == n_tris, "count mismatch!"

    bad_refs = [t for t in tris if not (1 <= t[1] <= n_points and 1 <= t[2] <= n_points and 1 <= t[3] <= n_points)]
    print(f"Triangles with out-of-range point refs: {len(bad_refs)}")

    type_counts = defaultdict(int)
    for t in tris:
        type_counts[t[0]] += 1
    print(f"Triangle counts by type: {dict(sorted(type_counts.items()))}  "
          f"(1=wall, 2=gas inlet cap, 3=molecule inlet, 4=molecule outlet)")

    xs = [p[0] for p in points.values()]; ys = [p[1] for p in points.values()]; zs = [p[2] for p in points.values()]
    print(f"Bounding box (m): x[{min(xs):.6f},{max(xs):.6f}] y[{min(ys):.6f},{max(ys):.6f}] z[{min(zs):.6f},{max(zs):.6f}]")

    n_degenerate = sum(1 for t in tris if t[1] == t[2] or t[2] == t[3] or t[1] == t[3])
    print(f"Degenerate (repeated-vertex) triangles: {n_degenerate}")

    edge_count = defaultdict(int)
    for t in tris:
        p1, p2, p3 = t[1], t[2], t[3]
        for a, b in ((p1,p2),(p2,p3),(p3,p1)):
            k = (a,b) if a < b else (b,a)
            edge_count[k] += 1
    free = sum(1 for c in edge_count.values() if c == 1)
    nonmanifold = sum(1 for c in edge_count.values() if c > 2)
    print(f"Free edges (leaks/gaps -- should be 0): {free}")
    print(f"Non-manifold edges (expected >0 at multi-part boundaries): {nonmanifold}")

    ok = len(bad_refs) == 0 and n_degenerate == 0 and free == 0
    print("\nOK -- watertight, ready for read_surf" if ok else "\nISSUES FOUND (see above) -- do not trust this surf file yet")

if __name__ == "__main__":
    sys.exit(main())
