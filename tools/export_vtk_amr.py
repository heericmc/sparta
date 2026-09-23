#!/usr/bin/env python3
"""Export a tail-averaged 3D SPARTA grid dump on a NESTED (multi-level)
grid -- e.g. cases/takahashi-3d(-offset)'s `create_grid ... levels 2` --
to a uniform VTK ImageData (.vti) for isosurfacing in PyVista.

tools/export_vtk.py only handles a flat single-level grid (it reshapes by
unique lo-bounds). Here each cell is built as a voxel from its own
xlo..zhi bounds, cell data attached, and the result probe-resampled onto a
uniform grid over a cropped window -- a probe returns the value of the
cell containing each sample point, which is correct regardless of level.

Frames in the averaging window are matched by cell id (rows are sorted by
id per frame), since dump row order can change with load balancing.

Usage:
  python tools/export_vtk_amr.py RUN_DIR --surf SURF --out-prefix OUT
      [--frac 0.3] [--spacing 0.0006] [--bounds x0 x1 y0 y1 z0 z1]
Writes OUT_field.vti (point data u,v,w,nrho,temp) and OUT_walls.vtp.
"""
import argparse
import os
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_io import read_complete_frames
from plot_field_3d_slices import read_surf, FIELD_POS, NCOL, NRHO_CEILING
from export_vtk import build_wall_mesh


def tail_average(path, frac):
    size = os.path.getsize(path)
    tail_bytes = max(int(size * min(1.0, frac * 1.5 + 0.05)), 8 * 1024 * 1024)
    frames = read_complete_frames(path, tail_bytes=tail_bytes if tail_bytes < size else None).frames
    k = max(1, int(round(len(frames) * frac)))
    frames = frames[-k:]
    ref = None
    total = None
    for f in frames:
        if len(f.columns) != NCOL:
            raise ValueError(f"expected {NCOL} columns, got {f.columns}")
        d = f.data[np.argsort(f.data[:, 0], kind="stable")]
        if ref is None:
            ref = d[:, :10].copy()
            total = np.zeros((len(d), len(FIELD_POS)))
        elif not np.array_equal(d[:, 0], ref[:, 0]):
            raise ValueError(f"cell ids differ at step {f.timestep}")
        total += d[:, [FIELD_POS[n] for n in FIELD_POS]]
    return ref, total / len(frames), [f.timestep for f in frames]


def voxel_grid(geom, values, bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    lo, hi = geom[:, 4:7], geom[:, 7:10]
    keep = ((hi[:, 0] > x0) & (lo[:, 0] < x1) & (hi[:, 1] > y0) & (lo[:, 1] < y1) &
            (hi[:, 2] > z0) & (lo[:, 2] < z1) & (values[:, 0] < NRHO_CEILING))
    lo, hi, vals = lo[keep], hi[keep], values[keep]
    n = len(lo)
    corners = np.empty((n, 8, 3))
    for c, (i, j, k) in enumerate([(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]):
        corners[:, c, 0] = hi[:, 0] if i else lo[:, 0]
        corners[:, c, 1] = hi[:, 1] if j else lo[:, 1]
        corners[:, c, 2] = hi[:, 2] if k else lo[:, 2]
    cells = np.hstack([np.full((n, 1), 8), np.arange(8 * n).reshape(n, 8)]).ravel()
    ug = pv.UnstructuredGrid(cells, np.full(n, pv.CellType.VOXEL), corners.reshape(-1, 3))
    for i, name in enumerate(FIELD_POS):
        ug.cell_data[name] = vals[:, i].astype(np.float32)
    return ug


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--surf", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--frac", type=float, default=0.3)
    p.add_argument("--spacing", type=float, default=0.0006)
    p.add_argument("--bounds", type=float, nargs=6, default=[-0.021, 0.154, -0.04, 0.04, -0.04, 0.04])
    a = p.parse_args()

    geom, avg, steps = tail_average(os.path.join(a.run_dir, "field.grid"), a.frac)
    ug = voxel_grid(geom, avg, a.bounds)
    x0, x1, y0, y1, z0, z1 = a.bounds
    h = a.spacing
    dims = (int((x1 - x0) / h) + 1, int((y1 - y0) / h) + 1, int((z1 - z0) / h) + 1)
    img = pv.ImageData(dimensions=dims, spacing=(h, h, h), origin=(x0, y0, z0))
    sampled = img.sample(ug)
    for name in FIELD_POS:
        valid = sampled.point_data["vtkValidPointMask"].astype(bool)
        arr = np.asarray(sampled.point_data[name], dtype=np.float32)
        arr[~valid] = np.nan
        sampled.point_data[name] = arr

    points, tris, _ = read_surf(a.surf)
    field_path, wall_path = f"{a.out_prefix}_field.vti", f"{a.out_prefix}_walls.vtp"
    sampled.save(field_path)
    build_wall_mesh(points, tris).save(wall_path)
    print(f"averaged {len(steps)} frames (steps {steps[0]}-{steps[-1]}), {ug.n_cells} cells in window; "
          f"wrote {field_path} ({os.path.getsize(field_path)/1e6:.1f} MB, dims {dims}), {wall_path}")


if __name__ == "__main__":
    main()
