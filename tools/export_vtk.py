#!/usr/bin/env python3
"""Export one fluor-cell-3d field.grid frame + the wall mesh to compact VTK
files, for interactive 3D viewing in PyVista on a local machine (this repo's
own plot_field_3d_slices.py already does the HPC-side 2D slice plots; this
is for a real rotatable 3D view instead, which needs a proper structured
grid + a real desktop window, so it's meant to be viewed locally, not here).

Writes two small XML VTK files (a few MB, not the 100+MB raw field.grid):
  <out_prefix>_field.vtr  -- RectilinearGrid, point data: u,v,w,nrho,temp
  <out_prefix>_walls.vtp  -- PolyData, the housing wall mesh

Usage:
  python tools/export_vtk.py RUN_DIR --surf cell_fluor3d.surf --out-prefix OUT
      [--timestep N] [--frac F]
"""
import argparse
import os
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_field_3d_slices import load_frame, reshape_grid, read_surf, FIELD_POS


def build_field_grid(xs, ys, zs, fields):
    dx = np.median(np.diff(xs))
    dy = np.median(np.diff(ys))
    dz = np.median(np.diff(zs))
    x_edges = np.append(xs, xs[-1] + dx)
    y_edges = np.append(ys, ys[-1] + dy)
    z_edges = np.append(zs, zs[-1] + dz)
    grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
    for name in FIELD_POS:
        # PyVista is Fortran-order/point-major; reshape_grid's arrays are
        # (nx,ny,nz) C-order matching (x,y,z) axes directly, so a flat
        # Fortran-order raveL is what VTK's implicit cell ordering expects.
        grid.cell_data[name] = fields[name].ravel(order="F")
    return grid.cell_data_to_point_data()


def build_wall_mesh(points, tris):
    # points[0] is an unused dummy row (read_surf's point IDs are 1-based);
    # left in as an isolated, unreferenced vertex rather than reindexing
    # tris -- harmless in a PolyData, simpler than shifting every index.
    faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
    return pv.PolyData(points, faces)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--surf", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--timestep", type=int, default=None)
    p.add_argument("--frac", type=float, default=0.1)
    args = p.parse_args()

    frame = load_frame(args.run_dir, args.frac, args.timestep)
    xs, ys, zs, fields = reshape_grid(frame)
    field_grid = build_field_grid(xs, ys, zs, fields)

    points, tris, _tri_types = read_surf(args.surf)
    wall_mesh = build_wall_mesh(points, tris)

    field_path = f"{args.out_prefix}_field.vtr"
    wall_path = f"{args.out_prefix}_walls.vtp"
    field_grid.save(field_path)
    wall_mesh.save(wall_path)
    print(f"wrote {field_path} ({os.path.getsize(field_path)/1e6:.1f} MB), "
          f"{wall_path} ({os.path.getsize(wall_path)/1e6:.1f} MB), step {frame.timestep}")


if __name__ == "__main__":
    main()
