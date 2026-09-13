#!/usr/bin/env python3
"""Coarse-rebinned forward-velocity plot in Takahashi Fig. 1's style.

The paper never states DS2V's actual grid resolution, so this rebins our
fine adaptive-mesh field data onto a uniform coarse raster for display --
a plotting choice, not a re-simulation. Each source cell (at whatever its
native refinement level is) contributes to the coarse bin containing its
center, density-weighted, so bins average many small cells near the jet
core and few large cells far away -- same underlying physics, just a
coarser, less noisy picture.

Usage: python plot_fig1_coarse.py RUN_DIR --surf SURF --out OUT.png
       [--frac F] [--bin BIN_M]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from plot_fields_b5 import average_tail, surf_by_type, draw_geom, frame_label, FieldFormatError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--surf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frac", type=float, default=0.25)
    p.add_argument("--vmax", type=float, default=190.0)
    p.add_argument("--bin", type=float, default=1.0e-3, help="coarse bin size in meters (default 1mm)")
    a = p.parse_args()

    try:
        d = average_tail(os.path.join(a.run_dir, "field.grid"), a.frac)
    except FieldFormatError as exc:
        raise SystemExit(str(exc)) from exc
    surf = surf_by_type(os.path.join(a.run_dir, a.surf))

    good = d["nrho"] > 0
    xc = d["xc"][good]
    yc = d["yc"][good]
    u = d["u"][good]
    nrho = d["nrho"][good]

    box = d["box"]
    xlo, xhi = box[0]
    rtop = box[1][1]

    nx = max(1, int(round((xhi - xlo) / a.bin)))
    ny = max(1, int(round(rtop / a.bin)))
    xedges = np.linspace(xlo, xhi, nx + 1)
    yedges = np.linspace(-rtop, rtop, 2 * ny + 1)

    # mirror across the axis, weighting by density, before binning
    xc2 = np.concatenate([xc, xc])
    yc2 = np.concatenate([yc, -yc])
    u2 = np.concatenate([u, u])
    w2 = np.concatenate([nrho, nrho])

    num, _, _ = np.histogram2d(xc2, yc2, bins=[xedges, yedges], weights=w2 * u2)
    den, _, _ = np.histogram2d(xc2, yc2, bins=[xedges, yedges], weights=w2)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid_u = np.where(den > 0, num / den, np.nan)

    width = max(6.0, 6.5 * (xhi - xlo) / (2 * rtop) + 2.0)
    fig, ax = plt.subplots(figsize=(width, 6.5))
    pc = ax.pcolormesh(xedges, yedges, grid_u.T, cmap="jet", vmin=0.0, vmax=a.vmax, shading="flat")
    fig.colorbar(pc, ax=ax, label=r"Velocity component in the x direction (m s$^{-1}$)", pad=0.01)
    draw_geom(ax, surf)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-rtop, rtop)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("r (m)")
    ax.set_title("4 K He buffer gas, %s -- mean of %s -- %.2gmm bins" %
                  (os.path.basename(os.path.normpath(a.run_dir)), frame_label(d), a.bin * 1e3))
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
