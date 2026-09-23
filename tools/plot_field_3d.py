#!/usr/bin/env python3
"""Azimuthally bin a true-3D SPARTA grid dump (Cartesian x,y,z cells, no
axisymmetric ring-averaging trick) into (x, r=sqrt(y^2+z^2)) bins, so an
on-axis 3D run can be visually compared against the 2D-axisymmetric result
in the same style as tools/plot_fig1_style.py.

Reads only a byte-sized tail of the (possibly still-growing) dump, so it is
safe to re-run periodically WHILE the solver is still running, without
loading the whole multi-GB file (see field_io.read_complete_frames'
tail_bytes) and without needing the run to finish first.

Usage: python tools/plot_field_3d.py RUN_DIR --out OUT.png [--csv-out OUT.csv]
       [--frac F] [--nr N]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_io import read_complete_frames, FieldFormatError

NCOL = 15  # id xc yc zc xlo ylo zlo xhi yhi zhi nrho u v w temp


def load_tail_frames(path, frac):
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise FieldFormatError(f"cannot stat {path}: {exc}") from exc
    tail_bytes = max(int(size * min(1.0, frac * 1.5 + 0.05)), 8 * 1024 * 1024)
    frame_set = read_complete_frames(path, tail_bytes=tail_bytes if tail_bytes < size else None)
    frames = frame_set.frames
    if not frames:
        raise FieldFormatError(f"{path}: no complete frames yet")
    if any(len(f.columns) != NCOL for f in frames):
        raise FieldFormatError(f"{path}: expected {NCOL} columns (id xc yc zc xlo ylo zlo "
                                 f"xhi yhi zhi nrho u v w temp), got {frames[0].columns}")
    k = max(1, int(round(len(frames) * frac)))
    return frames[-k:]


def bin_axisymmetric(frames, nr):
    """Volume- and density-weighted (x, r) bins from one or more 3D frames.

    Processes one frame at a time and accumulates running bincount sums,
    rather than concatenating every column across all frames first -- for
    dozens of multi-GB frames, holding N full extra copies alongside the
    already-parsed frame data is what actually exhausts memory, not the
    parsing itself. Geometry (x_edges/r_edges) is static across frames, so
    it only needs to be derived once, from the first frame.

    Pooling many cells per (x, r) bin suppresses genuine per-Cartesian-cell
    shot noise in the sparse far-field plume. Cells are weighted by nrho
    alone (matching plot_fields_b5.average_tail's exact convention), NOT by
    nrho*volume: a cell cut by the curved cell wall reports a correct
    density for its own (much smaller than bounding-box) true gas volume,
    but the dump has no separate cut-volume field, so weighting by the full
    bounding-box volume massively over-credits those cells (observed:
    reported densities up to ~2.7e24 m^-3 outside the wall, ~100x anything
    in the real flow).

    Bin edges are uniform, sized to the COARSEST cell present (with a
    safety margin), not derived from the finest cells in the domain: this
    mesh has a small centrally-refined region (400um cells) inside an
    otherwise coarse (1600um-x / 1040um-y,z) base grid. Deriving edges from
    the finest cells anywhere produces bins narrower than the coarse
    cells -- a coarse cell's single center point then lands in only one of
    several such narrow bins, leaving the other, physically-covered bins
    at that same radius/column empty. That "point-sample split across
    finer-than-actual-resolution bins" was the real cause of the periodic
    (period = coarse/fine width ratio) striping, not aliasing in general
    and not the nrho-ceiling issue above (confirmed: identical striping
    with and without that fix).
    """
    d0 = frames[0].data
    xlo0, xhi0 = d0[:, 4], d0[:, 7]
    ylo0, yhi0, zlo0, zhi0 = d0[:, 5], d0[:, 8], d0[:, 6], d0[:, 9]
    x_min, x_max = xlo0.min(), xhi0.max()
    max_dx = (xhi0 - xlo0).max()
    max_dyz = max((yhi0 - ylo0).max(), (zhi0 - zlo0).max())
    r0 = np.sqrt(d0[:, 2]**2 + d0[:, 3]**2)
    r_max = r0.max() if r0.size else 1.0

    SAFETY = 1.5  # margin above the coarsest cell width, to tolerate float rounding
    nx = max(1, int((x_max - x_min) / (max_dx * SAFETY)))
    nr_auto = max(1, int(r_max / (max_dyz * SAFETY)))
    nr = min(nr, nr_auto) if nr else nr_auto

    x_edges = np.linspace(x_min, x_max, nx + 1)
    r_edges = np.linspace(0.0, r_max, nr + 1)
    nbins = (len(x_edges) - 1) * (len(r_edges) - 1)

    count = np.zeros(nbins)
    nrho_sum = np.zeros(nbins)
    w_sum = np.zeros(nbins)
    u_num = np.zeros(nbins)
    v_num = np.zeros(nbins)
    t_num = np.zeros(nbins)

    # Cells cut by the curved cell wall can report a wildly inflated nrho
    # (observed up to ~2.7e24 m^-3, vs ~1.4e22 anywhere in the real flow) --
    # a per-cell reporting artifact from a tiny true cut volume, not a real
    # density. NRHO_CEILING excludes such cells from every average (not
    # just downweights them: even bare nrho, without the volume factor
    # above, would still swamp a weighted mean at that magnitude).
    NRHO_CEILING = 1e23

    for f in frames:
        d = f.data
        xc, yc, zc = d[:, 1], d[:, 2], d[:, 3]
        nrho, u, v, w, temp = d[:, 10], d[:, 11], d[:, 12], d[:, 13], d[:, 14]
        keep = nrho < NRHO_CEILING
        xc, r_vals = xc[keep], np.sqrt(yc[keep]**2 + zc[keep]**2)
        nrho, u, v, w, temp = nrho[keep], u[keep], v[keep], w[keep], temp[keep]
        r = r_vals

        ix = np.clip(np.searchsorted(x_edges, xc, side="right") - 1, 0, len(x_edges) - 2)
        ir = np.clip(np.searchsorted(r_edges, r, side="right") - 1, 0, len(r_edges) - 2)
        bin_id = ix * (len(r_edges) - 1) + ir

        count += np.bincount(bin_id, minlength=nbins)
        nrho_sum += np.bincount(bin_id, weights=nrho, minlength=nbins)
        w_sum += np.bincount(bin_id, weights=nrho, minlength=nbins)
        u_num += np.bincount(bin_id, weights=nrho * u, minlength=nbins)
        v_num += np.bincount(bin_id, weights=nrho * np.sqrt(v**2 + w**2), minlength=nbins)
        t_num += np.bincount(bin_id, weights=nrho * temp, minlength=nbins)

    with np.errstate(invalid="ignore", divide="ignore"):
        nrho_mean = np.where(count > 0, nrho_sum / count, 0.0)
        u_mean = np.where(w_sum > 0, u_num / w_sum, 0.0)
        v_mean = np.where(w_sum > 0, v_num / w_sum, 0.0)  # "radial" speed magnitude
        t_mean = np.where(w_sum > 0, t_num / w_sum, 0.0)

    nrho_mean = nrho_mean.reshape(len(x_edges) - 1, len(r_edges) - 1)
    u_mean = u_mean.reshape(len(x_edges) - 1, len(r_edges) - 1)
    v_mean = v_mean.reshape(len(x_edges) - 1, len(r_edges) - 1)
    t_mean = t_mean.reshape(len(x_edges) - 1, len(r_edges) - 1)
    return x_edges, r_edges, nrho_mean, u_mean, v_mean, t_mean


def write_csv(path, x_edges, r_edges, nrho, u, v, t, nframe, steps):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# nframe={nframe} steps={steps[0]}-{steps[-1]}\n")
        fh.write("x_lo,x_hi,r_lo,r_hi,nrho,u,r_speed,temp\n")
        for i in range(len(x_edges) - 1):
            for j in range(len(r_edges) - 1):
                if nrho[i, j] <= 0:
                    continue
                fh.write(f"{x_edges[i]},{x_edges[i+1]},{r_edges[j]},{r_edges[j+1]},"
                          f"{nrho[i,j]},{u[i,j]},{v[i,j]},{t[i,j]}\n")


def plot_png(path, x_edges, r_edges, u, vmin, vmax, run_label, steps, nframe):
    good = u != 0.0
    fig, ax = plt.subplots(figsize=(10, 6.5))
    X, R = np.meshgrid(x_edges, r_edges, indexing="ij")
    masked = np.ma.masked_where(~good, u)
    for sign in (1, -1):
        pcm = ax.pcolormesh(X, sign * R, masked, cmap="jet", vmin=vmin, vmax=vmax, shading="flat")
    fig.colorbar(pcm, ax=ax, label=r"Velocity component in the x direction (m s$^{-1}$)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("r (m)")
    ax.set_title(f"4 K He buffer gas (3D, azimuthally binned), {run_label} -- "
                 f"mean of {nframe} frames, steps {steps[0]}-{steps[-1]}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", default=None, help="PNG output path")
    p.add_argument("--csv-out", default=None, help="CSV output path")
    p.add_argument("--frac", type=float, default=0.25)
    p.add_argument("--nr", type=int, default=80)
    p.add_argument("--vmin", type=float, default=-5.0)
    p.add_argument("--vmax", type=float, default=190.0)
    a = p.parse_args()
    if a.out is None and a.csv_out is None:
        raise SystemExit("need at least one of --out / --csv-out")

    field_path = os.path.join(a.run_dir, "field.grid")
    try:
        frames = load_tail_frames(field_path, a.frac)
    except FieldFormatError as exc:
        raise SystemExit(str(exc)) from exc

    x_edges, r_edges, nrho, u, v, t = bin_axisymmetric(frames, a.nr)
    steps = [f.timestep for f in frames]

    if a.csv_out:
        write_csv(a.csv_out, x_edges, r_edges, nrho, u, v, t, len(frames), steps)
        print("wrote", a.csv_out)
    if a.out:
        plot_png(a.out, x_edges, r_edges, u, a.vmin, a.vmax,
                  os.path.basename(os.path.normpath(a.run_dir)), steps, len(frames))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
