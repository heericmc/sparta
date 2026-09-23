#!/usr/bin/env python3
"""Orthogonal-slice plots of a true-3D SPARTA grid dump (fluor-cell-3d and
any other flat, single-level Cartesian grid.grid dump -- NOT valid for a
nested-refinement grid like takahashi-3d/b5-lean, which needs level-aware
reshaping this script does not do).

Why slices instead of the azimuthal (x, r) binning tools/plot_field_3d.py
uses: that tool assumes near-axisymmetry, which does not hold here (the gas
inlet is off-axis by construction). A real 2D heatmap through a plane that
actually contains the ports shows the non-axisymmetric structure directly;
azimuthal binning would wash it out.

Reshapes the dump's flat cell list into a (nx, ny, nz) array from the cell
centers, then plots three orthogonal 2D slices (XY, XZ, YZ) of one field
component. Pass --surf to additionally overlay raw wall-mesh/cutting-plane
intersection lines -- for this case's fused multi-part CAD geometry that is
often visually dense (real panels/bosses near the slice, not a bug) rather
than a clean silhouette, since this script does not build closed polygons
from the raw triangle-plane crossings. Off by default for that reason.

Usage:
  python tools/plot_field_3d_slices.py RUN_DIR --out OUT.png
      [--field u|v|w|nrho|temp] [--surf cell_fluor3d.surf]
      [--x X] [--y Y] [--z Z] [--timestep N] [--frac F]

Only reads a byte-sized tail of the (possibly still-growing) dump by
default (see field_io.read_complete_frames), so it is safe to re-run while
the solver is still writing to field.grid.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_io import read_complete_frames, select_frame, FieldFormatError

# Velocity components are signed (polarity) -> diverging, neutral at 0.
# Density/temperature are magnitude-only -> sequential, single hue.
DIVERGING_FIELDS = {"u", "v", "w"}
FIELD_UNITS = {"u": "m/s", "v": "m/s", "w": "m/s", "nrho": "m$^{-3}$", "temp": "K"}

# Dump columns are "id xc yc zc xlo ylo zlo xhi yhi zhi f_ag[1..5]" -- the
# last 5 are named after the averaging fix, not semantically (matches
# tools/plot_field_3d.py's NCOL=15 positional convention; f_ag[1..5] =
# nrho,u,v,w,temp in that order, per this deck's `compute cg`/`compute tg`).
NCOL = 15
FIELD_POS = {"nrho": 10, "u": 11, "v": 12, "w": 13, "temp": 14}

# Cells cut by the curved cell wall can report a wildly inflated nrho
# (observed up to ~2.7e24 m^-3 in this repo's other 3D case, vs ~1.4e22
# anywhere in the real flow) -- a reporting artifact from a tiny true cut
# volume, not a real density. Same convention as tools/plot_field_3d.py.
NRHO_CEILING = 1e23


def load_frame(run_dir, frac, timestep):
    path = os.path.join(run_dir, "field.grid")
    size = os.path.getsize(path)
    tail_bytes = max(int(size * min(1.0, frac * 1.5 + 0.05)), 8 * 1024 * 1024)
    frame_set = read_complete_frames(path, tail_bytes=tail_bytes if tail_bytes < size else None)
    return select_frame(frame_set.frames, timestep=timestep)


def reshape_grid(frame):
    """Turn the flat per-cell dump into structured (nx,ny,nz) arrays.

    Only valid for a flat, single create_grid level. Indexes by the cell's
    lo-bound corner (xlo,ylo,zlo), not its center (xc,yc,zc): a cell that
    SPARTA's cut-cell algorithm splits into sub-cells (because the surface
    carves a disconnected pocket out of it) keeps its parent's exact lo/hi
    bounds on every sub-cell, but each sub-cell gets its OWN centroid --
    indexing by center would treat split sub-cells as extra, spuriously
    distinct grid positions (observed: 53464 rows vs. 53460 = 60*27*33 flat
    cells, from 4 split cells, when indexed by center). Indexing by lo-bound
    correctly folds split sub-cells back onto their one parent grid index.
    """
    if len(frame.columns) != NCOL:
        raise FieldFormatError(
            f"expected {NCOL} columns (id xc yc zc xlo ylo zlo xhi yhi zhi "
            f"f_ag[1..5]), got {frame.columns}")
    d = frame.data
    xlo, ylo, zlo = d[:, 4], d[:, 5], d[:, 6]
    xs, ys, zs = (np.unique(np.round(a, 9)) for a in (xlo, ylo, zlo))
    nx, ny, nz = len(xs), len(ys), len(zs)
    if nx * ny * nz != len(np.unique(np.round(np.stack([xlo, ylo, zlo], axis=1), 9), axis=0)):
        raise FieldFormatError(
            f"{nx}x{ny}x{nz}={nx*ny*nz} distinct grid cells expected from "
            "unique lo-bounds, but that doesn't match the actual number of "
            "unique (xlo,ylo,zlo) combos -- this dump is not a flat "
            "single-level grid (nested refinement isn't supported by this "
            "script)")
    ix = np.searchsorted(xs, np.round(xlo, 9))
    iy = np.searchsorted(ys, np.round(ylo, 9))
    iz = np.searchsorted(zs, np.round(zlo, 9))

    nrho = d[:, FIELD_POS["nrho"]]
    keep = nrho < NRHO_CEILING  # drop cut-cell density-spike artifacts
    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"dropping {n_dropped} cut-cell artifact rows (nrho >= {NRHO_CEILING:g})")

    counts = np.zeros((nx, ny, nz))
    np.add.at(counts, (ix[keep], iy[keep], iz[keep]), 1)

    fields = {}
    for name, pos in FIELD_POS.items():
        total = np.zeros((nx, ny, nz))
        # Accumulate rather than assign: a split cell's surviving sub-cells
        # (after the ceiling filter) share one grid index, so average them
        # instead of letting the last one silently win.
        np.add.at(total, (ix[keep], iy[keep], iz[keep]), d[keep, pos])
        fields[name] = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)
    return xs, ys, zs, fields


def read_surf(path):
    with open(path) as f:
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
    points = np.zeros((n_points + 1, 3))
    for _ in range(n_points):
        parts = lines[i].split()
        points[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
        i += 1
    while lines[i].strip() != "Triangles":
        i += 1
    i += 1
    while lines[i].strip() == "":
        i += 1
    # Triangle line format is "<id> <type> <p1> <p2> <p3>" (confirmed against
    # the raw file -- NOT "<id> <p1> <p2> <p3>"). Matches check_surf3d.py's
    # parsing convention.
    tris = np.zeros((n_tris, 3), dtype=int)
    tri_types = np.zeros(n_tris, dtype=int)
    for k in range(n_tris):
        parts = lines[i].split()
        tri_types[k] = int(parts[1])
        tris[k] = (int(parts[2]), int(parts[3]), int(parts[4]))
        i += 1
    return points, tris, tri_types


def plane_intersection_segments(points, tris, axis, value):
    """Line segments where the mesh crosses the plane `axis == value`."""
    coord = points[:, axis] - value
    # A triangle nearly parallel to the cutting plane (all 3 vertices within
    # TANGENT_EPS of it) does not contribute a meaningful edge to the
    # cross-section outline -- computing one anyway is numerically unstable
    # (near-zero denominators) and, for a whole coplanar panel (e.g. the gas
    # inlet cap's flat face sitting almost exactly at a chosen slice value),
    # produces hundreds of spurious near-arbitrary long segments instead of
    # the panel's real silhouette, which comes from its *non*-tangent
    # neighboring triangles at the panel's edge. 1 micron is far below any
    # real mesh feature here (mm scale) and far above float noise.
    TANGENT_EPS = 1e-6
    segments = []
    for tri in tris:
        s = coord[tri]
        if np.ptp(s) < TANGENT_EPS:
            continue  # triangle is ~parallel to the cutting plane
        signs = np.sign(s)
        if np.all(signs >= 0) or np.all(signs <= 0):
            continue  # triangle does not straddle the plane
        pts3 = points[tri]
        crossings = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            sa, sb = s[a], s[b]
            if sa == 0 or (sa < 0) != (sb < 0):
                if sa == sb:
                    continue
                t = sa / (sa - sb)
                crossings.append(pts3[a] + t * (pts3[b] - pts3[a]))
        if len(crossings) >= 2:
            segments.append((crossings[0], crossings[1]))
    return segments


def other_axes(axis):
    return [a for a in (0, 1, 2) if a != axis]


# The deck's own `region gas` bounds (in.ne_fluor3d_scoping's
# `region gas block 0.031 0.082 0.001 0.043 -0.005 0.039`) -- deliberately
# chosen by the case's author to bound the real interior cavity for particle
# fill, so it's already inset from the housing's outer flanges/bosses/notch
# corners. Used as the zoom row's crop+mask: a geometric point-in-polygon
# test against the raw wall mesh was tried first and repeatedly misclassified
# corner/notch cells (a global parity test fails for a shell with real wall
# thickness; a local nearest-segment-normal test then failed at the notch
# vertices themselves, where no single nearby segment's normal represents
# the true local outward direction) -- this box sidesteps needing any of
# that by reusing a bound the deck's author already verified was interior.
GAS_REGION_BOUNDS_M = ((0.031, 0.082), (0.001, 0.043), (-0.005, 0.039))

# Even a cell inside GAS_REGION_BOUNDS_M can be a genuine cut cell right at
# the real wall (this grid is coarse relative to the geometry -- a known,
# separately-tracked gap), with reported data still mixing in a sliver of
# exterior/ballistic flow regardless of where its nominal center falls.
# Eroding the box inward by roughly one grid cell excludes that boundary
# layer outright, rather than trying to classify it correctly cell-by-cell.
ZOOM_MARGIN_M = 0.005


def eroded_bounds(bounds, margin):
    return tuple((lo + margin, hi - margin) for lo, hi in bounds)


def cell_centers(lo_coords):
    """Approximate cell centers from a flat grid's unique lo-bound values
    (reshape_grid only stores lo-bounds, not hi -- see its docstring)."""
    dx = np.median(np.diff(lo_coords))
    return lo_coords + dx / 2


def plot_slice(ax, xs, ys, zs, fields, field, axis, index, surf, cmap, norm, lims=None, mask_to_cavity=False):
    axes_names = ["x", "y", "z"]
    coords = [xs, ys, zs]
    a1, a2 = other_axes(axis)
    arr = fields[field]
    slab = np.take(arr, index, axis=axis).copy()
    if mask_to_cavity:
        eroded = eroded_bounds(GAS_REGION_BOUNDS_M, ZOOM_MARGIN_M)
        (b1lo, b1hi), (b2lo, b2hi) = eroded[a1], eroded[a2]
        c1c, c2c = cell_centers(coords[a1]), cell_centers(coords[a2])
        inside = ((c1c >= b1lo) & (c1c <= b1hi))[:, None] & ((c2c >= b2lo) & (c2c <= b2hi))[None, :]
        slab = np.where(inside, slab, np.nan)
    # orient so imshow rows/cols follow (a1, a2)
    c1, c2 = coords[a1], coords[a2]
    mesh = ax.pcolormesh(c1 * 1e3, c2 * 1e3, slab.T, cmap=cmap, norm=norm, shading="nearest")
    if surf is not None:
        points, tris, _tri_types = surf
        segs = plane_intersection_segments(points, tris, axis, coords[axis][index])
        for p, q in segs:
            ax.plot([p[a1] * 1e3, q[a1] * 1e3], [p[a2] * 1e3, q[a2] * 1e3],
                    color="0.15", linewidth=0.6, zorder=3)
    ax.set_xlabel(f"{axes_names[a1]} (mm)")
    ax.set_ylabel(f"{axes_names[a2]} (mm)")
    ax.set_title(f"{axes_names[axis]} = {coords[axis][index]*1e3:.1f} mm slice")
    ax.set_aspect("equal")
    if lims is not None:
        (x0, x1), (y0, y1) = lims
        ax.set_xlim(x0 * 1e3, x1 * 1e3)
        ax.set_ylim(y0 * 1e3, y1 * 1e3)
    return mesh


def crop_norm(fields, field, xs, ys, zs, bbox, cmap_default, vmax_override, percentile=75):
    """A (cmap, norm) pair scaled only to cells inside `bbox` -- so a small
    real interior signal isn't washed out by a much larger one elsewhere in
    the domain on a shared scale (see plot_slice's caller).

    Even restricted to the cavity, |field| here tends to be bimodal: a calm
    bulk (median magnitude ~1 m/s in this case) plus a real near-port/wall
    population running up into cut-cell-artifact extremes (~100 m/s by the
    98th percentile). The 98th-percentile-of-everything default used for
    the full-domain scale would just reproduce that problem at a smaller
    scale here, so this row defaults to a much lower percentile (75th) to
    actually show the calm bulk instead of being set by its own outliers.
    """
    (x0, x1), (y0, y1), (z0, z1) = bbox
    xm = (xs >= x0) & (xs <= x1)
    ym = (ys >= y0) & (ys <= y1)
    zm = (zs >= z0) & (zs <= z1)
    mask = xm[:, None, None] & ym[None, :, None] & zm[None, None, :]
    values = fields[field][mask]
    values = values[np.isfinite(values)]
    if field in DIVERGING_FIELDS:
        vmax = vmax_override if vmax_override is not None else (
            np.percentile(np.abs(values), percentile) if values.size else 1.0)
        return "RdBu_r", TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    vmax = vmax_override if vmax_override is not None else (
        np.percentile(values, percentile) if values.size else 1.0)
    vmin = 0.0 if field == "nrho" else (np.percentile(values, 100 - percentile) if values.size else 0.0)
    return cmap_default, Normalize(vmin=vmin, vmax=vmax)


def pick_default_index(fields, axis, weight_field="nrho"):
    """Slice through the plane with the most mass, so an off-axis port isn't
    missed by defaulting to the geometric center."""
    w = np.nan_to_num(fields[weight_field])
    totals = np.nansum(w, axis=tuple(other_axes(axis)))
    return int(np.argmax(totals))


def port_centroid_x(points, tris, tri_types, type_id):
    sel = tris[tri_types == type_id]
    return float(points[sel.ravel(), 0].mean())


def nearest_index(coords, value):
    return int(np.argmin(np.abs(coords - value)))


def plot_xslices(fig, xs, ys, zs, fields, field, x_values_m, surf, cmap, norm, port_labels, mask_to_cavity=False):
    """A row of YZ panels (cross-sections perpendicular to the channel axis,
    x) at the given x positions -- for seeing flow approach/pass/leave a
    port, since both molecule channels run along x (see port_centroid_x)."""
    n = len(x_values_m)
    axs = fig.subplots(1, n, squeeze=False)[0]
    mesh = None
    for ax, xv in zip(axs, x_values_m):
        idx = nearest_index(xs, xv)
        mesh = plot_slice(ax, xs, ys, zs, fields, field, 0, idx, surf, cmap, norm, mask_to_cavity=mask_to_cavity)
        label = port_labels.get(round(xv * 1e3))
        title = f"x = {xs[idx]*1e3:.1f} mm"
        if label:
            title += f"\n({label})"
        ax.set_title(title)
    return axs, mesh


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--field", default="u", choices=list(FIELD_UNITS))
    p.add_argument("--surf", default=None,
                    help="path to cell_fluor3d.surf for wall-outline overlay (opt-in: for this "
                    "case's fused multi-part CAD geometry, a raw plane-mesh intersection often "
                    "shows real but visually dense clutter from real, non-degenerate panels/bosses "
                    "near the cutting plane, not a bug -- it is not a substitute for a proper "
                    "closed-polygon silhouette, which this script does not build)")
    p.add_argument("--x", type=int, default=None, help="x-slice cell index (default: highest-mass plane)")
    p.add_argument("--y", type=int, default=None, help="y-slice cell index (default: highest-mass plane)")
    p.add_argument("--z", type=int, default=None, help="z-slice cell index (default: highest-mass plane)")
    p.add_argument("--xslices", default=None,
                    help="comma-separated x positions in mm -> switch to a row of YZ "
                    "cross-sections at those x's instead of the default 3-panel view "
                    "(both molecule channels run along x, so this is the 'before/after "
                    "an aperture' view). Overrides --x/--y/--z.")
    p.add_argument("--near-ports", action="store_true",
                    help="shorthand for --xslices at -6/0/+6mm around both the molecule "
                    "inlet and outlet channel centroids (needs --surf, to locate them)")
    p.add_argument("--timestep", type=int, default=None)
    p.add_argument("--frac", type=float, default=0.15)
    p.add_argument("--vmax", type=float, default=None,
                    help="manual color-scale bound for the full-domain row (same units as "
                    "--field), overriding the default 98th-percentile bound.")
    p.add_argument("--vmax-zoom", type=float, default=None,
                    help="manual color-scale bound for the cavity-only zoom row (see --zoom). "
                    "Overrides that row's own 98th-percentile-of-the-cavity default.")
    p.add_argument("--zoom", dest="zoom", action="store_true", default=None,
                    help="add a second row cropped+masked to the deck's own `region gas` bounds "
                    "(the interior cavity), with its own color scale fit to just that region -- "
                    "so real but small interior flow isn't hidden by a shared scale dominated by "
                    "a much faster exterior plume. On by default; pass --no-zoom for just the "
                    "single full-domain row. Pass --surf too for a wall-outline overlay on it.")
    p.add_argument("--no-zoom", dest="zoom", action="store_false")
    args = p.parse_args()

    frame = load_frame(args.run_dir, args.frac, args.timestep)
    xs, ys, zs, fields = reshape_grid(frame)

    surf = read_surf(args.surf) if args.surf else None

    arr = fields[args.field]
    finite = arr[np.isfinite(arr)]
    if args.field in DIVERGING_FIELDS:
        # The 98th-percentile magnitude, not the raw max: a handful of cells in a
        # free-streaming exterior plume (100+ m/s) can be two orders of magnitude
        # above genuine interior-cavity flow (order 1 m/s) on the same case's
        # grid -- a max-based scale renders the entire interior as flat white
        # (it isn't 0, it's just small next to those few outliers). --vmax
        # overrides this when a specific range is wanted instead.
        vmax = args.vmax if args.vmax is not None else (
            np.percentile(np.abs(finite), 98) if finite.size else 1.0)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = "RdBu_r"
    else:
        # Same reasoning as the diverging branch above: nrho/temp can have
        # the SAME few-outlier-cells problem (a cut-cell density spike near
        # a port saturating the whole plot to flat purple via matplotlib's
        # raw min/max default) -- confirmed on this case's real settled
        # nrho field, not hypothetical. vmin=0 is used for nrho (a real
        # physical floor) rather than a low percentile.
        vmax = args.vmax if args.vmax is not None else (
            np.percentile(finite, 98) if finite.size else 1.0)
        vmin = 0.0 if args.field == "nrho" else (np.percentile(finite, 2) if finite.size else 0.0)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"

    if args.near_ports and not surf:
        p.error("--near-ports needs --surf to locate the ports")

    do_zoom = args.zoom if args.zoom is not None else True
    bbox = GAS_REGION_BOUNDS_M if do_zoom else None  # viewport: the full (un-eroded) box
    zoom_cmap, zoom_norm = (crop_norm(fields, args.field, xs, ys, zs,
                                       eroded_bounds(bbox, ZOOM_MARGIN_M), cmap, args.vmax_zoom)
                             if do_zoom else (None, None))

    x_values_m = None
    port_labels = {}
    if args.near_ports:
        points, tris, tri_types = surf
        inlet_x = port_centroid_x(points, tris, tri_types, 3)
        outlet_x = port_centroid_x(points, tris, tri_types, 4)
        offsets = (-0.006, 0.0, 0.006)
        x_values_m = [inlet_x + d for d in offsets] + [outlet_x + d for d in offsets]
        for name, x0 in (("molecule inlet", inlet_x), ("molecule outlet", outlet_x)):
            for d in offsets:
                tag = "before" if d < 0 else "after" if d > 0 else "at"
                port_labels[round((x0 + d) * 1e3)] = f"{tag} {name}"
    elif args.xslices:
        x_values_m = [float(v) * 1e-3 for v in args.xslices.split(",")]

    if x_values_m is not None:
        n = len(x_values_m)
        nrows = 2 if do_zoom else 1
        fig = plt.figure(figsize=(4 * n, 5 * nrows))
        subfigs = fig.subfigures(nrows, 1) if do_zoom else [fig]
        axs, mesh = plot_xslices(subfigs[0], xs, ys, zs, fields, args.field, x_values_m, surf, cmap, norm, port_labels)
        cbar = subfigs[0].colorbar(mesh, ax=axs, shrink=0.85, pad=0.02)
        cbar.set_label(f"{args.field} ({FIELD_UNITS[args.field]})")
        if do_zoom:
            (_, _), (y0, y1), (z0, z1) = bbox
            zaxs, zmesh = plot_xslices(subfigs[1], xs, ys, zs, fields, args.field, x_values_m,
                                        surf, zoom_cmap, zoom_norm, port_labels, mask_to_cavity=True)
            for ax in zaxs:
                ax.set_xlim(y0 * 1e3, y1 * 1e3)
                ax.set_ylim(z0 * 1e3, z1 * 1e3)
                ax.set_title(ax.get_title() + " (zoom)")
            zcbar = subfigs[1].colorbar(zmesh, ax=zaxs, shrink=0.85, pad=0.02)
            zcbar.set_label(f"{args.field} ({FIELD_UNITS[args.field]}), cavity-only scale")
        fig.suptitle(f"fluor-cell-3d: {args.field} at step {frame.timestep}")
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out} (step {frame.timestep}, x slices (mm): "
              f"{[round(v*1e3,1) for v in x_values_m]}, zoom row: {do_zoom})")
        return

    ix = args.x if args.x is not None else pick_default_index(fields, 0)
    iy = args.y if args.y is not None else pick_default_index(fields, 1)
    iz = args.z if args.z is not None else pick_default_index(fields, 2)
    slice_specs = [(2, iz), (1, iy), (0, ix)]  # (axis, index) per panel, XY/XZ/YZ

    nrows = 2 if do_zoom else 1
    fig, all_axs = plt.subplots(nrows, 3, figsize=(16, 5 * nrows), squeeze=False)

    meshes = [plot_slice(all_axs[0][col], xs, ys, zs, fields, args.field, axis, index, surf, cmap, norm)
              for col, (axis, index) in enumerate(slice_specs)]
    cbar = fig.colorbar(meshes[0], ax=all_axs[0], shrink=0.85, pad=0.02)
    cbar.set_label(f"{args.field} ({FIELD_UNITS[args.field]})")

    if do_zoom:
        bbox3 = list(bbox)  # (x,y,z) pairs, indexed like other_axes()
        zmeshes = []
        for col, (axis, index) in enumerate(slice_specs):
            a1, a2 = other_axes(axis)
            lims = (bbox3[a1], bbox3[a2])
            zmeshes.append(plot_slice(all_axs[1][col], xs, ys, zs, fields, args.field, axis, index,
                                       surf, zoom_cmap, zoom_norm, lims=lims, mask_to_cavity=True))
            all_axs[1][col].set_title(all_axs[1][col].get_title() + " (zoom)")
        zcbar = fig.colorbar(zmeshes[0], ax=all_axs[1], shrink=0.85, pad=0.02)
        zcbar.set_label(f"{args.field} ({FIELD_UNITS[args.field]}), cavity-only scale")

    fig.suptitle(f"fluor-cell-3d: {args.field} at step {frame.timestep}"
                 f" (highest-mass slice per plane unless overridden)")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} (step {frame.timestep}, slices x[{ix}]={xs[ix]*1e3:.1f}mm "
          f"y[{iy}]={ys[iy]*1e3:.1f}mm z[{iz}]={zs[iz]*1e3:.1f}mm, zoom row: {do_zoom})")


if __name__ == "__main__":
    main()
