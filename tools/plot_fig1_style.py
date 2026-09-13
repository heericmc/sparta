#!/usr/bin/env python3
"""Single-panel forward-velocity plot matching Takahashi et al. Fig. 1's
style: jet colormap, 0-190 m/s color scale, no density/temperature panels.

Usage: python plot_fig1_style.py RUN_DIR --surf SURF --out OUT.png [--frac F]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from plot_fields_b5 import average_tail, surf_by_type, verts_and_mirror, panel, frame_label, FieldFormatError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--surf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frac", type=float, default=0.25)
    p.add_argument("--vmin", type=float, default=-5.0)
    p.add_argument("--vmax", type=float, default=190.0)
    p.add_argument("--cone-deg", type=float, default=None,
                   help="mask out cells beyond this half-angle (degrees) from --vertex-x, r=0")
    p.add_argument("--vertex-x", type=float, default=0.0535,
                   help="x-position of the cone vertex (default: aperture exit)")
    p.add_argument("--min-nrho", type=float, default=None,
                   help="drop cells with mean density below this (m^-3) -- removes stray "
                        "near-empty cells (a rare particle over a long run) that show up as "
                        "isolated speckles/streaks far from the real jet structure")
    a = p.parse_args()

    try:
        d = average_tail(os.path.join(a.run_dir, "field.grid"), a.frac)
    except FieldFormatError as exc:
        raise SystemExit(str(exc)) from exc
    surf = surf_by_type(os.path.join(a.run_dir, a.surf))
    if a.min_nrho is not None:
        good = d["nrho"] > a.min_nrho
        verts = []
        for sgn in (1, -1):
            for lo, blo, hi, bhi in zip(d["xlo"][good], d["ylo"][good], d["xhi"][good], d["yhi"][good]):
                verts.append([(lo, sgn * blo), (hi, sgn * blo), (hi, sgn * bhi), (lo, sgn * bhi)])
    else:
        good, verts = verts_and_mirror(d)
    if a.cone_deg is not None:
        import math
        tan_lim = math.tan(math.radians(a.cone_deg))
        dx = d["xc"][good] - a.vertex_x
        # keep points upstream of the vertex (dx<=0, e.g. inside the cell/tube) as-is;
        # downstream of the vertex, mask anything beyond the half-angle cone.
        within = (dx <= 0) | (np.abs(d["yc"][good]) <= tan_lim * np.maximum(dx, 1e-9))
        verts = [v for v, k in zip(verts, np.concatenate([within, within])) if k]
        cone_mask = np.concatenate([within, within])
    else:
        cone_mask = None
    mir = lambda arr: np.concatenate([arr[good], arr[good]])[cone_mask] if cone_mask is not None else np.concatenate([arr[good], arr[good]])

    box = d["box"]
    if a.cone_deg is not None:
        import math
        r_max = math.tan(math.radians(a.cone_deg)) * (box[0][1] - a.vertex_x)
        r_max = min(r_max, box[1][1])
        box = [box[0], [0.0, r_max]]
    x_span = box[0][1] - box[0][0]
    r_span = 2 * box[1][1]
    height = 6.5
    width = max(6.0, height * x_span / r_span + 2.0)  # +2" for colorbar/labels
    fig, ax = plt.subplots(figsize=(width, height))
    panel(ax, fig, verts, mir(d["u"]),
          r"Velocity component in the x direction (m s$^{-1}$)",
          "jet", surf, box, clim=(a.vmin, a.vmax))
    ax.set_xlabel("x (m)")
    title_suffix = f" ({a.cone_deg:.0f}deg cone from x={a.vertex_x})" if a.cone_deg is not None else ""
    ax.set_title("4 K He buffer gas, %s -- mean of %s%s" % (os.path.basename(os.path.normpath(a.run_dir)), frame_label(d), title_suffix))
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
