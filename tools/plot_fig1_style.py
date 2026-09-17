#!/usr/bin/env python3
"""Single-panel forward-velocity plot matching Takahashi et al. Fig. 1's
style: jet colormap, 0-190 m/s color scale, no density/temperature panels.

Filters by the LOCAL VELOCITY VECTOR's own angle from the forward (+x)
axis, not by cell position -- "full forward" (any cell whose flow is
forward-or-sideways, angle <= 90 deg) is always shown unrestricted;
--backscatter-deg additionally allows up to that many degrees PAST
sideways (angle in (90, 90+backscatter_deg]) so genuine backflow/
backscatter is visible, while cells whose velocity points even further
backward (usually stray, near-single-particle noise -- see the
takahashi-case-and-axis-dip / long-run investigation) are hidden.

This angle mask is only applied OUTSIDE the solid flow channel (tube bore
+ cell bore + aperture throat, using --rb/--rcell/--cell-length/--xexit/
--rap, defaults matching gen_takahashi.py). Cells inside that channel are
always shown unfiltered -- internal circulation there can legitimately
point every which way and isn't "backscatter" in the free-jet-expansion
sense this filter is meant to clean up. Gating on x-position alone (an
earlier version of this script did that) is wrong: the wide domain used
for backscatter viewing includes vacuum space OUTSIDE the cell wall at
x <= xexit (added so wraparound backflow around the plate is visible),
and gas sitting there IS real backscatter even though its x looks
"upstream" -- gating by x alone left that whole region unfiltered no
matter what --backscatter-deg was set to.

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
    p.add_argument("--backscatter-deg", type=float, default=20.0,
                   help="allow the local velocity vector to point up to this many degrees "
                        "PAST sideways (90deg) before a cell is masked out. Full forward "
                        "(<=90deg) is always shown unrestricted. Negative disables masking. "
                        "Only applied downstream of --aperture-x.")
    p.add_argument("--rb", type=float, default=0.002,
                   help="inlet tube bore radius (m), x<0 region.")
    p.add_argument("--rcell", type=float, default=0.00635,
                   help="cell bore radius (m), 0<=x<=cell-length region.")
    p.add_argument("--cell-length", type=float, default=0.053,
                   help="cell body length (m) -- x of the plate's cell-side face.")
    p.add_argument("--xexit", type=float, default=0.0535,
                   help="x position of the aperture exit / plate's downstream face (m).")
    p.add_argument("--rap", type=float, default=0.0025,
                   help="aperture bore radius (m), cell-length<=x<=xexit region.")
    a = p.parse_args()

    try:
        d = average_tail(os.path.join(a.run_dir, "field.grid"), a.frac, fast_tail=True)
    except FieldFormatError as exc:
        raise SystemExit(str(exc)) from exc
    surf = surf_by_type(os.path.join(a.run_dir, a.surf))
    good, verts = verts_and_mirror(d)
    ngood = int(np.sum(good))

    if a.backscatter_deg is not None and a.backscatter_deg >= 0:
        u = d["u"][good]
        v = d["v"][good]
        xc = d["xc"][good]
        yc = d["yc"][good]
        # angle of the local velocity vector from the forward (+x) axis;
        # 0 = straight forward, 90 = straight sideways, 180 = straight backward.
        vel_angle_deg = np.degrees(np.arctan2(np.abs(v), u))
        angle_ok = vel_angle_deg < (90.0 + a.backscatter_deg)
        in_tube = (xc < 0) & (yc <= a.rb)
        in_cell = (xc >= 0) & (xc <= a.cell_length) & (yc <= a.rcell)
        in_throat = (xc > a.cell_length) & (xc <= a.xexit) & (yc <= a.rap)
        inside_channel = in_tube | in_cell | in_throat
        # only mask cells outside the solid flow channel; inside it, keep everything.
        within = inside_channel | angle_ok
    else:
        within = np.ones(ngood, dtype=bool)

    verts = [vv for vv, k in zip(verts, np.concatenate([within, within])) if k]
    keep_mask = np.concatenate([within, within])
    vals = lambda arr: np.concatenate([arr[good], arr[good]])[keep_mask]

    box = d["box"]
    x_span = box[0][1] - box[0][0]
    r_span = 2 * box[1][1]
    height = 6.5
    width = max(6.0, height * x_span / r_span + 2.0)  # +2" for colorbar/labels
    fig, ax = plt.subplots(figsize=(width, height))
    panel(ax, fig, verts, vals(d["u"]),
          r"Velocity component in the x direction (m s$^{-1}$)",
          "jet", surf, box, clim=(a.vmin, a.vmax))
    ax.set_xlabel("x (m)")
    title_suffix = (f" (full forward + {a.backscatter_deg:.0f}deg backscatter)"
                     if (a.backscatter_deg is not None and a.backscatter_deg >= 0) else "")
    ax.set_title("4 K He buffer gas, %s -- mean of %s%s" % (os.path.basename(os.path.normpath(a.run_dir)), frame_label(d), title_suffix))
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
