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
Position-angle masking is NOT what this does -- a cell's position angle
from the aperture can never exceed ~90 deg anyway (that space is solid
plate), so masking by position can't isolate backscatter at all.

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
                        "(<=90deg) is always shown unrestricted. Negative disables masking.")
    a = p.parse_args()

    try:
        d = average_tail(os.path.join(a.run_dir, "field.grid"), a.frac)
    except FieldFormatError as exc:
        raise SystemExit(str(exc)) from exc
    surf = surf_by_type(os.path.join(a.run_dir, a.surf))
    good, verts = verts_and_mirror(d)

    if a.backscatter_deg is not None and a.backscatter_deg >= 0:
        u = d["u"][good]
        v = d["v"][good]
        # angle of the local velocity vector from the forward (+x) axis;
        # 0 = straight forward, 90 = straight sideways, 180 = straight backward.
        vel_angle_deg = np.degrees(np.arctan2(np.abs(v), u))
        within = vel_angle_deg <= (90.0 + a.backscatter_deg)
        verts = [vv for vv, k in zip(verts, np.concatenate([within, within])) if k]
        keep_mask = np.concatenate([within, within])
    else:
        keep_mask = None

    mir = lambda arr: (np.concatenate([arr[good], arr[good]])[keep_mask]
                        if keep_mask is not None else np.concatenate([arr[good], arr[good]]))

    box = d["box"]
    x_span = box[0][1] - box[0][0]
    r_span = 2 * box[1][1]
    height = 6.5
    width = max(6.0, height * x_span / r_span + 2.0)  # +2" for colorbar/labels
    fig, ax = plt.subplots(figsize=(width, height))
    panel(ax, fig, verts, mir(d["u"]),
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
