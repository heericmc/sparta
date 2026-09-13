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
    p.add_argument("--vmax", type=float, default=190.0)
    a = p.parse_args()

    try:
        d = average_tail(os.path.join(a.run_dir, "field.grid"), a.frac)
    except FieldFormatError as exc:
        raise SystemExit(str(exc)) from exc
    surf = surf_by_type(os.path.join(a.run_dir, a.surf))
    good, verts = verts_and_mirror(d)
    mir = lambda arr: np.concatenate([arr[good], arr[good]])

    box = d["box"]
    x_span = box[0][1] - box[0][0]
    r_span = 2 * box[1][1]
    height = 6.5
    width = max(6.0, height * x_span / r_span + 2.0)  # +2" for colorbar/labels
    fig, ax = plt.subplots(figsize=(width, height))
    panel(ax, fig, verts, mir(d["u"]),
          r"Velocity component in the x direction (m s$^{-1}$)",
          "jet", surf, d["box"], clim=(0.0, a.vmax))
    ax.set_xlabel("x (m)")
    ax.set_title("4 K He buffer gas, %s -- mean of %s" % (os.path.basename(os.path.normpath(a.run_dir)), frame_label(d)))
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
