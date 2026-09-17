#!/usr/bin/env python3
"""Split a run's frames into two independent halves and plot each with the
same density-weighted averaging, to distinguish real geometric/mesh
streaking (same streaks in both halves) from genuine statistical noise
(different streaks in each half, since the halves are independent samples).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from field_io import read_complete_frames
from plot_fields_b5 import surf_by_type, draw_geom

RUN_DIR = "results/he/takahashi-longrun"
SURF = "cell_takahashi.surf"
VMAX = 190.0

result = read_complete_frames(f"{RUN_DIR}/field.grid")
frames = result.frames  # includes step-0 (all zero); skip it
frames = frames[1:]
n = len(frames)
half = n // 2
first_half = frames[:half]
second_half = frames[half:]
print(f"total non-zero-step frames: {n}, first half: {len(first_half)}, second half: {len(second_half)}")


def density_weighted_field(fr_list):
    ids = fr_list[0].data[:, 0]
    stack = np.stack([fr.data for fr in fr_list])  # (nframes, ncells, 11)
    nrho = stack[:, :, 7]
    u = stack[:, :, 8]
    wsum = nrho.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_mean = np.where(wsum > 0, (nrho * u).sum(axis=0) / wsum, 0.0)
    nrho_mean = nrho.mean(axis=0)
    geom = stack[0, :, 1:7]
    return dict(xc=geom[:, 0], yc=geom[:, 1], xlo=geom[:, 2], ylo=geom[:, 3],
                xhi=geom[:, 4], yhi=geom[:, 5], nrho=nrho_mean, u=u_mean)


def verts_and_mirror(d):
    good = d["nrho"] > 0
    v = []
    for sgn in (1, -1):
        for a, b, c, e in zip(d["xlo"][good], d["ylo"][good], d["xhi"][good], d["yhi"][good]):
            v.append([(a, sgn * b), (c, sgn * b), (c, sgn * e), (a, sgn * e)])
    return good, v


surf = surf_by_type(os.path.join(RUN_DIR, SURF))

fig, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
for ax, fr_list, label in [(axes[0], first_half, "FIRST half"), (axes[1], second_half, "SECOND half")]:
    d = density_weighted_field(fr_list)
    good, verts = verts_and_mirror(d)
    mir = lambda a: np.concatenate([a[good], a[good]])
    from matplotlib.collections import PolyCollection
    pc = PolyCollection(verts, array=mir(d["u"]), cmap="jet", edgecolors="none")
    pc.set_clim(-5, VMAX)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, label="u (m/s)", pad=0.01)
    draw_geom(ax, surf)
    ax.set_xlim(-0.02, 0.1544)
    ax.set_ylim(-0.06552, 0.06552)
    ax.set_aspect("equal")
    ax.set_title(f"{label}: {len(fr_list)} frames ({fr_list[0].timestep}-{fr_list[-1].timestep} steps)")
ax.set_xlabel("x (m)")
fig.tight_layout()
out = f"{RUN_DIR}/half_compare.png"
fig.savefig(out, dpi=140)
print("wrote", out)
