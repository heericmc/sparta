"""Extract on-axis aperture-exit conditions (nrho, u, T) from one SPARTA
run's field.grid, compute Mach number M and Reynolds number R, and append
the result as one row to a CSV. Run once per SCCM point (one process each,
to keep memory bounded -- see wsl_sparta_gotchas memory).

Usage: python3 extract_fig2_data.py <sccm> <run_dir> <out_csv>
"""
import csv
import os
import sys

sys.path.insert(0, os.path.expanduser("~/CBGB-SPARTA-2D/tools"))
from plot_fields_b5 import average_tail
import numpy as np

sccm = sys.argv[1]
run_dir = sys.argv[2]
out_csv = sys.argv[3]

path = os.path.join(run_dir, "field.grid")
d = average_tail(path, 0.25)  # last 25% of frames; also gives full convergence trend
x = d["xc"]; y = d["yc"]
# window ends AT the aperture exit plane (0.0535), not past it -- extending even
# 1-2mm downstream samples the free-jet expansion (n drops, M rises fast there)
# and inflates M while barely moving R. Confirmed on sccm1: this window gives
# M=0.68 vs paper's read ~0.61, while the old 0.0530-0.0550 window gave M=1.04.
mask = (x > 0.0525) & (x < 0.0535) & (y < 0.0008)
idx = np.where(mask)[0]
nrho = d["nrho"][idx]; u = d["u"][idx]; t = d["t"][idx]
w = nrho
wsum = w.sum()
u_w = float((w * u).sum() / wsum)
t_w = float((w * t).sum() / wsum)
nrho_avg = float(nrho.mean())

kB = 1.380649e-23
m_he = 6.6464731e-27
sigma = 2e-19
cs = (5.0 / 3.0 * kB * t_w / m_he) ** 0.5
M = u_w / cs
lam = 1.0 / (2 ** 0.5 * nrho_avg * sigma)
K = lam / 0.005
R = 2 * M / K

# convergence check: how much did bulk density drift over the run?
trend = d["trend"]
drift_pct = 100.0 * (trend[-1][1] - trend[0][1]) / trend[-1][1]

row = dict(sccm=sccm, run_dir=run_dir, nrho=nrho_avg, u=u_w, T=t_w, M=M, R=R,
           nframes=d["nframe"], ntot_frames=d["ntot"], drift_pct=drift_pct)

write_header = not os.path.exists(out_csv)
with open(out_csv, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)

print(f"sccm={sccm}: nrho={nrho_avg:.3e} u={u_w:.2f} T={t_w:.3f} M={M:.4f} R={R:.4f} "
      f"drift={drift_pct:.1f}% ({d['nframe']}/{d['ntot']} frames used) -> {out_csv}")
