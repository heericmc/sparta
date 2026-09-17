"""Extract forward velocity (u) and temperature (T) ~3cm downstream of the
aperture exit from one converged run's field.grid, using the fast tail-only
reader. Compute the Takahashi et al. Eq.(5)-style divergence estimate purely
from the He buffer gas's own velocity/temperature field:
  dv_perp (FWHM) = 2*sqrt(2*ln2) * sqrt(kB*T/m_He)   [Maxwellian one-component FWHM]
  dtheta = 2*arctan(dv_perp / (2*u))                  [Eq. 5]
Appends one row (sccm, u, T, dv_perp, dtheta_deg) to a CSV.

Usage: python3 extract_fig8_data.py <sccm> <run_dir> <out_csv>
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/CBGB-SPARTA-2D/tools"))
from plot_fields_b5 import average_tail
import numpy as np

sccm = sys.argv[1]
run_dir = sys.argv[2]
out_csv = sys.argv[3]

kB = 1.380649e-23
m_he = 6.6464731e-27

path = os.path.join(run_dir, "field.grid")
d = average_tail(path, 0.25, fast_tail=True)
x = d["xc"]; y = d["yc"]
# ~3cm downstream of the aperture exit (x=0.0535), matching the paper's Fig.6
# measurement convention ("reported values are measured 3 cm from the exit
# aperture"), on-axis.
x_target = 0.0535 + 0.03
mask = (x > x_target - 0.0015) & (x < x_target + 0.0015) & (y < 0.001)
idx = np.where(mask)[0]
nrho = d["nrho"][idx]; u = d["u"][idx]; t = d["t"][idx]
w = nrho
wsum = w.sum()
u_w = float((w * u).sum() / wsum)
t_w = float((w * t).sum() / wsum)

dv_perp = 2 * math.sqrt(2 * math.log(2)) * math.sqrt(kB * t_w / m_he)
dtheta_rad = 2 * math.atan(dv_perp / (2 * u_w))
dtheta_deg = math.degrees(dtheta_rad)

row = dict(sccm=sccm, u=u_w, T=t_w, dv_perp=dv_perp, dtheta_deg=dtheta_deg, ncell=len(idx))
write_header = not os.path.exists(out_csv)
with open(out_csv, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)

print(f"sccm={sccm}: u={u_w:.2f} T={t_w:.3f} dv_perp={dv_perp:.3f} dtheta={dtheta_deg:.2f}deg "
      f"(ncell={len(idx)}) -> {out_csv}")
