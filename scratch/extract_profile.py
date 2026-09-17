"""Extract the near-aperture on-axis profile (xc,yc,nrho,u,t for cells in a
band around the aperture) from one converged run's field.grid, using the
fast tail-only reader (no full-file read needed since we already verified
convergence). Saves a small CSV for later x-position scanning without
re-reading the multi-GB source file each time.

Usage: python3 extract_profile.py <sccm> <run_dir> <out_csv>
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
d = average_tail(path, 0.25, fast_tail=True)
x = d["xc"]; y = d["yc"]
mask = (x > 0.045) & (x < 0.060) & (y < 0.001)
idx = np.where(mask)[0]

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sccm", "xc", "yc", "nrho", "u", "t"])
    for i in idx:
        w.writerow([sccm, d["xc"][i], d["yc"][i], d["nrho"][i], d["u"][i], d["t"][i]])

print(f"sccm={sccm}: wrote {len(idx)} near-aperture cells -> {out_csv}")
