#!/usr/bin/env python3
"""Usage: python tools/fluor3d_asymmetry.py RUN_DIR/field.grid
Asymmetry + leak check for a fluor-cell-3d run (flat or nested grid):
volume-weighted, tail-averaged over the last 30% of field.grid frames."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_io import read_complete_frames
from plot_field_3d_slices import NRHO_CEILING
fr = read_complete_frames(sys.argv[1]).frames
# a resumed job re-dumps its starting step with an empty average; keep the first copy of each step
seen = set(); fr = [f for f in fr if not (f.timestep in seen or seen.add(f.timestep))]
print('frame nrho sums', [f'{f.timestep}:{f.data[:, 10].sum():.2e}' for f in fr[-6:]])
k = max(1, int(round(len(fr) * 0.3))); fr = fr[-k:]
tot = 0
for f in fr:
    tot = tot + f.data[np.argsort(f.data[:, 0], kind="stable")]
A = tot / len(fr)
print("frames", [f.timestep for f in fr])
xc, yc, zc = A[:, 1]*1e3, A[:, 2]*1e3, A[:, 3]*1e3
vol = np.prod(A[:, 7:10] - A[:, 4:7], axis=1) * 1e9   # mm^3
n, u, v, w, T = A[:, 10], A[:, 11], A[:, 12], A[:, 13], A[:, 14]
n = np.where(n < NRHO_CEILING, n, 0)
N = n * vol                                          # (volume-weighted density)
AY = AZ = 19.05; XEXIT = 82.55
dy, dz = yc - AY, zc - AZ
def avg(q, m): return (N*q)[m].sum() / N[m].sum()
print("\n=== exterior asymmetry, window |dy|,|dz|<38mm")
for lab, m in [("2.5-26.5mm past outlet", (xc > XEXIT+2.5) & (xc < XEXIT+26.5)),
               ("26.5-117.5mm past outlet", (xc > XEXIT+26.5)),
               ("behind back port (x<30.5)", (xc < 30.48))]:
    m = m & (abs(dy) < 38) & (abs(dz) < 38)
    sgn = 1 if "past" in lab else -1
    r = lambda s: (N[m&s].sum()/N[m&~s].sum(), (N*u)[m&s].sum()/(N*u)[m&~s].sum())
    zr, yr = r(dz > 0), r(dy > 0)
    print(f"{lab:27s} dens +z/-z {zr[0]:.3f} flux {zr[1]:.3f} | +y/-y dens {yr[0]:.3f} flux {yr[1]:.3f} | "
          f"<w> {avg(w,m):+6.2f} <v> {avg(v,m):+6.2f} <u> {avg(u,m):+7.2f}")
inh = (xc > 30.48) & (xc < 82.55) & (yc > 0) & (yc < 44.45) & (zc > -6.35) & (zc < 38.1) & (n > 0)
print(f"\n=== chamber: <n> {N[inh].sum()/vol[inh].sum():.3e} m^-3, T {avg(T,inh):.1f} K, "
      f"<u> {avg(u,inh):+.2f} <v> {avg(v,inh):+.2f} <w> {avg(w,inh):+.2f}")
cav = inh & (xc > 36) & (xc < 72)
zb = np.arange(0, 37, 3.0)
for z0 in zb:
    m = cav & (zc >= z0) & (zc < z0 + 3)
    if m.any(): print(f" z {z0:4.0f}-{z0+3:2.0f}: <w> {avg(w,m):+6.2f} <u> {avg(u,m):+6.2f} <n> {N[m].sum()/vol[m].sum():.2e}")
above = (zc > 38.1) & (zc < 48) & (xc > 30.5) & (xc < 82.5)
print(f"\nabove housing top (z 38-48): <n> {N[above].sum()/max(vol[above].sum(),1):.2e}  (sealed should be ~ambient spill only)")
