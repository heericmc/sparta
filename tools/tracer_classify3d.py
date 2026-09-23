#!/usr/bin/env python3
"""Usage: python tools/tracer_classify3d.py OUT [OUT ...]
Classify ParticleTracing3D.jl --saveall output (idx x y z xn yn zn vx vy vz collides time):
box-face exit beyond the outlet = transmitted, box exit at/before it = lost, interior end = wall hit."""
import sys
import numpy as np
OUTLET_X = 0.08255
BOX = (0.005, 0.200, -0.020, 0.070, -0.030, 0.080)
EPS = 1e-5
for path in sys.argv[1:]:
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) == 12 and not p[0].startswith("-"):
            try: rows.append([float(q) for q in p])
            except ValueError: pass
    r = np.array(rows)
    idx, x, y, z, xn, yn, zn, vx, vy, vz, col, t = r.T
    at_box = np.zeros(len(r), bool)
    for c, (lo, hi) in zip((xn, yn, zn), ((BOX[0], BOX[1]), (BOX[2], BOX[3]), (BOX[4], BOX[5]))):
        at_box |= (abs(c - lo) < EPS) | (abs(c - hi) < EPS)
    tr, lost, stuck = at_box & (xn > OUTLET_X), at_box & (xn <= OUTLET_X), ~at_box
    n = len(r); sp = np.sqrt(vx**2 + vy**2 + vz**2)
    print(f"{path}: n={n}  transmitted {tr.sum()} ({100*tr.mean():.1f}%)  lost {lost.sum()} ({100*lost.mean():.1f}%)  "
          f"wall {stuck.sum()} ({100*stuck.mean():.1f}%)")
    print(f"  collides mean {col.mean():.0f} median {np.median(col):.0f} max {col.max():.0f}; ballistic {(col==0).sum()}; "
          f"exit speed mean {sp.mean():.0f} m/s; transit time median {np.median(t)*1e3:.3f} ms")
    if tr.any():
        print(f"  transmitted: exit speed mean {sp[tr].mean():.0f} m/s, collides mean {col[tr].mean():.0f}, time median {np.median(t[tr])*1e3:.3f} ms")
    if stuck.any():
        print(f"  wall-hit x positions (mm): quartiles {np.percentile(xn[stuck]*1e3,[5,25,50,75,95]).round(1)}")
