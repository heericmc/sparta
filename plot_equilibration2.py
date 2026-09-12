import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = "results/he/first-he-fine"
DT = 1e-7  # s per step, from the deck

# ---------- 1. Parse field.grid (multi-frame grid dump) ----------
def parse_grid_dump(path):
    frames = []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "ITEM: TIMESTEP":
            step = int(lines[i+1])
            i += 2
            assert lines[i].strip() == "ITEM: NUMBER OF CELLS"
            ncells = int(lines[i+1])
            i += 2
            assert lines[i].strip().startswith("ITEM: BOX BOUNDS")
            i += 4
            header = lines[i].strip()
            i += 1
            cols = header.replace("ITEM: CELLS", "").split()
            rows = []
            for _ in range(ncells):
                rows.append(lines[i].split())
                i += 1
            arr = np.array(rows, dtype=float)
            frames.append((step, cols, arr))
        else:
            i += 1
    return frames

frames = parse_grid_dump(f"{RUN_DIR}/field.grid")
cols = frames[0][1]
ci = {c: k for k, c in enumerate(cols)}

def region_avg(arr, xlo, xhi, ylo, yhi, col):
    xc = arr[:, ci["xc"]]
    yc = arr[:, ci["yc"]]
    mask = (xc >= xlo) & (xc < xhi) & (yc >= ylo) & (yc < yhi)
    if mask.sum() == 0:
        return np.nan
    vals = arr[mask, ci[col]]
    if col == "f_ag[1]":
        return vals.mean()
    nrho = arr[mask, ci["f_ag[1]"]]
    w = nrho.copy()
    if w.sum() <= 0:
        return vals.mean()
    return np.average(vals, weights=w)

# three probe regions along the axis
CELL_BOX = (0.010, 0.030, 0.0, 0.010)   # deep bulk cell gas
FRONT_BOX = (0.050, 0.062, 0.0, 0.010)  # front of cell, just upstream of the aperture plate
PLUME_BOX = (0.075, 0.090, 0.0, 0.010)  # far downstream plume

regions = {"in-cell": CELL_BOX, "front-of-cell": FRONT_BOX, "far plume": PLUME_BOX}
colors = {"in-cell": "tab:blue", "front-of-cell": "tab:green", "far plume": "tab:orange"}

t_ms = []
data = {name: {"n": [], "u": [], "T": []} for name in regions}
for step, _, arr in frames:
    t_ms.append(step * DT * 1e3)
    for name, box in regions.items():
        data[name]["n"].append(region_avg(arr, *box, "f_ag[1]"))
        data[name]["u"].append(region_avg(arr, *box, "f_ag[2]"))
        data[name]["T"].append(region_avg(arr, *box, "f_ag[4]"))
t_ms = np.array(t_ms)
for name in regions:
    for k in data[name]:
        data[name][k] = np.array(data[name][k])

# ---------- 2. Parse station flux dumps ----------
def parse_surf_dump_simple(path):
    steps, vals = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == "ITEM: TIMESTEP":
            step = int(lines[i+1]); i += 2
            i += 2
            nsurf = int(lines[i-1])
            i += 4
            i += 1
            total = 0.0
            for _ in range(nsurf):
                parts = lines[i].split(); i += 1
                total += float(parts[-1])
            steps.append(step); vals.append(total)
        else:
            i += 1
    return np.array(steps), np.array(vals)

tf_step, tf_val = parse_surf_dump_simple(f"{RUN_DIR}/tf.out")
tr_step, tr_val = parse_surf_dump_simple(f"{RUN_DIR}/tr.out")
af_step, af_val = parse_surf_dump_simple(f"{RUN_DIR}/af.out")
ar_step, ar_val = parse_surf_dump_simple(f"{RUN_DIR}/ar.out")
tube_net = tf_val - tr_val
ap_net = af_val - ar_val
tf_ms = tf_step * DT * 1e3

# ---------- 3. Global stats table ----------
step_g, np_g, natt_g, ncoll_g = [], [], [], []
with open(f"{RUN_DIR}/run.log") as f:
    text = f.read()
m = re.search(r"Step CPU Np Natt Ncoll f_src\[1\] f_src\[2\]\s*\n(.*?)\nLoop time", text, re.S)
for line in m.group(1).strip().splitlines():
    parts = line.split()
    step_g.append(float(parts[0])); np_g.append(float(parts[2]))
    natt_g.append(float(parts[3])); ncoll_g.append(float(parts[4]))
step_g = np.array(step_g); t_g_ms = step_g * DT * 1e3
np_g = np.array(np_g); natt_g = np.array(natt_g); ncoll_g = np.array(ncoll_g)
coll_frac = ncoll_g / np.where(natt_g == 0, np.nan, natt_g)

# ---------- Plot ----------
fig, axes = plt.subplots(3, 2, figsize=(13, 13))

ax = axes[0, 0]
for name in regions:
    ax.plot(t_ms, data[name]["n"], "-", color=colors[name], label=name, lw=1.3)
ax.set_yscale("log")
ax.set_ylabel("number density $n$ (m$^{-3}$, log)")
ax.set_xlabel("time (ms)")
ax.set_title("Density vs time (0.1ms windows, log scale)")
ax.legend(fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)
ax.grid(alpha=0.3, which="both")

ax = axes[0, 1]
for name in regions:
    ax.plot(t_ms, data[name]["n"], "-", color=colors[name], label=name, lw=1.3)
ax.set_ylabel("number density $n$ (m$^{-3}$, linear)")
ax.set_xlabel("time (ms)")
ax.set_title("Density vs time (linear scale)")
ax.legend(fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)

ax = axes[1, 0]
for name in regions:
    ax.plot(t_ms, data[name]["u"], "-", color=colors[name], label=name, lw=1.3)
ax.set_ylabel("axial (forward) velocity $u$ (m/s)")
ax.set_xlabel("time (ms)")
ax.set_title("Forward velocity vs time")
ax.legend(fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)

ax = axes[1, 1]
for name in regions:
    absu = np.abs(data[name]["u"])
    ax.plot(t_ms, np.where(absu > 0, absu, np.nan), "-", color=colors[name], label=name, lw=1.3)
ax.set_yscale("log")
ax.set_ylabel("|forward velocity| (m/s, log)")
ax.set_xlabel("time (ms)")
ax.set_title("|Forward velocity| vs time (log scale)")
ax.legend(fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)
ax.grid(alpha=0.3, which="both")

ax = axes[2, 0]
ax.plot(tf_ms, tube_net, "-", label="tube station net flux (fwd-rev)", lw=0.8)
ax.plot(tf_ms, ap_net, "-", label="aperture station net flux (fwd-rev)", lw=0.8)
ax.set_ylabel("net flux (window-avg, arb. norm.)")
ax.set_xlabel("time (ms)")
ax.set_title("Station net mass-flux balance (0.1ms cadence)")
ax.legend(fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)

ax = axes[2, 1]
ax.plot(t_g_ms, coll_frac, "o-", color="tab:red", ms=3)
ax.set_ylabel("Ncoll / Nattempt")
ax.set_xlabel("time (ms)")
ax.set_title("Collision acceptance fraction")
ax.axvline(10, color="gray", ls=":", lw=1)

for ax in axes.flat:
    ax.grid(alpha=0.3)

fig.suptitle("CBGB-SPARTA-2D helium run 'first-he-fine': equilibration diagnostics, 0.1ms cadence\n"
             "(dotted line = start of README's 10ms 'settled' window)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{RUN_DIR}/equilibration_fine.png", dpi=140)
print(f"wrote {RUN_DIR}/equilibration_fine.png")
print(f"{len(t_ms)} field-grid frames parsed")
