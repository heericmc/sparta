import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = "results/he/first-he"
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
            i += 4  # skip 3 bound lines + move past
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
# columns: id xc yc xlo ylo xhi yhi f_ag[1]=nrho f_ag[2]=u f_ag[3]=v f_ag[4]=temp

def region_avg(arr, xlo, xhi, ylo, yhi, col):
    xc = arr[:, ci["xc"]]
    yc = arr[:, ci["yc"]]
    mask = (xc >= xlo) & (xc < xhi) & (yc >= ylo) & (yc < yhi)
    if mask.sum() == 0:
        return np.nan
    vals = arr[mask, ci[col]]
    nrho = arr[mask, ci["f_ag[1]"]]
    # density-weight the velocity/temp average (particles-weighted), simple mean for nrho itself
    if col == "f_ag[1]":
        return vals.mean()
    w = nrho.copy()
    if w.sum() <= 0:
        return vals.mean()
    return np.average(vals, weights=w)

# in-cell probe box: well inside the buffer-gas cell body, near axis
CELL_BOX = (0.020, 0.040, 0.0, 0.010)
# far-plume probe box: downstream of the aperture, in free expansion
PLUME_BOX = (0.075, 0.090, 0.0, 0.010)

t_ms, n_cell, n_plume, u_cell, u_plume, T_cell, T_plume = [], [], [], [], [], [], []
for step, _, arr in frames:
    t_ms.append(step * DT * 1e3)
    n_cell.append(region_avg(arr, *CELL_BOX, "f_ag[1]"))
    n_plume.append(region_avg(arr, *PLUME_BOX, "f_ag[1]"))
    u_cell.append(region_avg(arr, *CELL_BOX, "f_ag[2]"))
    u_plume.append(region_avg(arr, *PLUME_BOX, "f_ag[2]"))
    T_cell.append(region_avg(arr, *CELL_BOX, "f_ag[4]"))
    T_plume.append(region_avg(arr, *PLUME_BOX, "f_ag[4]"))

t_ms = np.array(t_ms)

# ---------- 2. Parse station flux dumps (finer cadence) ----------
def parse_surf_dump(path, field_name):
    steps, vals = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == "ITEM: TIMESTEP":
            step = int(lines[i+1]); i += 2
            assert lines[i].strip() == "ITEM: NUMBER OF SURFS"
            nsurf = int(lines[i+1]); i += 2
            i += 4  # BOX BOUNDS block
            header = lines[i].split(); i += 1
            fidx = header.index(field_name)
            total = 0.0
            for _ in range(nsurf):
                parts = lines[i].split(); i += 1
                total += float(parts[fidx - 2])  # header has "ITEM:" "CELLS/SURFS" prefix tokens stripped below
            steps.append(step); vals.append(total)
        else:
            i += 1
    return np.array(steps), np.array(vals)

def parse_surf_dump_simple(path):
    steps, vals = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == "ITEM: TIMESTEP":
            step = int(lines[i+1]); i += 2
            i += 2  # NUMBER OF SURFS + value
            nsurf = int(lines[i-1])
            i += 4  # BOX BOUNDS block (3 lines) - already consumed 1 via NUMBER OF SURFS block above; recompute below
            i += 1  # header line "ITEM: SURFS id f_x"
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

# ---------- 3. Parse global stats table from run.log ----------
step_g, np_g, natt_g, ncoll_g, fsrc1_g, fsrc2_g = [], [], [], [], [], []
with open(f"{RUN_DIR}/run.log") as f:
    text = f.read()
m = re.search(r"Step CPU Np Natt Ncoll f_src\[1\] f_src\[2\]\s*\n(.*?)\nLoop time", text, re.S)
for line in m.group(1).strip().splitlines():
    parts = line.split()
    step_g.append(float(parts[0])); np_g.append(float(parts[2]))
    natt_g.append(float(parts[3])); ncoll_g.append(float(parts[4]))
    fsrc1_g.append(float(parts[5])); fsrc2_g.append(float(parts[6]))
step_g = np.array(step_g); t_g_ms = step_g * DT * 1e3
np_g = np.array(np_g); natt_g = np.array(natt_g); ncoll_g = np.array(ncoll_g)
coll_frac = ncoll_g / natt_g

# ---------- Plot ----------
fig, axes = plt.subplots(3, 2, figsize=(13, 12))

ax = axes[0, 0]
ax.plot(t_ms, n_cell, "o-", label="in-cell (x=20-40mm, r<10mm)")
ax.plot(t_ms, n_plume, "s-", label="far plume (x=75-90mm, r<10mm)")
ax.set_ylabel("number density $n$ (m$^{-3}$)")
ax.set_xlabel("time (ms)")
ax.set_title("Density vs time (2ms-averaged windows)")
ax.legend(fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)

ax = axes[0, 1]
ax.plot(t_ms, u_cell, "o-", label="in-cell")
ax.plot(t_ms, u_plume, "s-", label="far plume")
ax.set_ylabel("axial (forward) velocity $u$ (m/s)")
ax.set_xlabel("time (ms)")
ax.set_title("Forward velocity vs time")
ax.legend(fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)

ax = axes[1, 0]
ax.plot(t_ms, T_cell, "o-", label="in-cell")
ax.plot(t_ms, T_plume, "s-", label="far plume")
ax.set_ylabel("translational temperature (K)")
ax.set_xlabel("time (ms)")
ax.set_title("Temperature vs time")
ax.legend(fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)

ax = axes[1, 1]
ax.plot(t_g_ms, np_g, "o-", color="tab:purple")
ax.set_ylabel("total simulator particles $N_p$")
ax.set_xlabel("time (ms)")
ax.set_title("Global particle count (whole domain)")
ax.axvline(10, color="gray", ls=":", lw=1)

ax = axes[2, 0]
ax.plot(tf_ms, tube_net, "-", label="tube station net flux (fwd-rev)")
ax.plot(tf_ms, ap_net, "-", label="aperture station net flux (fwd-rev)")
ax.set_ylabel("net flux (window-avg, arb. norm.)")
ax.set_xlabel("time (ms)")
ax.set_title("Station net mass-flux balance (0.1ms cadence)")
ax.legend(fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)

ax = axes[2, 1]
ax.plot(t_g_ms, coll_frac, "o-", color="tab:red")
ax.set_ylabel("Ncoll / Nattempt")
ax.set_xlabel("time (ms)")
ax.set_title("Collision acceptance fraction")
ax.axvline(10, color="gray", ls=":", lw=1)

for ax in axes.flat:
    ax.grid(alpha=0.3)

fig.suptitle("CBGB-SPARTA-2D helium run 'first-he': equilibration diagnostics over 12 ms\n"
             "(dotted line = start of README's 10ms 'settled' window)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results/he/first-he/equilibration.png", dpi=140)
print("wrote results/he/first-he/equilibration.png")

print("\nSummary table (grid-averaged, 2ms windows):")
print(f"{'t(ms)':>6} {'n_cell':>12} {'n_plume':>12} {'u_cell':>9} {'u_plume':>9} {'T_cell':>8} {'T_plume':>8}")
for i in range(len(t_ms)):
    print(f"{t_ms[i]:6.1f} {n_cell[i]:12.4e} {n_plume[i]:12.4e} {u_cell[i]:9.2f} {u_plume[i]:9.2f} {T_cell[i]:8.2f} {T_plume[i]:8.2f}")
