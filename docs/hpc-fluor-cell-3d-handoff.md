# Handoff: running the 3D fluorescence cooling cell on HPC

This is a from-scratch guide — it assumes you (the reader) have no prior
context on this repo or this case. It gets you from a clean checkout to a
completed **buffer-gas-only** 3D neon DSMC run.

**Scope of this pass: buffer gas (neon) only.** Do not implement or run
anything involving the BaF+ ion/molecule tracer. That's a separate, larger
piece of work (extending `tracer/ParticleTracing.jl` from 2D-axisymmetric to
true 3D) that starts only after the person running this project has reviewed
the buffer-gas results below and confirmed the flow conditions to use. If you
finish everything in this doc, stop and report back — don't start on the
molecule/ion side on your own initiative.

## 0. What this case is

`cases/fluor-cell-3d/` is a **true 3D** SPARTA case (not axisymmetric, unlike
the older `cases/b5-lean/` and `cases/takahashi-single/` cases) for a real CAD
geometry: a cryogenic buffer-gas beam cell with a side-mounted gas inlet, so
the flow is genuinely non-axisymmetric and can't be reduced to 2D. The surface
mesh was generated from a STEP CAD file, defeatured (mounting screw holes
plugged, leaving only the 3 functional ports), and independently verified
(volume, watertightness, and normal-orientation cross-checks against the
source CAD — see the commit that added this case for details).

The three ports, as SPARTA surf types in `cell_fluor3d.surf`:

| Type | What | Notes |
|---|---|---|
| 1 | Generic wall | 23K diffuse-reflecting, everywhere else |
| 2 | Buffer gas inlet | **Sealed** — a capped stub, not an open hole. Gas is injected via `emit/surf` mass-flow, same convention as `cases/b5-lean` and `cases/takahashi-3d` use for their inlets. |
| 3 | Molecule inlet channel | 2.54mm, open to exterior. SPARTA doesn't track molecules — this is tagged only so its neon flux can be measured separately if useful. |
| 4 | Molecule outlet channel | 3.175mm, open to exterior. Same. |

Buffer gas is **23K neon**, not the helium the other cases in this repo use.
`ne.species`/`ne.vss` are new (there was no neon data in this repo before) —
`ne.vss` uses literature VHS parameters (Bird 1994, Table A1: d=2.79e-10m,
omega=0.66), not the simplified hard-sphere convention `he.vss` uses.

## 1. Environment setup

Follow the main [README](../README.md) section "1. Install and clone" for
the base environment (WSL2/Ubuntu if on Windows, apt dependencies, Julia —
you won't need Julia for this pass, but installing it now saves a step
later). Use this repo's URL, not the upstream `arianjad/CBGB-SPARTA-2D` one:

```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/heericmc/sparta.git
cd sparta
git checkout experiments
git pull

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-2d.txt
```

Build the pinned SPARTA solver exactly as the README describes — this case
needs nothing beyond the standard plain-MPI build:

```bash
bash tools/wsl/build_sparta_plain.sh --print-plan
BUILD_JOBS=4 TEST_JOBS=1 bash tools/wsl/build_sparta_plain.sh
export SPARTA_PLAIN_EXE="$HOME/opt/sparta-27Aug2026/bin/spa_mpi"
```

If that exact pinned build (`27Aug2026`) is already installed from earlier
work on this machine, just set `SPARTA_PLAIN_EXE` to it and skip the build.

## 2. Sanity-check the surf file itself (no SPARTA needed)

Before touching SPARTA, confirm the geometry file transferred correctly:

```bash
cd cases/fluor-cell-3d
python3 check_surf3d.py
```

Expect: `3469 points, 6842 triangles`, triangle counts by type
`{1: 6684, 2: 30, 3: 64, 4: 64}`, `Free edges: 0`, and `OK -- watertight,
ready for read_surf` at the end. If any of those numbers differ, **stop and
report back** rather than proceeding — it means the file didn't transfer
correctly (check line endings/binary-vs-text transfer mode) or was edited.

## 3. Smoke test: does SPARTA accept the geometry?

This is a `run 0` — it only checks that SPARTA reads the surface without
errors (correct format, watertight from SPARTA's own perspective, normals
consistent). No physics, no tuning, seconds to run.

```bash
cd ~/code/sparta/cases/fluor-cell-3d
"$SPARTA_PLAIN_EXE" -in in.smoketest_3d_fluor
```

Expect it to run to completion and print stats for a single step with no
errors. If `read_surf` errors out (e.g. about unclosed surfaces or bad
normals), stop and report back with the full error — don't try to work
around it by editing the surf file by hand.

## 4. The real buffer-gas run

`in.ne_fluor3d_scoping` is the actual scoping deck, structured like
`cases/takahashi-3d/in.he_takahashi_3d_scoping`. It needs 5 values supplied
on the command line via `-var` (they are intentionally not hardcoded in the
deck, matching how every other case in this repo works — see
`tools/run_helium.sh` for the pattern):

| Variable | Meaning | Starting value (see derivation below) |
|---|---|---|
| `MDOT` | buffer gas mass flow, kg/s | `6.002164e-08` |
| `FILLN` | initial prefill density, m^-3 | `5.4e21` |
| `FNUM` | simulation particle weight | `2.5e17` (carried over from `cases/b5-lean`, same order-of-magnitude density regime — check particle-per-cell counts once running, see step 5) |
| `DT` | timestep, s | `1e-7` (carried over from `cases/b5-lean` as a starting point — **not verified for this grid**, see step 5) |
| `STEPS` | total steps | start with a **short check**, `2000`, before committing to a long run |

**Where `MDOT` and `FILLN` came from** (do this yourself if you change the
target flow rate — these are NOT universal constants, they scale with SCCM):

```python
import math
kB = 1.380649e-23  # J/K
amu = 1.66053906660e-27  # kg
m_ne = 20.1797 * amu
T = 23.0
v_mean_ne = math.sqrt(8*kB*T/(math.pi*m_ne))          # 155.34 m/s

sccm = 4.0                        # <-- change this to scan flow rate
ndot = sccm * 4.478e17            # particles/s per SCCM (standard conversion)
mdot = ndot * m_ne                # kg/s  -> this is MDOT

RAP = 3.175e-3 / 2                # molecule-outlet radius, the dominant exit
filln = 4*ndot / (1.07*v_mean_ne*math.pi*RAP**2)   # m^-3 -> this is FILLN
```

This is the exact same aperture-law method `cases/b5-lean/gen_b5.py` uses for
its He fill estimate, just re-derived for neon at 23K. **Caveat specific to
this geometry**: the B5 cell has one exit aperture; this cell has *two* open
channels (molecule inlet, 2.54mm, AND molecule outlet, 3.175mm) that buffer
gas can escape through. The formula above only accounts for the outlet, so
it's an overestimate — expect the true equilibrium density to settle lower.
This doesn't need to be exact: `FILLN` only affects how long the run takes to
settle to equilibrium, not what that equilibrium actually is (SPARTA
determines that self-consistently from the geometry, `MDOT`, and boundary
conditions) — same caveat the README states for the He case ("This is an
initial estimate, not a measured equilibrium density").

Run the short check first:

```bash
cd ~/code/sparta/cases/fluor-cell-3d
"$SPARTA_PLAIN_EXE" -in in.ne_fluor3d_scoping \
  -var MDOT 6.002164e-08 -var FILLN 5.4e21 -var FNUM 2.5e17 \
  -var DT 1e-7 -var STEPS 2000
```

If that completes cleanly, move to a real settling run. Increase `STEPS`
substantially — the He example needed 120,000 steps (12ms simulated) for
10ms settling + 2ms sampling at its `DT`; this geometry and grid haven't been
characterized the same way, so **don't assume the same step count is enough
or that `DT`/grid are already right** — see step 5 before trusting a long
run's results.

For multi-rank runs, use `mpirun -np N` in front of the command, same as
every other case:

```bash
mpirun -np 4 "$SPARTA_PLAIN_EXE" -in in.ne_fluor3d_scoping -var MDOT ... [etc]
```

## 5. Known gaps you'll need to close — don't skip this

This case is newer and less mature than `cases/b5-lean`/`cases/takahashi-3d`.
Specifically:

- **`create_box`/`create_grid` in `in.ne_fluor3d_scoping` are placeholders**,
  not derived from a mean-free-path calculation. The mature 2D case
  (`in.he_b5_mflow`) uses nested 2:1 refinement regions sized so the finest
  cells resolve the smallest geometric features (see its `region level*`
  blocks and `gen_b5.py`'s `nonface()` checks, which assert cell boundaries
  don't land exactly on a geometric feature). This case has none of that yet
  — a single flat grid level. Before trusting quantitative results, check
  the actual mean free path in the settled field against the cell size (see
  [Controlling numerical accuracy](accuracy.md) for the general method) and
  refine the grid around the three ports (1.588/2.54/3.175mm — all much
  smaller than a naive flat grid cell) the same way the 2D case does.
- **No 3D equivalent of `tools/check_grid_2d.py` exists.** The 2D case has an
  automated nested-refinement checker; this one doesn't yet. You'll need to
  either write one or check manually.
- **No 3D equivalent of `tools/plot_fields_b5.py` exists** for visualizing
  the field dump — that tool explicitly only reads the 2D `Points`/`Lines`
  surf format (see `tools/plot_fields.py`'s `wall_segments()`, which parses
  `Lines` records — our surf file has `Triangles` records instead, a
  different format). For now, inspect `field.grid` as text, or write a
  minimal 3D reader before trying to reuse the existing plotting code.
- **No automated run-launcher** like `tools/run_helium.sh`/`run_lean.sh`
  exists for this case (those are 2D-specific, staging `.surf`/`.species`
  files and writing a provenance manifest). This doc has you invoke
  `spa_mpi` directly. If you're doing enough runs that this gets painful,
  consider writing an equivalent launcher rather than hand-invoking every
  time — but that's optional, not required to get a first result.

None of these block getting a first run done, but they matter before trusting
the *quantitative* result (e.g. before reading off a specific density or
flow number to report back).

## 6. What to check afterward, and what to send back

Once you have a completed run:

1. Confirm it actually finished: check the tail of the run's stdout/log for
   the final `run` command's stats, and that SPARTA exited without an error
   (no crash, no `ERROR:` lines).
2. Look at the `dump gd grid all ... field.grid` output the deck writes —
   this is the neon density/velocity/temperature field, dumped every 20000
   steps (averaged over 2000-step windows, per the `fix ag ave/grid`
   settings — same cadence as the He cases). Confirm density near the sealed
   gas-inlet cap is elevated relative to downstream (buffer gas should be
   visibly flowing from the type-2 cap toward the two open channels), and
   that the field looks like it's stopped changing between the last couple
   of saved frames (settled), not still transient.
3. Report back: whether it ran cleanly, how many steps you needed for
   settling, what `MDOT`/`FILLN`/`FNUM`/`DT`/grid you ended up using (if you
   changed anything from the starting values above), and either the raw
   `field.grid` file or a summary of the settled density/velocity near each
   of the three ports. That's what gets reviewed before deciding on a flow
   rate and moving to the molecule/ion tracer work.

Do not proceed to the BaF+/ion tracer extension without that review and an
explicit go-ahead — this doc's scope stops here.
