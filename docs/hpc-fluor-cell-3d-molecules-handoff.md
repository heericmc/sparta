# Handoff: tracing BaF+ through the completed 3D neon field

This picks up after `docs/hpc-fluor-cell-3d-handoff.md` -- you should
already have a completed, settled `cases/fluor-cell-3d` neon DSMC run
(a `field.grid` file with several complete, non-transient frames). If you
don't, stop and finish that first; this doc assumes it as an input and
does not repeat that setup.

**What this is**: does the hot (500 eV) BaF+ beam make it through the
cell, how much is lost, and how does that depend on buffer gas flow rate.
Not a detailed ion-collision-physics study -- see the "What this
deliberately does NOT model" section before reading too much into any one
number.

## 0. What's new here

`tracer/ParticleTracing3D.jl` is a new file, not a modification of the
existing `tracer/ParticleTracing.jl` (which stays 2D-axisymmetric-only,
used by `cases/b5-lean` and `cases/takahashi-single`, untouched). It
reuses that file's collision physics unchanged -- `collide!`, the exact
flux-weighted collision-partner sampler, and `freePath` are already a
general elastic two-body treatment, correct for any mass ratio and any
relative velocity, not a near-thermal approximation, so a 500 eV ion
slowing through 23K neon needs no new physics there. What's new is 3D
geometry (segment-vs-triangle-mesh collision detection against
`cell_fluor3d.surf`'s real triangulation, with a spatial grid so this
doesn't degrade to an O(triangle count) scan per free-flight step) and a
true 3D field lookup (no assumed axisymmetry, since this geometry's
off-axis gas inlet genuinely isn't axisymmetric).

**Testing status**: smoke-tested locally (single- and multi-threaded, a
slow near-thermal case and a fast 500 eV-equivalent case) against a
synthetic cube geometry and synthetic field, not against your real
`cell_fluor3d.surf`/`field.grid` -- those weren't available on the
machine that wrote this. Two real bugs were caught and fixed in that
testing (both noted in the file's own comments where they were fixed):
a zero-width lookup-table range that crashes on near-uniform field data,
and a `Threads.threadid()` assumption that breaks on some Julia versions'
thread-pool models. Treat your first real run as the actual first
end-to-end test of this code, and go through the validation checklist in
section 4 before trusting any result from it.

## 1. Freeze one field frame

`field.grid` accumulates one snapshot per `dump grid` interval over the
whole run (append-only) -- same situation as the 2D pipeline, which uses
`tools/field2tracer.py` to pick one frame before the tracer reads it. The
3D case doesn't need that tool's coordinate remapping or He-specific mass-
density conversion (the 3D tracer reads SPARTA's native column layout
directly), so there's a much simpler equivalent:

```bash
python tools/field3d_freeze.py "$DSMC_RUN_ROOT/<your-run-name>" \
  --out cases/fluor-cell-3d/frozen_field.grid
```

Default is the latest complete frame (same selection flags as
`field2tracer.py`: `--timestep`, `--until-step`, `--until-ms` + `--dt`).
Confirm you're not freezing a still-transient frame -- same settling
caveat as the 2D pipeline (README: "Allow the first 10ms for settling").

## 2. Smoke test against the real geometry

Before spending real particle-count/statistics budget, run a handful of
particles and actually look at the output:

```bash
cd cases/fluor-cell-3d
julia --project=../../tracer --threads=4 ../../tracer/ParticleTracing3D.jl \
  cell_fluor3d.surf frozen_field.grid \
  -n 20 -x 0.03048 -y 0.01905 -z 0.01905 \
  --vx 24843.71 --vy 0 --vz 0 \
  -m 20.1797 -M 156.325 --sigma 5.0e-19 \
  --saveall 1 --seed 1
```

Check before anything else:
- It runs without erroring. If it crashes on a range/bounds error, that's
  a real-data edge case this local smoke testing didn't happen to hit --
  see the comments at the two fixed bugs in `ParticleTracing3D.jl` for
  what that class of bug looks like, and don't just silence the error.
- `--spawnclip` (on by default) didn't reject the spawn point as inside
  geometry. The spawn point (0.03048, 0.01905, 0.01905) is the molecule-
  inlet port's location from `gen_fluor3d.py` -- if that's wrong (e.g. the
  port moved because the geometry changed since this doc was written),
  every particle will fail to spawn and it'll error out after 1000 tries.
- The output rows' exit positions and velocities make physical sense (see
  section 4) -- don't just check "did it produce numbers."

## 3. Physics parameters

| Parameter | Value | Why |
|---|---|---|
| `-M` (BaF+ mass) | `156.325` amu | Matches the existing `cases/b5-lean/baf-hot-source.conf` (156.325 u) |
| `-m` (buffer gas mass) | `20.1797` amu | Neon, matches `ne.species` |
| `--vx` | `24843.71` m/s | v = sqrt(2 * 500eV / m_BaF+); recompute if the injection energy changes |
| `-x -y -z` | `0.03048 0.01905 0.01905` | Molecule-inlet port location (2.54mm channel, axial) |
| `--sigma` | `5.0e-19` down to `1.5e-19` m^2 | See below -- run as a scan, not one number |
| `-T` | `0.0` (default) | Perfectly monoenergetic, mono-directional beam. **This is a simplification, not a measurement** -- if the real source has known energy/angular spread, set a nonzero `-T` (adds a thermal-style spread around the mean velocity) or check whether that spread matters for your conclusion before assuming it doesn't. |

**Cross section**: no direct Ba+/BaF+/Ne data exists in the literature
search done for this (see project discussion). Anchored to measured
Ba+ mobility in He and Ar (Dressler et al., *J. Chem. Phys.* 89, 4707
(1988) and 93, 5118 (1990)), converted to a momentum-transfer cross
section and interpolated to Ne by polarizability scaling: central
estimate ~5e-19 m^2. That's a near-thermal (zero-field mobility) value;
at 500 eV the ion is in the repulsive-core regime where the true cross
section is expected to be smaller by a factor of a few, becoming accurate
again once the ion has mostly thermalized. Run the actual transmission/
flow-rate result at at least `--sigma 5.0e-19` and `--sigma 1.5e-19` and
report both -- if the conclusion (e.g. "flow rate X is best") doesn't
change between them, the uncertainty here doesn't matter for the decision
being made; if it does change, that's important to know before trusting
either number alone.

**What "success" and "loss" mean in the output** (columns: `idx x y z
xnext ynext znext vx vy vz collides time`; `xnext,ynext,znext` is where
the particle's path actually ended):
- Transmitted: `xnext` near 0.08255 m (the molecule-outlet port, per
  `gen_fluor3d.py`'s `MOLECULE_OUTLET` location) and `collides` typically
  large (it thermalized and diffused out, not just... coasting straight
  through unimpeded, which is worth checking, see below).
- Lost back out the inlet: `xnext` near 0.03048 m again.
- Stuck on a wall: `xnext` on the housing wall (i.e. NOT at either port's
  x/y/z), matching the existing 2D convention that molecules "stick when
  they hit a wall" -- `collide!`/`propagate` returns as soon as
  `getCollision` reports a wall hit, so a "stuck" particle's final
  `xnext,ynext,znext` in the output IS where it stuck.
Aggregate over runs: transmission fraction = (# ending near the outlet) /
`n`. `--saveall 1` prints every particle regardless of outcome (needed to
compute this ratio); the default (0) only prints particles that left the
simulation BOX bounds, which is a different, box-edge-based criterion
that will undercount "reached the outlet" if the box extends past it (it
does, per the scoping deck's placeholder `create_box`).

## 4. Validation checklist -- do this before trusting a real result

1. **Energy sanity**: for a sample of transmitted/lost particles, is the
   exit speed (`sqrt(vx^2+vy^2+vz^2)`) much lower than the 24843.71 m/s
   injection speed, comparable to the neon field's own velocity scale
   (tens to hundreds of m/s)? If particles are exiting still near
   injection speed with very few `collides`, something is wrong (wrong
   density/cross-section scale, or the particle punched through without
   really seeing the gas) -- don't trust a transmission fraction from
   that run.
2. **Collision count sanity**: cross-check the reported `collides` per
   particle against a rough estimate: mean free path lambda =
   1/(nrho*sigma) using the buffer gas density near the inlet from your
   completed neon run, and the cell's own physical scale (~50mm). A wildly
   different order of magnitude from what a simple diffusion estimate
   (steps ~ (distance/lambda)^2) suggests is worth investigating before
   trusting the run.
3. **Free-edge-style sanity for the tracer's own geometry read**: confirm
   `read_surf3d` parsed the expected point/triangle counts -- add a quick
   `println(length(points), " ", length(triangles))` right after the
   `read_surf3d` call if you want to eyeball this against
   `check_surf3d.py`'s own reported counts for the same file.
4. **Reproducibility**: same `--seed`, same thread count -> same result
   (per the existing 2D file's own documented caveat: threaded runs only
   reproduce at a FIXED thread count, not across different `--threads`
   values). If you can't reproduce a run with its own recorded seed, don't
   trust its output.
5. Run the `--sigma` sensitivity pair from section 3 and report both.

## 5. What this deliberately does NOT model

Carried over from the project discussion that scoped this work, restated
here since it's easy to lose track of once there's a working pipeline
producing numbers:

- **No dissociation / internal energy exchange.** BaF+ is treated as a
  single rigid point mass throughout. A single hard early collision can
  plausibly transfer more energy than BaF's ~5-6 eV bond dissociation
  energy; this model has no way to represent the molecule breaking apart,
  and silently assumes it survives intact for the whole trajectory. If
  the actual question depends on survival fraction (not just transport),
  this model cannot answer it.
- **No charge exchange or other reactive channels.**
- **Energy-independent cross section** (a single `--sigma` value across
  the whole 500eV-to-thermal slowing-down range), not a physically
  modeled energy-dependent one. The sensitivity scan in section 3 is the
  mitigation for this, not a fix for it.
- **No ion-guide/applied field.** Free-flight legs are straight lines
  (unlike the 2D file's harmonic-trap-aware propagation, which isn't
  applicable here since there's no such field in this cell -- confirm
  that assumption is actually right for your setup before trusting a
  trajectory shape).
- **Frozen buffer-gas field** (same one-way-coupling assumption the 2D
  pipeline documents) -- a single 500 eV ion's energy deposition into the
  gas is assumed negligible against the gas's own heat capacity at your
  ion flux; this hasn't been explicitly checked against your actual ion
  rate.
- **`StatsArray`'s r/z binning is a 2D SUMMARY PROJECTION** (axial x vs.
  distance off the x-axis), kept for continuity with the 2D file's output
  shape. It is not itself 3D physics and will not show y/z asymmetry from
  the off-axis gas inlet -- fine for a quick profile, not for anything
  that needs real 3D spatial resolution.

## 6. What to report back

- Whether the smoke test (section 2) and a real statistics-quality run
  (larger `-n`, at least a few hundred particles per condition) completed
  cleanly.
- Transmission / lost-to-inlet / stuck-on-wall fractions, at both
  `--sigma` values from section 3, for whatever flow rate(s) the
  completed neon run(s) used.
- Results of the validation checklist in section 4 -- not just "it ran,"
  but whether the numbers hold together physically.
- If you scan flow rate: which completed neon runs (or newly-run ones at
  different `MDOT`/`FILLN`) you traced against, and the resulting
  transmission-fraction-vs-flow-rate trend.
