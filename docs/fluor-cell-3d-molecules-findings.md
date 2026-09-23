# fluor-cell-3d BaF+ tracer: bugs found and validated results

> **SUPERSEDED (2026-09-23): the neon field these results were traced
> against was wrong.** The surf file had the optical-window slot in the
> housing's top face left open, and ~70% of the injected neon escaped through
> it. The chamber sat at ~2e20 m^-3 instead of ~5.6e21. Everything below the
> "Sealed-window rerun" section was computed on that leaky field and is kept
> only for the record. See [Sealed-window rerun](#sealed-window-rerun) for
> current results.

## Sealed-window rerun

### How the leak was found

The exterior plume was lopsided upward. Measured over half-spaces about the
port axis (y = z = 19.05mm), in a |dy|,|dz| < 38mm window:

| Region | density +z/-z | x-flux +z/-z | control: density +y/-y | control: flux +y/-y | density-weighted mean w |
|---|---|---|---|---|---|
| 2.5-26.5mm past outlet | 4.25 | 5.02 | 0.90 | 0.86 | +31.9 m/s |
| 26.5-117.5mm past outlet | 2.00 | 2.46 | 0.92 | 0.91 | +9.4 m/s |
| behind the back port | 3.10 | 3.86 | 0.82 | 0.81 | +28.7 m/s |

**Ruled out: inlet placement.** Two controlled Takahashi-cell 3D runs gave a
symmetric plume (+z/-z within ~1-5%, the same as their +y/-y controls):

- `cases/takahashi-3d-offset`: inlet tube axis shifted 3mm off the cell axis.
- `cases/takahashi-3d-side`: inlet tube entering through the side wall from
  -z, like this cell's gas inlet.

**Found: an open slot in the top face.** `gen_fluor3d.py` excludes the
optical window (#1/#4) on the assumption that the top face is continuous. It
isn't. A 30 x 7mm slot (x 42-72mm, y 15.4-22.6mm, 229mm^2) runs from z=38.1
down to the chamber ceiling (z~34.3). Evidence of the leak:

- Gas rose inside the chamber, from ~0 m/s at the floor to +16 m/s just
  under the slot.
- It left the slot at ~+82 m/s.
- The upward flux through a plane just above the housing was 1.3e18 /s,
  against 1.79e18 /s injected (4 SCCM).

A matching bottom slot is sealed by the z=-6.35 plate, which is why only the
top leaked.

### Fix

`cases/fluor-cell-3d/seal_top_slot.py` edits `gen_fluor3d.py`'s output
directly (no cadquery needed) and writes `cell_fluor3d_sealed.surf`:

- It removes the slot's 24 side-wall triangles.
- It caps the two rims left behind (the top-face rim and the
  chamber-ceiling rim) with fans wound to match their neighbours.
- The result has 0 free, 0 non-manifold and 0 inconsistently wound edges.
  The solid volume grows by 950mm^3, which is the slot's volume.

Sealed, the chamber fills to the handoff's aperture-law density. There, the
23K Ne mean free path is ~0.4mm, so the flat 3.25mm grid would be ~9 mean
free paths per cell. `in.ne_fluor3d_sealed_run` therefore refines the
chamber and port channels 8x8x8, to ~0.41mm cells (1.49M cells total). Other
run settings:

- FNUM 2e11
- DT 5e-7 (still < 1/4 of the mean collision time)
- 50000-step averaging windows

The run took 600000 steps (0.3 s), as six ~9-minute mit_quicktest jobs
(`tools/wsl/sbatch_fluor3d_sealed.sh`).

### Sealed neon field (averaged over steps 450000-600000)

- **Chamber:** <n> = 5.63e21 m^-3, uniform to <1% with height (aperture-law
  estimate: 5.4e21); T = 23.0 K; all mean velocities < 0.3 m/s.
- **Settling:** total nrho drifts < 1% over the last 250000 steps.
- **Above the housing top:** 4e14 m^-3, down from 3.5e18.
- **Averaging caveat:** a resumed job re-dumps its starting step with an
  empty average. Drop duplicate timesteps before tail-averaging; leaving them
  in lowered averages by ~20%.

| Region | density +z/-z | x-flux +z/-z | control: density +y/-y | control: flux +y/-y | mean w |
|---|---|---|---|---|---|
| 2.5-26.5mm past outlet | 0.833 | 1.243 | 0.830 | 1.247 | -0.36 m/s |
| 26.5-117.5mm past outlet | 0.916 | 1.094 | 0.910 | 1.101 | -0.96 m/s |
| behind the back port | 0.820 | 1.271 | 0.820 | 1.277 | -0.22 m/s |

The plume is now symmetric: z matches the y control to 3 decimals. The ratios
aren't 1.00 because the exterior grid (3.33mm) doesn't split at the axis, so
the cells straddling the axis are all counted in the minus half. The spacing
is the same in y and z, so the bias is the same in both columns.

### BaF+ tracer on the sealed field

Setup: 1000 particles per condition, `--seed 42`, frozen step-600000 frame,
500 eV (24843.71 m/s) injected at the molecule-inlet port. The three
conditions from handoff section 3 (commit 5bca552):

| Cross section | Transmitted | Lost (back out inlet) | Wall hit | Mean collisions |
|---|---|---|---|---|
| energy-dependent (default: 2.0e-18 falling as 1/v to a 1.5e-19 floor) | **0.8%** (8) | 0.1% | 99.1% | 10640 |
| constant 2.0e-18 | 0.0% | 35.1% | 64.9% | 3924 |
| constant 1.5e-19 | 0.3% (3) | 0.0% | 99.7% | 50 |

In the leaky field, the 1.5e-19 and 5.0e-19 runs gave 72-89% transmitted.

At the real chamber density, a 500 eV BaF+ thermalizes within a few mm of
entry. It then diffuses to a wall (the tracer treats a wall hit as a loss)
long before the near-stagnant gas (< 0.3 m/s) can carry it to the 3.175mm
outlet. By condition:

- **Energy-dependent:** wall hits are spread along x = 48-77mm, with a pile
  on the end wall around the outlet channel at x = 77.2mm.
- **Constant 2.0e-18:** ions stop right at the entrance (wall hits at
  x ~ 31-33mm), and 35% diffuse back out the inlet channel.
- **Constant 1.5e-19:** ions coast most of the way and then strike the end
  wall around the outlet.

Thermalization check: ions with > 1000 collisions leave at a mean 74 m/s
(energy-dependent) or 77-87 m/s (constant 2e-18). The flux-weighted mean
speed at 23 K is 66 m/s, so there is no heating artifact. The higher overall
mean exit speed in the energy-dependent run (232 m/s) comes from ions that
hit a wall before thermalizing.

Scripts used (classifier, half-space asymmetry analysis) are in
`tools/fluor3d_asymmetry.py` and `tools/tracer_classify3d.py`.

# Original (leaky-field) results

Follow-up to `docs/hpc-fluor-cell-3d-molecules-handoff.md`, working through its
section 2 smoke test and section 4 validation checklist against the completed,
settled neon run (`cases/fluor-cell-3d`, MDOT=6.002164e-08 kg/s, FILLN=5.4e21,
FNUM=1e11, settled from ~step 290000 onward, frozen at step 360000).

## Bugs found and fixed in `tracer/ParticleTracing3D.jl`

The handoff doc's own testing-status note said to treat the first real run
against this repo's actual geometry/field as the real first end-to-end test.
It was -- the section 2 smoke test (`-n 20`) surfaced two real bugs, neither
caught by the file's synthetic-cube local testing:

1. **Exit positions weren't clipped to the actual crossing point.**
   `freePath` samples free-flight steps up to 1000m as a safety cap (this
   case's box is ~0.2m). When a sampled step overshot the wall mesh or the
   simulation box, `getCollision`/`propagate` reported the raw, un-clipped
   endpoint as the particle's final position instead of where it actually
   crossed. In the first smoke test this put roughly half of 20 particles
   1-289 meters from a 0.2m box -- a silent bad value, not a crash, so
   easy to miss without checking individual rows (which the doc's own
   validation checklist calls for, and which is what caught this).

   Fixed by having `segment_hits_triangle` return the intersection
   fraction `t` (it already computed this internally, via the standard
   Moller-Trumbore parametrization, and simply discarded it), threading
   that through `getCollision` (now returns `(code, t)` instead of a bare
   code), and clipping `xnext` to the true crossing point in `propagate`
   before returning. The clip is nudged a hair (1e-9, sub-angstrom on this
   scale) past the exact boundary so a later re-classification call on the
   same output position reliably re-detects it as outside/on the mesh,
   rather than landing exactly on a boundary where a plain `>`/`<`
   comparison can go either way.

2. **A dummy call broke once (1)'s interface changed.** `SimulateParticles`
   makes one throwaway call to `propagate` purely to measure the output
   tuple's length for array allocation, using an inline placeholder
   `getCollision` of `(x,y)->true` (a stand-in for "the very first
   step always immediately collides", not real physics). That still
   returned a bare `Bool` after the interface changed to `(code, t)`,
   causing a `BoundsError: attempt to access Bool at index [2]` the
   moment the real fix went in. Fixed to `(x,y)->(1,1.0)`, matching the
   new interface.

Both call sites at `getCollision(SPAWN_REF, xpart)` (the `--spawnclip`
check) and the exit-classification re-check were updated for the new
`(code, t)` return shape as well.

Verified after the fix: all exit positions land within the domain; the
[300-particle statistics runs](#results) below ran cleanly with no
further position anomalies.

## Validation checklist (handoff doc section 4)

| Check | Result |
|---|---|
| Geometry read counts | 1127 points / 2258 triangles -- exact match to `check_surf3d.py`'s own count for the same file |
| Reproducibility | Identical output across repeated runs at the same `--seed`/`--threads` |
| Energy sanity | Exit speed retains 81-94% of the 24843.71 m/s injection speed even after collisions -- not down to the neon field's own ~1-200 m/s scale. Not a bug: BaF+ (156u) on Ne (20u) is a heavy-on-light collision (max ~40% KE transfer for a dead-on hit, less for glancing), and mean collision count here is only 0.6-2.1 -- nowhere near enough hits for real thermalization in this short a transit. |
| Collision-count sanity | Path-integrated mean-free-path estimate along the inlet-to-outlet centerline, using the real settled neon field's own density, gives ~5.4 expected collisions at sigma=5.0e-19; observed mean was 2.06. Same order of magnitude; the gap is plausibly explained by particles scattering off the direct centerline after their first hit, shortening their remaining path through the dense gas region relative to the naive straight-line estimate. |
| sigma sensitivity | Run at both 5.0e-19 and 1.5e-19 m^2 -- the transmission fraction moves meaningfully between them (see below), so this cross-section uncertainty is not negligible for a precise number. |

**Classification methodology note**: the handoff doc's literal "transmitted
if `xnext` near 0.08255m [the outlet]" criterion does not hold in practice.
The outlet is an open channel, not a wall -- a particle that passes it with
positive vx just keeps coasting through the near-vacuum downstream region to
the simulation box's far edge (x=0.2m) rather than stopping near the outlet
itself. Classification used instead: a particle that exits at a box face
with x beyond the outlet counts as transmitted; exits at a box face with
x at or before the outlet count as lost back toward the inlet; anything
that does NOT end at a box face (i.e. an interior position) is a wall hit
(stuck). This is well-justified given the mesh is confirmed watertight: a
particle can only leave the housing through a port opening or a wall hit,
so a box-exit downstream of the outlet essentially had to have passed
through it.

## Results

n=300 per condition, `--seed 42`, traced against the one completed neon
run (flow rate corresponding to MDOT=6.002164e-08 kg/s / ~4 SCCM):

| sigma (m^2) | Transmitted | Lost (back out inlet) | Stuck on wall | Mean collides | Ballistic (0 collides) |
|---|---|---|---|---|---|
| 5.0e-19 | 72.3% | 0.0% | 27.7% | 2.06 | 16.0% |
| 1.5e-19 | 89.3% | 0.0% | 10.7% | 0.61 | 57.0% |

The transmission fraction moves by 17 percentage points across the
cross-section uncertainty band -- both values say "mostly transmitted"
qualitatively, but a precise number should be reported as a range, not
collapsed to one point estimate, given this spread.

Notably, "lost back out the inlet" was 0% at both cross sections --- no
particle backscattered hard enough to exit back out the inlet side of the
box at this flow rate. Only one flow rate (the completed neon run) was
traced; a flow-rate scan would need additional neon runs at different
MDOT/FILLN.
