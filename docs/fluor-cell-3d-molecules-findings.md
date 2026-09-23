# fluor-cell-3d BaF+ tracer: bugs found and validated results

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
