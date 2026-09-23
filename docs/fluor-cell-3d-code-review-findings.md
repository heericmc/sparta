# fluor-cell-3d code review findings

Pre-run correctness sweep across `cases/fluor-cell-3d/` (SPARTA/neon side) and
`tracer/ParticleTracing.jl` (BaF+ ion side), done before committing HPC time
to the buffer-gas run. Repo state at the time of this review: branch
`experiments`, commit `511a0bc` (after the duplicate-edges fix, the
part-fusing fix, and the stray-optical-window fix).

## A. Neon/SPARTA side

**Verified correct:**

- `MDOT`/`FILLN` aperture-law derivation in
  `docs/hpc-fluor-cell-3d-handoff.md` step 4 — independently recomputed from
  scratch (v_mean(23K Ne) = 155.36 m/s, MDOT(4 sccm) = 6.0022e-8 kg/s,
  FILLN ≈ 5.44e21), matches the doc's quoted values exactly. Same method as
  `cases/b5-lean/gen_b5.py`'s He derivation, just re-derived for Ne.
- Ne atomic mass in `ne.species` (3.350918e-26 kg) = 20.1797 amu ×
  1.66053906660e-27 kg/amu. Correct.
- `mixture negas Ne nrho 1.0e22 temp 23.0` in `in.ne_fluor3d_scoping` — looks
  like a placeholder at a glance, but matches the identical convention in
  `cases/b5-lean/in.he_b5_mflow` and `cases/takahashi-3d/in.he_takahashi_3d_scoping`.
  Real flux comes from `emit/surf ... mflow`, not this nrho value. Confirmed
  intentional repo-wide convention, not a copy-paste bug.
- Geometry bounding box is fully contained within the `create_box`/region
  `gas` bounds in both the smoke-test and scoping decks, with reasonable
  margin.
- `gen_fluor3d.py`'s mm→m unit conversion (`MM_TO_M = 1e-3`) is applied once,
  consistently, at final point conversion. No unit-mixing found.
- `rotdof=0` in `ne.species` — correct for a monatomic gas (dof=3,
  translational only), confirmed against Bird 1994 Table A1's own dof=3
  entry for Ne.

**Confirmed bug (small, now quantified):**

- **`ne.vss` VHS reference diameter is off by ~1%.** Recomputing the VHS
  diameter from Bird 1994 Table A1's own inputs for Ne (mass = 3.350918e-26
  kg, reference viscosity μ_ref = 2.975e-5 N·s/m² at T_ref = 273.15 K,
  omega = 0.66) via the standard VHS relation

  ```
  mu_ref = 15 * sqrt(pi * m * kB * T_ref) / (2*pi * d_ref^2 * (5-2*omega)*(7-2*omega))
  ```

  gives **d_ref ≈ 2.766e-10 m**, not the `2.79e-10` currently in
  `cases/fluor-cell-3d/ne.vss`. That's a ~0.9% error in diameter, ~1.7% in
  collision cross-section (sigma ∝ d²) — small, but real and directly
  computable from the table's own numbers, not a rounding artifact.
  **Action: update `ne.vss` diameter to `2.766e-10`** (or re-derive at
  T_ref = 273 K exactly if that's what the printed table uses instead of
  273.15 — the difference is in the 4th significant figure, not the source
  of this discrepancy).

**Already-known, now quantified (not new):**

- The flat `create_grid 60 27 33` in `in.ne_fluor3d_scoping` gives cells
  ≈3.25–3.33mm on a side, vs. the smallest port feature (1.588mm gas-inlet
  cap) — at most 1–2 cells across that port. The handoff doc (step 5) and
  the deck's own TODO comment already flag this as an unresolved grid
  refinement gap. This review confirms it's a real, not hypothetical,
  resolution problem, but doesn't block a first run per the doc.

## B. BaF+/Julia tracer side — not a bug list, a "not built yet" list

1. **`tracer/ParticleTracing.jl` cannot consume this case's output at all.**
   `build_field` parses a 2D-axisymmetric 8-column dump layout
   (`x,y,T,rho,rho_m,vx,vy,vz`) matching the He cases'
   `dump gd grid all ... id xc yc xlo ylo xhi yhi f_ag[*]`.
   `interpolate!` explicitly reconstructs a 3D velocity from one stored
   radial component via `interp[4]*cos(theta), interp[4]*sin(theta)` — a
   hard azimuthal-symmetry assumption. `in.ne_fluor3d_scoping`'s actual dump
   (`id xc yc zc xlo ylo zlo xhi yhi zhi f_ag[*]`, full 3D u,v,w) is a
   different column layout, and the flow is genuinely non-axisymmetric —
   the whole reason this case exists. `getCollision` similarly only
   intersects 2D r-z line segments (2D `.surf` format), not the 3D triangle
   mesh. A 3D field reader/interpolator/collision detector for this case
   does not exist yet (the handoff doc already says so; this confirms it's
   a hard blocker, not an optional convenience).

2. **The elastic hard-sphere collision model is not valid at 500 eV,
   independent of the dimensionality problem above.** `collide!` /
   `exact_partner` do energy-conserving elastic scattering with a fixed,
   energy-independent cross section (a CLI scalar `--sigma`) — no
   energy-dependent cross section, no polarization/Langevin term, no
   charge-exchange channel. This matches what `in.ne_fluor3d_scoping`'s own
   header comment already self-flags: a 500 eV BaF+ ion thermalizing in 23K
   neon is a stopping-power/energy-loss problem (many collisions, large
   fractional energy loss each), "flagged, not implemented." Spawning a
   monoenergetic 500 eV particle works fine today (`-T 0 --vz <speed>`
   gives a delta-function velocity); the collision physics downstream of
   that spawn is what's missing.

3. **`cases/b5-lean/baf-hot-source.conf` is a red herring, not a usable
   template.** `SOURCE_T_K=1000.0` (thermal ablation source, ~0.1 eV mean
   energy — four orders of magnitude below 500 eV), different buffer gas
   (He, not Ne), different cell. Its `CROSS_SECTION_M2=2.7e-18` was tuned
   for BaF-He at ~1000K and must not be reused for BaF+-Ne at 500 eV.
   Flagged explicitly because it's the file someone would naturally
   copy-from, and the mismatch isn't obvious at a glance.

4. **No charge/Coulomb machinery exists anywhere in the tracer.**
   `ne.species`'s charge column is 0.0 (correct — SPARTA only ever sees
   neutral Ne). If a polarization or charge-exchange treatment is ever
   added for the ion side, it's new code, not a missing wire-up of
   something partially built.

**Net assessment:** the neon/SPARTA side is in good shape after the VHS
diameter fix — everything else checked out numerically correct, with one
already-tracked grid-resolution gap that doesn't block a first run. The
BaF+ half is not reachable yet with existing code: it needs (a) a new 3D
field reader/interpolator/collision detector, and (b) an actual
stopping-power/energy-loss collision model, before any 500 eV ion
trajectory result would be physically meaningful. Plotting the neon field
is reachable now; plotting BaF+ characteristics is a separate, scoped
follow-up.
