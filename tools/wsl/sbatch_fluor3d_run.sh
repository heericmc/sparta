#!/bin/bash
#SBATCH --job-name=fluor3d-run
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/12.2.0 openmpi/4.1.4

CASEDIR=cases/fluor-cell-3d
# Same convention as tools/wsl/sbatch_3d_run.sh: large/fast-growing dump goes
# on $SCRATCH, not $HOME (200GB, shared with everything else).
RUNDIR="$HOME/orcd/scratch/cbgb-sparta-3d/fluor-cell-3d-ne"
PLOT_MODULE_PY="deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64"

mkdir -p "$RUNDIR" results/figures
cp -n "$CASEDIR"/in.ne_fluor3d_run "$CASEDIR"/in.ne_fluor3d_resume \
      "$CASEDIR"/cell_fluor3d.surf "$CASEDIR"/ne.species "$CASEDIR"/ne.vss "$RUNDIR/"
cd "$RUNDIR"

# Checkpointed (mirrors cases/takahashi-3d's in.he_takahashi_3d_run/_resume
# pattern): writes a SPARTA restart file every CHECKPOINT_STEPS, so a job
# killed early (walltime limit, node failure) doesn't lose all progress --
# resume from the latest restart.<step> file with in.ne_fluor3d_resume
# (see docs/hpc-fluor-cell-3d-handoff.md for how to check that). Chunked to
# match the field.grid dump cadence (20000 steps/frame): 6 chunks x 20000 =
# 120000 steps total (12ms simulated) -- the same He-case-derived starting
# point as before, NOT yet verified sufficient for this grid/geometry. If
# the settled-field check shows it isn't settled, resume with more chunks
# instead of restarting from scratch.
NCHUNKS=6
CHECKPOINT_STEPS=20000

# --- background progress-plot monitor, same pattern as sbatch_3d_run.sh:
# reads only a tail of the growing field.grid, safe while solver is live.
(
  cd "$SLURM_SUBMIT_DIR"
  module purge
  module load $PLOT_MODULE_PY
  source .venv/bin/activate
  cd "$RUNDIR"
  while true; do
    sleep 300
    if [ ! -f field.grid ]; then continue; fi
    python "$SLURM_SUBMIT_DIR/tools/plot_field_3d.py" . \
      --out "$SLURM_SUBMIT_DIR/results/figures/fluor3d_ne_progress.png" \
      --frac 0.15 --nr 80 2>>monitor.log || true
  done
) &
monitor_pid=$!

# FNUM=1e11, not the doc's original 2.5e17 (carried over from the 2D
# cases/b5-lean case, which turned out wrong by ~5-6 orders of magnitude for
# this geometry -- 2.5e17 produced ZERO simulated particles for the entire
# duration of an earlier 120000-step run, since it rounds both the initial
# fill and the mflow injection rate to 0). Chosen via a short 3-point
# mit_quicktest scan (1e12/5e11/1e11, 10000 steps each): all three gave
# adequate per-cell statistics, but all were also trivially cheap relative
# to the 6h budget here, so 1e11 (~667 particles/occupied cell, ~30min
# extrapolated) was picked for the extra statistics near the under-resolved
# small ports, not because it was the minimum that worked.
mpirun -np "$SLURM_NTASKS" "$HOME/opt/sparta-27Aug2026/bin/spa_mpi" \
  -in in.ne_fluor3d_run \
  -var MDOT 6.002164e-08 -var FILLN 5.4e21 -var FNUM 1e11 -var DT 1e-7 \
  -var DUMPFREQ 20000 -var CHECKPOINT_STEPS $CHECKPOINT_STEPS -var NCHUNKS $NCHUNKS \
  &> run.log
rc=$?
kill "$monitor_pid" 2>/dev/null
echo "solver exit code: $rc"
tail -30 run.log
exit "$rc"
