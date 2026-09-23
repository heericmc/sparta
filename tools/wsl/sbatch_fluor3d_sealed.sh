#!/bin/bash
#SBATCH --job-name=fluor3d-sealed
#SBATCH --account=mit_general
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.log

# Sealed-window fluor-cell-3d neon run (cases/fluor-cell-3d/in.ne_fluor3d_sealed_run:
# cell_fluor3d_sealed.surf + 8x8x8-refined chamber). Chunked for
# mit_quicktest's 15-min cap: NCHUNKS (default 2) x 50000 steps per job (~9 min: ~50s per 10000 steps), restart
# written after every chunk; pass RESTARTFILE=restart.%.<step> to continue.
# Keep NP=40 across the chain (see sbatch_fluor3d_resume.sh).
#
# DT 5e-7 (not 1e-7): the sealed chamber needs ~100ms+ of simulated time to
# fill (pump-out time V/C ~ 50cm^3 / (v_mean/4 x 13mm^2 of ports) ~ 0.1s),
# and 5e-7 is still < 1/4 of the ~2.4us mean collision time at 5e21 m^-3 and
# moves a mean-speed atom 0.08mm (< 1/5 of a 0.41mm chamber cell) per step.
# FNUM 2e11: ~1-2 simulated particles per chamber cell at the full fill,
# similar per-cell statistics to the settled Takahashi 3D reference.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load gcc/12.2.0 openmpi/4.1.4

CASEDIR=cases/fluor-cell-3d
RUNDIR="$HOME/orcd/scratch/cbgb-sparta-3d/fluor-cell-3d-ne-sealed"
mkdir -p "$RUNDIR"
cp -n "$CASEDIR"/in.ne_fluor3d_sealed_run "$CASEDIR"/in.ne_fluor3d_sealed_resume \
      "$CASEDIR"/cell_fluor3d_sealed.surf "$CASEDIR"/ne.species "$CASEDIR"/ne.vss "$RUNDIR/"
cd "$RUNDIR"

COMMON="-var MDOT 6.002164e-08 -var FILLN 5.4e21 -var FNUM 2e11 -var DT 5e-7 \
        -var DUMPFREQ 50000 -var CHECKPOINT_STEPS 50000 -var NCHUNKS ${NCHUNKS:-2}"
if [ -n "${RESTARTFILE:-}" ]; then
  mpirun -np "$SLURM_NTASKS" "$HOME/opt/sparta-27Aug2026/bin/spa_mpi" \
    -in in.ne_fluor3d_sealed_resume -var RESTARTFILE "$RESTARTFILE" $COMMON &>> run.log
else
  mpirun -np "$SLURM_NTASKS" "$HOME/opt/sparta-27Aug2026/bin/spa_mpi" \
    -in in.ne_fluor3d_sealed_run $COMMON &> run.log
fi
rc=$?
echo "solver exit code: $rc"
tail -5 run.log
exit "$rc"
