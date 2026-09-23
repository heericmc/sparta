#!/bin/bash
#SBATCH --job-name=fluor3d-resume
#SBATCH --account=mit_general
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.log

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/12.2.0 openmpi/4.1.4

CASEDIR=cases/fluor-cell-3d
RUNDIR="$HOME/orcd/scratch/cbgb-sparta-3d/fluor-cell-3d-ne"
cp -n "$CASEDIR"/in.ne_fluor3d_resume "$RUNDIR/"
cd "$RUNDIR"

# mit_quicktest instead of mit_normal: mit_normal had ~1100-1400 jobs queued
# and this same-sized job (6 chunks x 20000 steps) took only 9m32s once it
# actually ran -- comfortably inside quicktest's 15-min cap, and quicktest's
# queue is far shorter (its own separate, higher-priority queue). If a
# future continuation ever DOES run long enough to hit the 15-min limit,
# checkpointing means nothing is lost -- just resume again from whatever
# restart.*.<step> file was last written.
#
# RESTARTFILE must match the process count used at write time (SPARTA
# redistributes across a different count on read, but doing so here failed
# with "ghost cells do not exist" during a mismatched-NP validation test --
# not actually a deck bug, just untested with mismatched NP, so keep NP=40
# consistent with the original run unless deliberately changing it).
RESTARTFILE="${RESTARTFILE:-restart.%.120000}"
NCHUNKS="${NCHUNKS:-6}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-20000}"

mpirun -np "$SLURM_NTASKS" "$HOME/opt/sparta-27Aug2026/bin/spa_mpi" \
  -in in.ne_fluor3d_resume \
  -var RESTARTFILE "$RESTARTFILE" \
  -var MDOT 6.002164e-08 -var FILLN 5.4e21 -var FNUM 1e11 -var DT 1e-7 \
  -var DUMPFREQ 20000 -var CHECKPOINT_STEPS "$CHECKPOINT_STEPS" -var NCHUNKS "$NCHUNKS" \
  &> resume.log
rc=$?
echo "solver exit code: $rc"
tail -30 resume.log
exit "$rc"
