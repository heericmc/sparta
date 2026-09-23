#!/bin/bash
#SBATCH --job-name=tak3d-offset
#SBATCH --account=mit_general
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.log

# Offset-inlet Takahashi 3D cell (cases/takahashi-3d-offset): identical
# physics/grid/parameters to the on-axis reference (tools/wsl/sbatch_3d_run.sh)
# except the inlet tube axis is shifted 3mm in y. One 200000-step chunk per
# job (~8 min on 40 cores for the reference geometry), so each fits
# mit_quicktest's 15-min cap; pass RESTARTFILE=restart.%.<step> to resume.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load gcc/12.2.0 openmpi/4.1.4

CASEDIR=cases/takahashi-3d-offset
RUNDIR="$HOME/orcd/scratch/cbgb-sparta-3d/takahashi3d-offset"
mkdir -p "$RUNDIR"
cp -n "$CASEDIR"/in.he_takahashi_3d_offset_run "$CASEDIR"/in.he_takahashi_3d_offset_resume \
      "$CASEDIR"/cell_takahashi_3d_offset.surf "$CASEDIR"/he.species "$CASEDIR"/he.vss "$RUNDIR/"
cd "$RUNDIR"

COMMON="-var MDOT 5.952587e-08 -var FILLN 1.1719e22 -var FNUM 5.0e11 -var DT 1e-7 \
        -var DUMPFREQ 10000 -var CHECKPOINT_STEPS 200000 -var NCHUNKS 1"
if [ -n "${RESTARTFILE:-}" ]; then
  mpirun -np "$SLURM_NTASKS" "$HOME/opt/sparta-27Aug2026/bin/spa_mpi" \
    -in in.he_takahashi_3d_offset_resume -var RESTARTFILE "$RESTARTFILE" $COMMON &>> run.log
else
  mpirun -np "$SLURM_NTASKS" "$HOME/opt/sparta-27Aug2026/bin/spa_mpi" \
    -in in.he_takahashi_3d_offset_run $COMMON &> run.log
fi
rc=$?
echo "solver exit code: $rc"
tail -5 run.log
exit "$rc"
