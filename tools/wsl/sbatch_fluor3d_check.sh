#!/bin/bash
#SBATCH --job-name=fluor3d-check
#SBATCH --account=mit_general
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.log

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/12.2.0 openmpi/4.1.4

RUNDIR="$HOME/orcd/scratch/cbgb-sparta-3d/fluor-cell-3d-ne"
cd "$RUNDIR"

mpirun -np "$SLURM_NTASKS" "$HOME/opt/sparta-27Aug2026/bin/spa_mpi" \
  -in in.ne_fluor3d_scoping \
  -var MDOT 6.002164e-08 -var FILLN 5.4e21 -var FNUM 2.5e17 \
  -var DT 1e-7 -var STEPS 2000 \
  &> check.log
rc=$?
echo "solver exit code: $rc"
tail -30 check.log
exit "$rc"
