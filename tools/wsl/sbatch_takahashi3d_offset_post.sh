#!/bin/bash
#SBATCH --job-name=tak3d-offset-post
#SBATCH --account=mit_general
#SBATCH --partition=mit_quicktest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.log

# Post-processing for the offset-inlet Takahashi 3D run: the same
# azimuthal (x,r) plot as the on-axis reference (tools/wsl/sbatch_3d_postprocess.sh,
# --frac 0.3 --nr 100), plus a uniform-grid VTK export for PyVista.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64
source .venv/bin/activate

RUNDIR="$HOME/orcd/scratch/cbgb-sparta-3d/takahashi3d-offset"
mkdir -p results/figures results/exports

python tools/plot_field_3d.py "$RUNDIR" \
  --out results/figures/takahashi3d_offset_orcd.png \
  --csv-out results/figures/takahashi3d_offset_orcd_field.csv \
  --frac 0.3 --nr 100

python tools/export_vtk_amr.py "$RUNDIR" --surf "$RUNDIR/cell_takahashi_3d_offset.surf" \
  --out-prefix results/exports/takahashi3d_offset --frac 0.3
