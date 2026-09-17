#!/usr/bin/env bash
set -uo pipefail
cd ~/CBGB-SPARTA-2D
source .venv/bin/activate
export SPARTA_PLAIN_EXE="$HOME/opt/sparta-27Aug2026/bin/spa_mpi"
export DSMC_RUN_ROOT="$PWD/results/he"
export DSMC_LOG_ROOT="$PWD/results/logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

declare -A MDOT=( [1]=2.976293e-09 [2]=5.952587e-09 [4]=1.190517e-08 [8]=2.381035e-08 [23]=6.845475e-08 [58]=1.726250e-07 [127]=3.779893e-07 )
declare -A FILLN=( [1]=5.8596e+20 [2]=1.1719e+21 [4]=2.3438e+21 [8]=4.6877e+21 [23]=1.3477e+22 [58]=3.3986e+22 [127]=7.4417e+22 )
declare -A FNUM=( [1]=5.000e+16 [2]=1.000e+17 [4]=2.000e+17 [8]=4.000e+17 [23]=1.150e+18 [58]=2.900e+18 [127]=6.350e+18 )

for sccm in 1 2 4 8 23 58 127; do
  name="takahashi-fig2-sccm${sccm}"
  rundir="$DSMC_RUN_ROOT/$name"
  if [ -f "$rundir/rc.sentinel" ] && [ "$(cat "$rundir/rc.sentinel")" = "0" ]; then
    echo "$(date -u +%FT%TZ) $name already complete, skipping"
    continue
  fi
  rm -rf "$rundir"
  echo "$(date -u +%FT%TZ) === starting $name (MDOT=${MDOT[$sccm]} FILLN=${FILLN[$sccm]} FNUM=${FNUM[$sccm]}) ==="
  bash cases/b5-lean/run_lean.sh "$name" 10 in.he_takahashi_mflow \
    "MDOT=${MDOT[$sccm]},FILLN=${FILLN[$sccm]},FNUM=${FNUM[$sccm]},DT=1e-7,STEPS=120000,WEIGHT=radius,SEED=8675309" \
    plain takahashi-single in.he_takahashi_mflow cell_takahashi.surf \
    tube_fwd.surf tube_rev.surf ap_fwd.surf ap_rev.surf
  echo "$(date -u +%FT%TZ) === finished $name rc=$? ==="
  sync && echo 1 > /proc/sys/vm/drop_caches 2>/dev/null
  echo "$(date -u +%FT%TZ) dropped page cache; $(free -h | awk 'NR==2{print \"buff/cache=\"$6\" available=\"$7}')"
done
echo "$(date -u +%FT%TZ) SWEEP_ALL_DONE"
