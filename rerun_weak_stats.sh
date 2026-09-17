#!/usr/bin/env bash
# Boost statistics for the 4 weak-spot tracer runs identified: SCCM=1
# gaussball/uniformball (Fig 3's noisiest points, ~22-30% rel err at N=5000)
# and SCCM=20 gaussball/uniformcell (Fig 4 histogram + Fig 5 heatmap, both
# wanted better sampling). Reuses the already-converted field data.
set -uo pipefail
cd ~/CBGB-SPARTA-2D
JULIA=".local/julia-1.9.4/bin/julia"
ROOT="results/molecules/fig345"
ZMID=0.0265
M_SRF=106.618
SIGMA=3e-18
TEMP=4.0
XOBS=0.06

run_one() {
  local sccm=$1 mode=$2 n=$3 seed=$4 spawnargs=$5
  local fdir="$ROOT/sccm${sccm}/field"
  local tdir="$ROOT/sccm${sccm}/trace_${mode}_big"
  rm -rf "$tdir"; mkdir -p "$tdir"
  echo "$(date -u +%FT%TZ) === tracing sccm${sccm} mode=${mode} n=${n} seed=${seed} ==="
  $JULIA --project=tracer tracer/accumulators/crossing.jl "$tdir" $XOBS \
    "$fdir/cell.surfs" "$fdir/DS2FF.DAT" \
    -n $n --seed $seed -M $M_SRF --sigma $SIGMA -T $TEMP \
    $spawnargs --spawnclip 1 --sampler exact --saveall 1 \
    --spawnout "$tdir/spawn.csv" > "$tdir/run.log" 2>&1
  rc=$?
  echo "$rc" > "$tdir/rc.sentinel"
  if [ "$rc" = "0" ]; then touch "$tdir/DONE"; fi
  tail -6 "$tdir/run.log"
  echo "$(date -u +%FT%TZ) === finished sccm${sccm} ${mode} rc=$rc ==="
}

run_one 1  gaussball    100000 2001 "--spawn gaussball -r 0 -z $ZMID --spawnsize 0.0033"
run_one 1  uniformball  100000 2002 "--spawn uniformball -r 0 -z $ZMID --spawnsize 0.005"
run_one 20 gaussball    50000  2003 "--spawn gaussball -r 0 -z $ZMID --spawnsize 0.0033"
run_one 20 uniformcell  50000  2004 "--spawn uniformcell --spawnrlo 0 --spawnrhi 0.00635 --spawnzlo 0 --spawnzhi 0.053"

echo "$(date -u +%FT%TZ) RERUN_WEAK_STATS_DONE"
