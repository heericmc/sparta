#!/usr/bin/env bash
# Fig 3/4/5 reproduction: trace SrF molecules (M=106.618u, sigma=3e-18 m^2,
# spawned thermalized at 4K with zero mean velocity) through 8 converged He
# buffer-gas fields (SCCM 1,2,4,8,20,23,58,127), using 3 spawn distributions
# per Takahashi et al.: gaussball (sigma=0.33cm), uniformball (1cm dia),
# uniformcell (whole cell). Fig 3 = extraction fraction vs SCCM per
# distribution; Fig 4 = pumpout-time histogram from the 20 SCCM gaussball
# run; Fig 5 = position-binned heatmap from the 20 SCCM uniformcell run
# (via --spawnout). XOBS=0.06m (just past the 0.0535m aperture exit).
set -uo pipefail
cd ~/CBGB-SPARTA-2D
source .venv/bin/activate
JULIA=".local/julia-1.9.4/bin/julia"
OUTROOT="results/molecules/fig345"
mkdir -p "$OUTROOT"

declare -A RUNDIR=( [1]=results/he/takahashi-fig2-sccm1-conv [2]=results/he/takahashi-fig2-sccm2-conv \
  [4]=results/he/takahashi-fig2-sccm4-conv [8]=results/he/takahashi-fig2-sccm8-conv \
  [20]=results/he/takahashi-longrun [23]=results/he/takahashi-fig2-sccm23-conv \
  [58]=results/he/takahashi-fig2-sccm58-conv [127]=results/he/takahashi-fig2-sccm127-conv )

N=5000
XOBS=0.06
M_SRF=106.618
SIGMA=3e-18
TEMP=4.0
ZMID=0.0265

seedctr=1000
for sccm in 1 2 4 8 20 23 58 127; do
  rundir="${RUNDIR[$sccm]}"
  fdir="$OUTROOT/sccm${sccm}/field"
  if [ ! -f "$fdir/cell.surfs" ]; then
    mkdir -p "$fdir"
    echo "$(date -u +%FT%TZ) converting field for sccm${sccm} from $rundir"
    python3 tools/field2tracer.py "$rundir" --out "$fdir" || { echo "CONVERT_FAILED sccm${sccm}"; continue; }
  fi
  for mode in gaussball uniformball uniformcell; do
    tdir="$OUTROOT/sccm${sccm}/trace_${mode}"
    if [ -f "$tdir/DONE" ]; then
      echo "$(date -u +%FT%TZ) sccm${sccm} ${mode} already done, skipping"
      continue
    fi
    rm -rf "$tdir"; mkdir -p "$tdir"
    seedctr=$((seedctr+1))
    echo "$(date -u +%FT%TZ) === tracing sccm${sccm} mode=${mode} seed=${seedctr} ==="
    case $mode in
      gaussball)
        spawnargs="--spawn gaussball -r 0 -z $ZMID --spawnsize 0.0033" ;;
      uniformball)
        spawnargs="--spawn uniformball -r 0 -z $ZMID --spawnsize 0.005" ;;
      uniformcell)
        spawnargs="--spawn uniformcell --spawnrlo 0 --spawnrhi 0.00635 --spawnzlo 0 --spawnzhi 0.053" ;;
    esac
    $JULIA --project=tracer tracer/accumulators/crossing.jl "$tdir" $XOBS \
      "$fdir/cell.surfs" "$fdir/DS2FF.DAT" \
      -n $N --seed $seedctr -M $M_SRF --sigma $SIGMA -T $TEMP \
      $spawnargs --spawnclip 1 --sampler exact --saveall 1 \
      --spawnout "$tdir/spawn.csv" > "$tdir/run.log" 2>&1
    rc=$?
    echo "$rc" > "$tdir/rc.sentinel"
    if [ "$rc" = "0" ]; then touch "$tdir/DONE"; fi
    tail -5 "$tdir/run.log"
    echo "$(date -u +%FT%TZ) === finished sccm${sccm} ${mode} rc=$rc ==="
  done
done
echo "$(date -u +%FT%TZ) FIG345_ALL_DONE"
