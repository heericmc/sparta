#!/usr/bin/env bash
# Fig.7: median SrF forward velocity (vz) vs. distance from the exit
# aperture, for each of the 7 main-sweep SCCM points. Reuses the already-
# converted He field data under results/molecules/fig345/sccmN/field/.
# Gaussball spawn (same central distribution as Fig.3/4) for consistency.
set -uo pipefail
cd ~/CBGB-SPARTA-2D
JULIA=".local/julia-1.9.4/bin/julia"
FIGROOT="results/molecules/fig345"
OUTROOT="results/molecules/fig7"
mkdir -p "$OUTROOT"

ZMID=0.0265
M_SRF=106.618
SIGMA=3e-18
TEMP=4.0
N=2000
APERTURE_X=0.0535

# distances downstream of the aperture, in meters
DISTANCES="0.0005 0.001 0.002 0.005 0.01 0.03"

RESULTS_CSV="/root/fig7-results.csv"
echo "sccm,distance_m,xobs_m,n_total,n_extracted,median_vz" > "$RESULTS_CSV"

seedctr=5000
for sccm in 1 2 4 8 23 58 127; do
  fdir="$FIGROOT/sccm${sccm}/field"
  if [ ! -f "$fdir/cell.surfs" ]; then
    echo "MISSING_FIELD sccm${sccm}, skipping"
    continue
  fi
  for d in $DISTANCES; do
    xobs=$(python3 -c "print(${APERTURE_X}+${d})")
    tdir="$OUTROOT/sccm${sccm}_d${d}"
    if [ -f "$tdir/DONE" ]; then
      echo "SKIP already done sccm=${sccm} d=${d}"
    else
      rm -rf "$tdir"; mkdir -p "$tdir"
      seedctr=$((seedctr+1))
      echo "$(date -u +%FT%TZ) === sccm=${sccm} d=${d} xobs=${xobs} seed=${seedctr} ==="
      $JULIA --project=tracer tracer/accumulators/crossing.jl "$tdir" "$xobs" \
        "$fdir/cell.surfs" "$fdir/DS2FF.DAT" \
        -n $N --seed $seedctr -M $M_SRF --sigma $SIGMA -T $TEMP \
        --spawn gaussball -r 0 -z $ZMID --spawnsize 0.0033 \
        --spawnclip 1 --sampler exact --saveall 1 \
        --spawnout "$tdir/spawn.csv" > "$tdir/run.log" 2>&1
      rc=$?
      echo "$rc" > "$tdir/rc.sentinel"
      if [ "$rc" = "0" ]; then touch "$tdir/DONE"; fi
      tail -4 "$tdir/run.log"
    fi

    # extract median vz over extracted (have=1) particles, append to results csv
    python3 - "$tdir/recs.csv" "$sccm" "$d" "$xobs" "$RESULTS_CSV" <<'PYEOF'
import csv, sys, statistics
recs_path, sccm, d, xobs, out_path = sys.argv[1:6]
vz = []
n_total = 0
try:
    with open(recs_path, newline="") as f:
        for row in csv.DictReader(f):
            n_total += 1
            if row["have"] == "1":
                vz.append(float(row["vz"]))
except FileNotFoundError:
    pass
median_vz = statistics.median(vz) if vz else ""
with open(out_path, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([sccm, d, xobs, n_total, len(vz), median_vz])
print(f"  sccm={sccm} d={d} n_extracted={len(vz)} median_vz={median_vz}")
PYEOF
  done
done

echo "$(date -u +%FT%TZ) FIG7_ALL_DONE"
