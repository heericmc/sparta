#!/usr/bin/env bash
# Boost statistics for the two sparsest Fig7 curves: SCCM=1 (extraction ~0.4%,
# only 3-7 extracted/distance at N=2000) and SCCM=2 (~2-6% extraction, ~27-44
# extracted/distance). Reruns all 6 distances at N=30000 (15x), then merges
# the improved median_vz values into /root/fig7-results.csv in place.
set -uo pipefail
cd ~/CBGB-SPARTA-2D
JULIA=".local/julia-1.9.4/bin/julia"
FIGROOT="results/molecules/fig345"
OUTROOT="results/molecules/fig7"

ZMID=0.0265
M_SRF=106.618
SIGMA=3e-18
TEMP=4.0
N=30000
APERTURE_X=0.0535
DISTANCES="0.0005 0.001 0.002 0.005 0.01 0.03"

seedctr=6000
for sccm in 1 2; do
  fdir="$FIGROOT/sccm${sccm}/field"
  for d in $DISTANCES; do
    xobs=$(python3 -c "print(${APERTURE_X}+${d})")
    tdir="$OUTROOT/sccm${sccm}_d${d}_big"
    if [ -f "$tdir/DONE" ]; then
      echo "SKIP already done sccm=${sccm} d=${d}"
    else
      rm -rf "$tdir"; mkdir -p "$tdir"
      seedctr=$((seedctr+1))
      echo "$(date -u +%FT%TZ) === BOOST sccm=${sccm} d=${d} xobs=${xobs} seed=${seedctr} ==="
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
  done
done

python3 - <<'PYEOF'
import csv, statistics, os

ROOT = os.path.expanduser("~/CBGB-SPARTA-2D/results/molecules/fig7")
DISTANCES = ["0.0005", "0.001", "0.002", "0.005", "0.01", "0.03"]
APERTURE_X = 0.0535

boosted = {}
for sccm in [1, 2]:
    for d in DISTANCES:
        tdir = os.path.join(ROOT, f"sccm{sccm}_d{d}_big")
        recs_path = os.path.join(tdir, "recs.csv")
        vz = []
        n_total = 0
        try:
            with open(recs_path, newline="") as f:
                for row in csv.DictReader(f):
                    n_total += 1
                    if row["have"] == "1":
                        vz.append(float(row["vz"]))
        except FileNotFoundError:
            continue
        median_vz = statistics.median(vz) if vz else ""
        boosted[(sccm, d)] = (n_total, len(vz), median_vz)
        print(f"  BOOSTED sccm={sccm} d={d}: n_extracted={len(vz)} median_vz={median_vz}")

path = "/root/fig7-results.csv"
rows = []
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

for row in rows:
    key = (int(float(row["sccm"])), row["distance_m"])
    if key in boosted:
        n_total, n_extracted, median_vz = boosted[key]
        row["n_total"] = n_total
        row["n_extracted"] = n_extracted
        row["median_vz"] = median_vz

with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print("merged boosted rows into", path)
PYEOF

echo "$(date -u +%FT%TZ) FIG7_BOOST_DONE"
