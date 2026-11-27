#!/usr/bin/env bash
# Run from the repository root with the Particula development interpreter active.
set -euo pipefail
study=studies/parcel-composition/boreal
root=${1:-.artifacts/boreal}
workers=${WORKERS:-4}
mkdir -p "$root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export BOREAL_TREE="$root/tree.so"
export PYTHONPATH=".:${COATED_ENV:?Set COATED_ENV to the directory containing scattnlay 2.4}:${PYTHONPATH:-}"
g++ -O3 -std=c++17 -shared -fPIC "$study/tree.cpp" -o "$BOREAL_TREE"
table="$root/optics.npz"
python "$study/optics.py" "$table"
python studies/parcel-composition/aging/run.py "$root/four-control" --seeds 1199 --limit 16
python "$study/campaign.py" "$root/base" --workers "$workers"
python "$study/events.py" "$root/base" --workers "$workers"
python "$study/qualify.py" "$root/base" "$table"
python "$study/snapshot.py" "$root/base" "$root/scores" --workers "$workers" --optics "$table" --update-events
python "$study/restart.py" "$root/base" "$root/restarts" --workers "$workers" --optics "$table"
python "$study/restart.py" "$root/base" "$root/identity" --optics "$table" --cap 0 --seeds 1199 --regimes c1b1n1
python "$study/selection.py" "$root/scores" "$study/caps.csv" --base "$root/base" --restarts "$root/restarts" --workers "$workers" --optics "$table"
cloud=(c1b0n0 c1b1n0 c1b0n1 c1b1n1 coupled_dark)
held=(1199 1320 1441 1562)
for cap in 128 256; do
    python "$study/restart.py" "$root/base" "$root/restarts" --workers "$workers" --cap "$cap" --regimes "${cloud[@]}" --optics "$table"
done
python "$study/cloud_screen.py" "$root/base" "$study"
python "$study/cloud_water.py" "$root/base" "$study/water-control.csv"
python "$study/campaign.py" "$root/repeat" --repeat 1 --seeds "${held[@]}" --workers "$workers"
python "$study/restart.py" "$root/repeat" "$root/paired" --seeds "${held[@]}" --workers "$workers" --optics "$table"
python "$study/campaign.py" "$root/double" --particles 200000 --seeds "${held[@]}" --regimes c1b1n1 --workers "$workers"
for pair in fine:150 finer:75 finest:30; do
    python "$study/campaign.py" "$root/${pair%:*}" --dt "${pair#*:}" --seeds "${held[@]}" --regimes c1b1n1 --workers "$workers"
done
python "$study/campaign.py" "$root/15sec" --dt 15 --seeds 1562 1441 --regimes c1b1n1 --workers "$workers"
python "$study/restart.py" "$root/finer" "$root/restart75" --seeds "${held[@]}" --regimes c1b1n1 --optics "$table"
python "$study/campaign.py" "$root/5sec" --dt 5 --seeds 1562 --regimes c1b1n1 --workers 1
python "$study/restart.py" "$root/15sec" "$root/restart15" --seeds 1441 1562 --regimes c1b1n1 --optics "$table"
python "$study/restart.py" "$root/double" "$root/restartdouble" --seeds "${held[@]}" --regimes c1b1n1 --optics "$table"
python "$study/forecast_minimum.py" "$root" "$study" --workers "$workers" --optics "$table"
python "$study/restart.py" "$root/base" "$root/restarts" --cap 20 --regimes c0b0n0 c0b1n0 --workers "$workers" --optics "$table"
python "$study/uncertainty.py" "$root" "$study/reference-checks.csv" --optics "$table"
python "$study/allocation.py" "$root/base" "$root/allocation" "$study" --workers "$workers" --optics "$table"
python "$study/allocated_restart.py" "$root" "$study" --workers "$workers" --optics "$table"
python "$study/shape.py" "$root/scores" "$study"
python "$study/figure.py" "$root/base" "$study" docs/images/parcel-boreal-cloud-error.png
python "$study/checkpoint_bytes.py" "$study/cloudy-restart.npz" "$study/checkpoint-bytes.csv"
python "$study/shape_noise.py" "$root" "$study/chi-uncertainty.csv"
python "$study/activation_feature.py" "$root/base" "$study/activation-features.csv"
python "$study/new_mode.py" "$root/base" "$study/new-mode.csv" --optics "$table"
python "$study/snapshot.py" "$root/repeat" "$root/snaprepeat" --seeds "${held[@]}" --grids 10 16 --rules 2 --limit 32 --workers "$workers" --optics "$table"
python "$study/snapshot.py" "$root/double" "$root/snapdouble" --seeds "${held[@]}" --regimes c1b1n1 --grids 10 16 --rules 2 --limit 32 --workers "$workers" --optics "$table"
python "$study/snapshot.py" "$root/15sec" "$root/snap15sec" --seeds 1441 1562 --regimes c1b1n1 --grids 10 16 --rules 2 --limit 32 --workers "$workers" --optics "$table"
python "$study/robustness.py" "$root" "$study"
python "$study/limiting_cases.py" "$root" "$study/limiting-cases.csv" --optics "$table"
python "$study/cloud_classes.py" "$root/base" "$study/cloud-classes.csv"
python "$study/cloud_classes.py" "$root/15sec" "$study/cloud-classes-refined.csv" --seeds 1562 --treatments 'class representatives'
python "$study/knot_boundary.py" "$study/knot-boundary.csv"
python "$study/results.py" "$root" "$study"
