#!/usr/bin/env bash
# Run from the checkout root with the development Python and isolated scattnlay.
set -euo pipefail
study=studies/parcel-composition/boreal
base=${1:-.artifacts/boreal}
root="$base/redraw-qualification"
mkdir -p "$root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export PYTHONPATH=".:${COATED_ENV:?Set COATED_ENV to the isolated scattnlay 2.4 directory}:${PYTHONPATH:-}"
export BOREAL_TREE="$base/tree.so"
g++ -O3 -std=c++17 -shared -fPIC "$study/tree.cpp" -o "$BOREAL_TREE"
table="$base/optics.npz"
if [[ ! -f "$table" ]]; then
    python "$study/optics.py" "$table"
fi
workers=${WORKERS:-6}
python "$study/redraw_design.py" "$root" --workers "$workers"
python "$study/redraw_run.py" "$root" train --optics "$table" --workers "$workers"
python "$study/redraw_results.py" "$root" train
python "$study/redraw_run.py" "$root" test --selection "$study/redraw-selection.json" --optics "$table" --workers "$workers"
python "$study/redraw_fine.py" "$root" "$study/redraw-selection.json" --optics "$table" --workers "$workers"
python "$study/redraw_results.py" "$root" test --optics "$table"
python "$study/redraw_audit.py" "$root" "$study/redraw-selection.json" "$study/redraw-storage.csv" --workers "$workers"
python "$study/redraw_comparators.py" "$root" "$study" --optics "$table" --workers "$workers"
