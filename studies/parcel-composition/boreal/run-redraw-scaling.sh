#!/usr/bin/env bash
# Run from the checkout root, with the Particula development Python activated.
set -euo pipefail
study=studies/parcel-composition/boreal
base=${1:-.artifacts/boreal}
root="$base/redraw-scaling"
mkdir -p "$root"
export PYTHONPATH=".:${COATED_ENV:?Set COATED_ENV to the scattnlay 2.4 directory}:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export BOREAL_TREE="$base/tree.so"
g++ -O3 -std=c++17 -shared -fPIC "$study/tree.cpp" -o "$BOREAL_TREE"
if [[ ! -f "$base/optics.npz" ]]; then
    python "$study/optics.py" "$base/optics.npz"
fi
python "$study/redraw_scaling.py" "$root" --optics "$base/optics.npz" --workers "${WORKERS:-6}"
python "$study/redraw_burst.py" "$root" --optics "$base/optics.npz" --workers "${WORKERS:-6}"
python "$study/redraw_memory.py" "$root" "$study/redraw-memory.csv"
python "$study/redraw_scale_results.py" "$root" "$study" docs/images --optics "$base/optics.npz"
