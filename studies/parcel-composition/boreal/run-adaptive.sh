#!/usr/bin/env bash
# Run at the repository root using the Particula development interpreter.
set -euo pipefail
study=studies/parcel-composition/boreal
root=${1:-.artifacts/boreal}
mkdir -p "$root"
export PYTHONPATH=".:${COATED_ENV:?Set COATED_ENV to the scattnlay 2.4 directory}:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export BOREAL_TREE="$root/tree.so"
g++ -O3 -std=c++17 -shared -fPIC "$study/tree.cpp" -o "$BOREAL_TREE"
if [[ ! -f "$root/optics.npz" ]]; then
    python "$study/optics.py" "$root/optics.npz"
fi
if [[ ! -f "$root/response-scattering.npz" ]]; then
    python "$study/response_optics.py" "$root/response-scattering.npz"
fi
python "$study/response_course.py" "$root/response-course" --workers "${WORKERS:-8}"
python "$study/adaptive_trials.py" "$root" "$root/adaptive-grid" --workers "${WORKERS:-8}"
python "$study/adaptive_spectra.py" "$root" "$study/adaptive-spectra.csv.gz" --workers "${WORKERS:-8}"
python "$study/adaptive_steps.py" "$root" "$study" --workers "${WORKERS:-8}"
python "$study/adaptive_results.py" "$study" docs/images/parcel-adaptive-grid.png --cache "$root/adaptive-grid"
