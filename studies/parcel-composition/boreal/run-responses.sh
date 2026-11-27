#!/usr/bin/env bash
# Run at the checkout root with the Particula development interpreter active.
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
python "$study/response_optics.py" "$root/response-scattering.npz"
python "$study/response_course.py" "$root/response-course" --workers "${WORKERS:-8}"
python "$study/response_optics_check.py" "$root" "$study/response-scattering-check.csv"
python "$study/response_pairs.py" "$root" "$root/response-pairs" --workers "${WORKERS:-8}"
python "$study/response_step_check.py" "$root" "$study/response-step-check.csv" --workers "${WORKERS:-8}"
python "$study/response_results.py" "$root/response-pairs" "$study" docs/images/parcel-response-qualification.png
