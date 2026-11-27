#!/usr/bin/env bash
# Run from the project root with the development Python and isolated scattnlay.
set -euo pipefail
study=studies/parcel-composition/boreal
root=${1:-.artifacts/boreal}
workers=${WORKERS:-4}
mkdir -p "$root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export PYTHONPATH=".:${COATED_ENV:?Set COATED_ENV to the directory containing scattnlay 2.4}:${PYTHONPATH:-}"
export BOREAL_TREE="$root/tree.so"
g++ -O3 -std=c++17 -shared -fPIC "$study/tree.cpp" -o "$BOREAL_TREE"
table="$root/optics.npz"
if [[ ! -f "$table" ]]; then
    python "$study/optics.py" "$table"
fi
python "$study/stage_two_design.py" "$root/stage-two" --workers "$workers"
python "$study/stage_two.py" "$root/stage-two" --workers "$workers" --optics "$table"
previous=()
if [[ -f "$root/base/1562/c1b1n1/burst-peak.npz" ]]; then
    previous=(--previous "$root/base")
fi
python "$study/stage_two_profiles.py" "$root/stage-two" "$study" --workers "$workers" "${previous[@]}"
python "$study/stage_two_controls.py" "$root/stage-two" "$study/stage-two-controls.csv" --optics "$table"
python "$study/stage_two_results.py" "$root/stage-two" "$study" docs/images/parcel-stage-two.png
