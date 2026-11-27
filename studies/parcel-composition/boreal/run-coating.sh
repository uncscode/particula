#!/usr/bin/env bash
# The stage-two reference trajectories and qualified optics table are inputs.
set -euo pipefail
root=${1:-.artifacts/boreal}
study=studies/parcel-composition/boreal
work="$root/coating"
export PYTHONPATH=".:${COATED_ENV:?Set COATED_ENV to the isolated scattnlay directory}:${PYTHONPATH:-}"
export BOREAL_TREE="$root/tree.so" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
common=("$root/stage-two" "$work" --optics "$root/optics.npz" --workers "${WORKERS:-4}")
python "$study/coating_replay.py" "${common[@]}" --scope six --methods baseline-0
python "$study/coating_attribution.py" "$work" "$study"
target=(--target-file "$study/coating-targets.csv")
python "$study/coating_replay.py" "${common[@]}" "${target[@]}" --scope six --methods shell-8 local-4 local-8 local-16 outside-8 exact-0 repeat-1 repeat-2 repeat-3 paired-1 paired-2 paired-3 paired4-1 paired4-2 paired4-3
python "$study/coating_replay.py" "${common[@]}" "${target[@]}" --scope others --methods baseline-0 local-4 local-8 shell-8
python "$study/coating_replay.py" "${common[@]}" "${target[@]}" --scope controls --methods local-4 local-8
python "$study/coating_results.py" "$work" "$study" docs/images/parcel-coating-restarts.png
