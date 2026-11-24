#!/usr/bin/env bash
# Run every scientific comparison and regenerate the report figures.
set -euo pipefail
export PARTICULA_ROOT="$(realpath "${1:?Particula worktree required}")"
export CLOUD_INPUTS="$(realpath "${2:?directory containing the author ZIP required}")"
mkdir -p "${3:?output directory required}"
export STUDY_PACKET="$(realpath "$3")"
py="${4:-python3}"
src="$(cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPATH="$PARTICULA_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cloud1833-mpl}"
export OPENBLAS_NUM_THREADS=1
build_dir="$(mktemp -d)"
trap 'rm -f "$build_dir/chamber.so"; rmdir "$build_dir"' EXIT
export CLOUD_SOLVER="$build_dir/chamber.so"
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$src/chamber.cu" -o "$CLOUD_SOLVER"
cd "$PARTICULA_ROOT"
"$py" "$src/study.py" pilot
"$py" "$src/study.py" production
"$py" "$src/study.py" controls
"$py" "$src/gas_replay.py"
"$py" "$src/mechanism.py"
"$py" "$src/dark_decay.py"
"$py" "$src/replot.py" "$STUDY_PACKET"
