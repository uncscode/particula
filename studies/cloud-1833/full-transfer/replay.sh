#!/usr/bin/env bash
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
export STUDY_PACKET="$(dirname "$here")"
export PARTICULA_ROOT="$(realpath "${1:?Particula worktree}")"
export PYTHONPATH="$PARTICULA_ROOT:$STUDY_PACKET/source"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py="${2:?Python with study dependencies}"
output="${3:?Output directory}"
build_dir="$(mktemp -d)"
export CLOUD_SOLVER="$build_dir/chamber.so"
trap 'rm -f "$CLOUD_SOLVER"; rmdir "$build_dir"' EXIT
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$STUDY_PACKET/source/chamber.cu" -o "$CLOUD_SOLVER"
"$py" "$here/simulate_frozen.py" "$output"
