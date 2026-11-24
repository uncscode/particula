#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export PACKET="$(dirname "$here")"
export PARTICULA_ROOT="$(realpath "${1:?particula checkout af0dc67bf7}")"
export RESULTS="$(realpath -m "${2:?output directory}")"
export STUDY_PACKET="$PACKET"
export PYTHONPATH="$PACKET/activity/source:$PACKET/source:$PARTICULA_ROOT:$PARTICULA_ROOT/packages/particula-aiomfac/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$RESULTS"
lib="$(mktemp -d)"
export CLOUD_SOLVER="$lib/chamber.so"
trap 'rm -f "$CLOUD_SOLVER"; rmdir "$lib"' EXIT
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$PACKET/source/chamber.cu" -o "$CLOUD_SOLVER"
"${3:-python3}" "$here/entry_timing.py"
"${3:-python3}" "$here/plot.py" "$RESULTS"
