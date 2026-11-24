#!/usr/bin/env bash
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
export PACKET="$(dirname "$here")" STUDY_PACKET="$(dirname "$here")"
export PARTICULA_ROOT="${PARTICULA_ROOT:?supply checkout at af0dc67bf7}"
export PYTHONPATH="$PACKET/five-nm-flux:$PACKET/activity/source:$PACKET/source:$PARTICULA_ROOT:$PARTICULA_ROOT/packages/particula-aiomfac/src"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
tmp="$(mktemp -d)"
trap 'rm -f "$tmp/chambers.so"; rmdir "$tmp"' EXIT
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$here/chambers.cu" -o "$tmp/chambers.so"
"${PY:-python3}" -u "$here/experiment.py" "$tmp/chambers.so" "$@"
