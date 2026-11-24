#!/usr/bin/env bash
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
export PACKET="$(dirname "$here")" STUDY_PACKET="$(dirname "$here")"
export PYTHONPATH="$PACKET/eddy-twenty:$PACKET/five-nm-flux:$PACKET/activity/source:$PACKET/source:${PARTICULA_ROOT:?frozen Particula checkout}:$PARTICULA_ROOT/packages/particula-aiomfac/src"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
tmp=$(mktemp -d)
trap 'rm -f "$tmp/chambers.so"; rmdir "$tmp"' EXIT
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$PACKET/eddy-twenty/chambers.cu" -o "$tmp/chambers.so"
"${PY:-python3}" -u "$here/campaign.py" "$tmp/chambers.so" "$@"
