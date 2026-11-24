#!/usr/bin/env bash
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
export PACKET="$(dirname "$here")" STUDY_PACKET="$(dirname "$here")"
export PARTICULA_ROOT="$(realpath "${1:?frozen Particula checkout}")"
out="$(realpath -m "${2:?output directory}")"
py="${3:?python executable}"
shift 3
export PYTHONPATH="$PACKET/ammonia-sequence:$PACKET/charge-evaporation:$PACKET/activity/source:$PACKET/source:$PARTICULA_ROOT:$PARTICULA_ROOT/packages/particula-aiomfac/src"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
mkdir -p "$out"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$here/chambers.cu" -o "$tmp/chamber.so"
if [[ " $* " != *" --pilot "* ]]; then
    nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$PACKET/ammonia-sequence/growth.cu" -o "$tmp/growth.so"
    g++ -O3 -std=c++17 -fopenmp -shared -fPIC "$here/passages.cpp" -o "$tmp/particles.so"
    "$py" "$here/observe.py" "$tmp" "$out" "$tmp/prior"
fi
"$py" "$here/campaign.py" "$out" "$tmp/chamber.so" "$@"
