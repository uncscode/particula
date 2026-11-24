#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export PACKET="$(dirname "$here")"
export PARTICULA_ROOT="$(realpath "${1:?particula checkout af0dc67bf7}")"
out="$(realpath -m "${2:?output directory}")"
py="${3:-python3}"
shift 3
export STUDY_PACKET="$PACKET"
export PYTHONPATH="$PACKET/activity/source:$PACKET/source:$PARTICULA_ROOT:$PARTICULA_ROOT/packages/particula-aiomfac/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
lib="$(mktemp -d)"
trap 'rm -f "$lib/cluster.so" "$lib/particles.so"; rmdir "$lib"' EXIT
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$here/clusters.cu" -o "$lib/cluster.so"
g++ -O3 -std=c++17 -fopenmp -shared -fPIC "$here/population.cpp" -o "$lib/particles.so"
"$py" "$here/campaign.py" --out "$out" --lib "$lib" "$@"
if [[ " $* " != *" --pilot "* && " $* " != *" --capture-checks "* ]]; then
    "$py" "$here/plot.py" "$out"
    "$py" "$here/diagnose.py" "$out"
    "$py" "$here/verify_kernel.py" "$lib/cluster.so" "$out/kernel-checks.json"
fi
