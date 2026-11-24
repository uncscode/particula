#!/bin/bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export PACKET="$(cd "$here/../.." && pwd)"
export PARTICULA_ROOT="$(realpath "${1:?pass the af0dc67bf7 checkout}")"
export ACT_OUT="$(realpath -m "${2:?pass the output directory}")"
export STUDY_PACKET="$PACKET"
export PYTHONPATH="$PARTICULA_ROOT:$PARTICULA_ROOT/packages/particula-aiomfac/src:$PACKET/source:$here"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cloud1833-mpl}"
py="${3:-python3}";mkdir -p "$ACT_OUT";libdir="$(mktemp -d)";export CLOUD_SOLVER="$libdir/activity.so"
trap 'rm -f "$CLOUD_SOLVER"; rmdir "$libdir"' EXIT
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$PACKET/source/chamber.cu" -o "$CLOUD_SOLVER"
cd "$PARTICULA_ROOT"
"$py" "$here/activity.py"
"$py" "$here/activity_summary.py"
"$py" "$here/acidic.py"
"$py" "$here/activity_contrasts.py"
