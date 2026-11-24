#!/bin/bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export PACKET="$(cd "$here/../.." && pwd)"
export PARTICULA_ROOT="$(realpath "${1:?pass the af0dc67bf7 checkout}")"
export INV_OUT="$(realpath -m "${2:?pass the output directory}")"
export STUDY_PACKET="$PACKET"
export PYTHONPATH="$PARTICULA_ROOT:$PARTICULA_ROOT/packages/particula-aiomfac/src:$PACKET/source:$PACKET/activity/source:$here"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cloud1833-mpl}"
py="${3:-python3}";mkdir -p "$INV_OUT";libdir="$(mktemp -d)";export INVERSE_SOLVER="$libdir/inverse.so"
export AUDIT_SOLVER="$libdir/audit.so"
export MULTI_SOLVER="$libdir/multi.so"
trap 'rm -f "$MULTI_SOLVER" "$AUDIT_SOLVER" "$INVERSE_SOLVER"; rmdir "$libdir"' EXIT
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$here/inverse.cu" -o "$INVERSE_SOLVER"
nvcc -O3 -DAUDIT "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$here/inverse.cu" -o "$AUDIT_SOLVER"
nvcc -O3 -DMULTIBOX "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$here/inverse.cu" -o "$MULTI_SOLVER"
cd "$PARTICULA_ROOT"
"$py" "$here/fit.py" pilot
"$py" "$here/fit.py" fit
"$py" "$here/refine.py"
"$py" "$here/acidfit.py"
"$py" "$here/transfer.py"
"$py" "$here/check_batch.py"
"$py" "$here/audit.py"
"$py" "$here/recovery.py"
"$py" "$here/ablate.py"
bash "$here/cluster-replay.sh" "$py"
"$py" "$here/summary.py"
"$py" "$here/inventory.py"
