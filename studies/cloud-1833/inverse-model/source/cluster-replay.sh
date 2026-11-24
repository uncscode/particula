#!/bin/bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PACKET/activity/source:$here:${PYTHONPATH:-}"
py="${1:-python3}"
dir="$(mktemp -d)"
trap 'rm -f "$dir"/cluster*.so; rmdir "$dir"' EXIT
for cells in 6 12 24; do
    export CLUSTER_CELLS="$cells"
    export INVERSE_SOLVER="$dir/cluster$cells.so"
    nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" "-DCLUSTER_CELLS=$cells" --shared -Xcompiler -fPIC "$here/inverse.cu" -o "$INVERSE_SOLVER"
    "$py" "$here/cluster_refine.py"
done
