#!/usr/bin/env bash
set -eu
here=$(cd -- "$(dirname -- "$0")" && pwd)
dst=${1:?Supply an empty output directory}
: "${PARTICULA_ROOT:?Set PARTICULA_ROOT to the Particula checkout}"
: "${PYTHON:?Set PYTHON to the Particula environment interpreter}"
export PYTHONPATH="$PARTICULA_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/liu-parcel-mpl}"
survival="$here/../model/results/ageing.csv"
"$PYTHON" "$here/parcel.py" pilot "$dst/pilot-cpu" --device cpu
"$PYTHON" "$here/parcel.py" pilot "$dst/pilot-gpu" --device cuda:0
"$PYTHON" "$here/parcel.py" matrix "$dst/results" --survival "$survival"
"$PYTHON" "$here/parcel.py" refine "$dst/refinement" --survival "$survival"
"$PYTHON" "$here/feedback.py" "$dst/feedback" --survival "$survival"
"$PYTHON" "$here/tree_sensitivity.py" "$dst/plant" --survival "$survival"
"$PYTHON" "$here/size_limit.py" "$dst/size-independent" --survival "$survival"
"$PYTHON" "$here/cutoff.py" "$dst" --survival "$survival"
"$PYTHON" "$here/fit_envelope.py" "$dst" --survival "$survival"
"$PYTHON" "$here/graphics.py" "$dst"
