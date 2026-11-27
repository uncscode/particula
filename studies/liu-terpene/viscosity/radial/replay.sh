#!/usr/bin/env bash
set -eu
here=$(cd -- "$(dirname -- "$0")" && pwd)
packet=${1:?Supply the analysis directory to read the frozen cyclic fractions and digitized observations}
dst=${2:?Supply a fresh output directory}
: "${PYTHON:?Set PYTHON to the Particula environment interpreter}"
: "${PARTICULA_ROOT:?Set PARTICULA_ROOT to the Particula checkout}"
export PYTHONPATH="$PARTICULA_ROOT"
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/liu-radial-mpl}"
"$PYTHON" "$here/radial.py" pilot "$packet" "$dst/pilot"
"$PYTHON" "$here/radial.py" matrix "$packet" "$dst/results"
"$PYTHON" "$here/radial.py" refine "$packet" "$dst/results"
"$PYTHON" "$here/fractional.py" "$packet" "$dst/results"
"$PYTHON" "$here/condition.py" "$packet" "$dst/results" --training-n 48 --check-n 128
"$PYTHON" "$here/condition.py" "$packet" "$dst/fine" --training-n 128 --check-n 224
"$PYTHON" "$here/identify.py" "$packet" "$dst/results"
"$PYTHON" "$here/layer_refine.py" "$packet" "$dst/results"
"$PYTHON" "$here/controls.py" "$dst/results"
"$PYTHON" "$here/figure.py" "$dst/results" "$dst/radial.png"
