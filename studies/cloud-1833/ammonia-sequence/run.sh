#!/usr/bin/env bash
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
export PACKET="$(dirname "$here")" STUDY_PACKET="$(dirname "$here")"
export PARTICULA_ROOT="${PARTICULA_ROOT:?set to the frozen Particula checkout}"
export PYTHONPATH="$here:$PACKET/charge-evaporation:$PACKET/activity/source:$PACKET/source:$PARTICULA_ROOT:$PARTICULA_ROOT/packages/particula-aiomfac/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
"${PY:-python3}" "$here/simulate.py" "$1" "$2" "$3"
