#!/usr/bin/env bash
# Reproduce into an empty output directory. Usage: bash replay.sh SOURCE_TREE OUTPUT_DIR
set -eu
packet=$(cd "$(dirname "$0")" && pwd)
tree=$(cd "${1:?Particula source checkout required}" && pwd)
out=${2:?output directory required}
mkdir -p "$out"
out=$(cd "$out" && pwd)
if [[ -n "$(ls -A "$out")" ]]; then echo 'Choose an empty output directory.' >&2; exit 1; fi
python=${LUND_PYTHON:-python}
export PYTHONPATH="$tree" OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
if [[ "$(git -C "$tree" rev-parse HEAD)" != db804546ce7af9443f3aafd7e4b3a58b187253ac ]]; then
  echo 'Select the recorded Particula revision.' >&2; exit 1
fi
cd "$tree"
"$python" -c 'import particula,sys; from pathlib import Path; assert Path(particula.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())' "$tree"
"$python" "$packet/study.py" "$packet" "$out" --pilot
for script in study challenges landmark uncertainty duration refinement; do
  "$python" "$packet/$script.py" "$packet" "$out"
done
"$python" "$packet/boxes.py" "$packet" "$out" --pilot
"$python" "$packet/boxes.py" "$packet" "$out"
"$python" "$packet/figures.py" "$out"
