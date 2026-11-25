#!/usr/bin/env bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
packet=$(cd "$here/.." && pwd)
tree=$(cd "${1:?Particula checkout required}" && pwd)
out=${2:?empty output directory required}
mkdir -p "$out"
out=$(cd "$out" && pwd)
if [[ -n "$(ls -A "$out")" ]]; then echo "Choose an empty output directory." >&2; exit 1; fi
test "$(git -C "$tree" rev-parse HEAD)" = db804546ce7af9443f3aafd7e4b3a58b187253ac
export PYTHONPATH="$tree" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
python=${LUND_PYTHON:-python}
cd "$tree"
"$python" -c 'import particula,sys; from pathlib import Path; assert Path(particula.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())' "$tree"
"$python" "$here/controls.py" "$out"
for stage in pilot main checks plates source tracks refine environment ventilation fine_controls; do
  "$python" "$here/cloud.py" "$packet" "$out" --stage "$stage"
done
"$python" "$here/analyze.py" "$out"
