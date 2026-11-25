#!/usr/bin/env bash
set -eu
base=$(cd "$(dirname "$0")" && pwd)
tree=$(cd "${1:?pinned Particula checkout required}" && pwd)
shift
test "$(git -C "$tree" rev-parse HEAD)" = db804546ce7af9443f3aafd7e4b3a58b187253ac
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$tree" OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
cd "$tree"
python=${LUND_PYTHON:-python}
"$python" -c 'import particula,sys; from pathlib import Path; assert Path(particula.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())' "$tree"
exec "$python" "$base/campaign.py" "$@"
