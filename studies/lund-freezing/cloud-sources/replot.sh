#!/usr/bin/env bash
set -eu
base=$(cd "$(dirname "$0")" && pwd)
tree=$(cd "${1:?pinned Particula checkout required}" && pwd)
data=$(cd "${2:-$base/results}" && pwd)
ledger="$data/runs.jsonl.gz"
if [[ ! -f "$ledger" ]]; then ledger="$data/runs.jsonl"; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$tree" OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
python=${LUND_PYTHON:-python}
cd "$tree"
test "$(git rev-parse HEAD)" = db804546ce7af9443f3aafd7e4b3a58b187253ac
"$python" "$base/check_inputs.py" "$data"
"$python" -c 'import sys; sys.path.insert(0,sys.argv[1]); from loading import loadings; loadings().to_csv(sys.argv[2])' "$base" "$data/air-volume.csv"
"$python" "$base/summarize.py" "$ledger" "$data"
"$python" "$base/evidence.py" "$ledger" "$data"
"$python" "$base/plots.py" "$ledger" "$data"
