#!/usr/bin/env bash
# Run from the retained packet with the Particula checkout and python paths exported.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?supply a fresh output directory}
mkdir -p "$out"
out=$(cd "$out" && pwd)
cache=$(mktemp -d)
trap 'rm -rf -- "$cache"' EXIT
for phase in bank fine; do bash "$here/run.sh" "$out" "$cache" "$phase"; done
OPENBLAS_NUM_THREADS=3 "${PY:-python3}" "$here/recovery.py" "$out" "$cache"
bash "$here/run.sh" "$out" "$cache" transfer
"${PY:-python3}" "$here/summarize.py" "$out"
"${PY:-python3}" - "$here" "$out" <<'CHECK'
from pathlib import Path
import sys
import pandas as pd
reference, replay = map(Path, sys.argv[1:])
for name in ("04-held-transfer.csv", "classification-summary.csv"):
    pd.testing.assert_frame_equal(pd.read_csv(reference/name), pd.read_csv(replay/name),
                                  rtol=1.e-8, atol=1.e-8)
for phase in ("bank", "fine", "transfer"):
    assert pd.read_csv(replay/(phase+"-ledger.csv")).max_budget_error.max() < 1.e-9
CHECK
