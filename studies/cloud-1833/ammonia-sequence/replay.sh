#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export PARTICULA_ROOT="$(realpath "${1:?Particula checkout af0dc67bf7}")"
out="$(realpath -m "${2:?new replay destination}")"
export PY="${3:?path to Particula environment Python}"
mkdir -p "$out"
build="$(mktemp -d)"
trap 'rm -rf -- "$build"' EXIT
bash "$here/build.sh" "$build"
# Materialize the frozen experiment inputs in the temporary build folder.
"$PY" - "$here/designs.json" "$build" <<'INPUTS'
import json, sys
from pathlib import Path
for name, settings in json.loads(Path(sys.argv[1]).read_text()).items():
    path = Path(sys.argv[2]) / (name + ".json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(settings))
INPUTS
for config in "$build"/[0-9][0-9]-*/*.json; do
    study="$(basename "$(dirname "$config")")"
    case_name="$(basename "$config" .json)"
    bash "$here/run.sh" "$config" "$out/$study/$case_name" "$build"
done
"$PY" "$here/summarize.py" "$out"
"$PY" "$here/transfer.py" "$out"
