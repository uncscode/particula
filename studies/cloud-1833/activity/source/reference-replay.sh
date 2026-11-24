#!/bin/bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dest="${1:?pass the activity replay output}/reference"
py="${2:-python3}"
mkdir -p "$dest"
cp "$here/../reference/"{extract.py,figure2.py,compare.py} "$dest/"
cp "${3:?directory containing cheng2015.pdf and observations-source.html}"/{cheng2015.pdf,observations-source.html} "$dest/"
export PYTHONDONTWRITEBYTECODE=1
"$py" "$dest/extract.py"
"$py" "$dest/figure2.py"
"$py" "$dest/compare.py"
