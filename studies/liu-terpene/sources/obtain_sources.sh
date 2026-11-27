#!/usr/bin/env bash
set -euo pipefail
# From an empty scratch directory. Keep the source PDFs there.
curl -fLsS https://aerosol.chem.uci.edu/publications/Irvine/2026_Liu_EST_Air_SOA_evaporation.pdf -o paper.pdf
curl -fLsS https://ndownloader.figshare.com/files/60675517 -o SI.pdf
sha256sum -c <<'HASH'
b4650c1aa0e75fcc8659cc2069899173e75336e1c648e5bd2953620064b030ab  paper.pdf
e0a237ca9015647232cb2be725f29946ec2ba1801493a470bddce4d132f07e38  SI.pdf
HASH
pdfimages -f 5 -l 5 -png paper.pdf source
pdfimages -f 8 -l 8 -png paper.pdf fig7
pdfimages -f 8 -l 8 -png SI.pdf s4
