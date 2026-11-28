"""Recover slide 14 spectrum vertices from the public SABRE workshop PDF."""

import argparse
import csv
import hashlib
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_SHA256 = (
    "00c64b32c4fcb6ee87767daa5433a25348df3db146fb1a30e98c6b6f3a22da68"
)
# Legend colors (8-bit RGB) and 30 s counts are in order of increasing N2O.
LEGEND = {
    (0, 0, 175): (90, 8),
    (0, 72, 204): (120, 188),
    (0, 178, 204): (150, 400),
    (76, 204, 127): (180, 237),
    (181, 204, 21): (210, 286),
    (204, 120, 0): (240, 205),
    (204, 15, 0): (270, 226),
    (104, 0, 0): (300, 359),
}
NUMERIC = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?"
# Coordinates in the page's 960 x 540 point viewBox. The third tick on
# each axis is an independent calibration check (see source note).
X_10, X_100 = 344.050652, 547.155494
Y_1, Y_10 = 353.638022, 253.456717
Y_BOTTOM = 383.796073


def read_graph(pdf):
    """Read the checksum-pinned PDF and render the specified page as SVG."""
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            "Source checksum mismatch; verify the slide and calibration."
        )
    with tempfile.TemporaryDirectory() as directory:
        svg = Path(directory) / "spectra.svg"
        subprocess.run(  # noqa: S603 -- fixed flags and pinned input PDF
            ["pdftocairo", "-svg", "-f", "14", "-l", "14", str(pdf), str(svg)],  # noqa: S607
            check=True,
        )
        return ET.parse(svg).getroot()  # noqa: S314 -- local Poppler output


def recover(pdf):
    """Return the original plot knots above its lower clipping line."""
    root = read_graph(pdf)
    spectra = {}
    for path in root.iter("{http://www.w3.org/2000/svg}path"):
        data = path.get("d", "")
        # Each spectrum has more than forty linear segments.
        if data.count("L") < 40 or path.get("fill") != "none":
            continue
        color = tuple(
            round(float(c) * 2.55)
            for c in re.findall(NUMERIC, path.get("stroke", ""))
        )
        if color not in LEGEND:
            continue
        n2o_lower, count = LEGEND[color]
        if n2o_lower in spectra or re.search(r"[^ML\d.,\-+\seE]", data):
            raise ValueError("Unexpected path geometry for this spectrum.")
        a, b, c, d, e, f = map(
            float, re.findall(NUMERIC, path.attrib["transform"])
        )
        vertices = [float(v) for v in re.findall(NUMERIC, data)]
        rows = []
        for x, y in zip(vertices[::2], vertices[1::2], strict=True):
            xp, yp = a * x + c * y + e, b * x + d * y + f
            if abs(yp - Y_BOTTOM) < 0.02:
                continue
            diameter = 10 ** (1 + (xp - X_10) / (X_100 - X_10))
            dndlogd = 10 ** ((Y_1 - yp) / (Y_1 - Y_10))
            rows.append((n2o_lower, n2o_lower + 30, count, diameter, dndlogd))
        if len(rows) != len(vertices) // 2 - 1:
            raise ValueError(
                "Expected exactly one endpoint on the lower clipping line."
            )
        if any(
            row[3] >= following[3]
            for row, following in zip(rows[:-1], rows[1:], strict=True)
        ):
            raise ValueError("Expected strictly increasing diameters.")
        spectra[n2o_lower] = rows
    if len(spectra) != len(LEGEND):
        raise ValueError("Did not recover all eight legend series.")
    return [row for key in sorted(spectra) for row in spectra[key]]


def main():
    """Restore the plotted curves as a table of diameter and density."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("jensen_spectra.csv"),
    )
    args = parser.parse_args()
    rows = recover(args.pdf)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "n2o_lo_ppbv",
                "n2o_hi_ppbv",
                "n_30s",
                "diameter_nm",
                "dndlog10d_cm3",
            ]
        )
        for low, high, count, diameter, density in rows:
            writer.writerow(
                [low, high, count, f"{diameter:.5f}", f"{density:.5f}"]
            )


if __name__ == "__main__":
    main()
