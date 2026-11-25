"""Extract the visible Figure 6b growth-factor band from the publisher PDF."""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    """Extract the calibrated envelope of plotted growth curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", default="growth_envelope.csv", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "figure"
        subprocess.run(  # noqa: S603
            ["pdfimages", "-f", "9", "-l", "9", "-png", args.pdf, prefix],  # noqa: S607
            check=True,
        )
        pixels = np.asarray(
            Image.open(f"{prefix}-000.png").convert("RGB"), dtype=int
        )
    if pixels.shape != (600, 2067, 3):
        raise ValueError(
            "Figure image has changed. Check the axis calibration."
        )
    # Panel b. RH 0, 100 at x=814,1358.5. GF 1.0,1.8 at y=455,27.
    with args.output.open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["rh_pct", "gf_min", "gf_max"])
        for rh in range(10, 95):
            x = round(814 + 5.445 * rh)
            stripe = pixels[1:480, x]
            colored = (stripe[:, 1] - stripe[:, 0] > 8) & (stripe[:, 0] < 232)
            y = np.flatnonzero(colored) + 1
            writer.writerow(
                [
                    rh,
                    f"{1 + (455 - max(y)) / 535:.6f}",
                    f"{1 + (455 - min(y)) / 535:.6f}",
                ]
            )


if __name__ == "__main__":
    main()
