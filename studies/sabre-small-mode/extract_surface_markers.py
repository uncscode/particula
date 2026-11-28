"""Read surface-mode markers from the 2024 SABRE meeting, slide 10.

Run: python extract_surface_markers.py Thornberry.pdf surface_markers.csv
Requires NumPy, Pillow, and Poppler's pdfimages.
"""

import argparse
import csv
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE = (
    "https://csl.noaa.gov/events/sabre2024/presentations/day1/Thornberry.pdf"
)
PDF_SHA256 = "b1ede8b83385f1aab6ac94c41361d84ea87c1d9f11e30391ddaa624d861f287b"


def diameter_nm(x):
    """Map a calibrated horizontal pixel position to diameter in nm."""
    # Embedded image: 10, 100, 1000 nm ticks center on x=301, 498, 695.
    return 100.0 * 10 ** ((x - 498) / 197)


def marker_center(image, y, left, right, kind):
    """Locate the colored symbol center within a bounded plot region."""
    region = image[y - 7 : y + 8, left:right]
    red, green, blue = np.moveaxis(region, -1, 0)
    if kind == "triangle":
        inside = (red > 235) & (green > 235) & (blue > 235)
    else:
        inside = (
            (red > 130) & (green > 165) & (blue - red > 20) & (green - red > 15)
        )
    x = np.nonzero(inside)[1] + left
    if len(x) < 80 or np.ptp(x) > 23:
        raise ValueError(f"Unexpected {kind} at image row {y}")
    return float(np.median(x))


def extract(pdf, out):
    """Export marker centers from the checksum-verified source."""
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError(
            f"Expected the slide deck from {SOURCE} with hash {PDF_SHA256}"
        )
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / "slide"
        subprocess.run(  # noqa: S603 - fixed Poppler executable and arguments
            [  # noqa: S607
                "pdfimages",
                "-f",
                "10",
                "-l",
                "10",
                "-png",
                str(pdf),
                str(prefix),
            ],
            check=True,
        )
        image = np.asarray(
            Image.open(prefix.with_name("slide-000.png")), dtype=float
        )
    if image.shape != (900, 870, 3):
        raise ValueError("Unexpected source image dimensions")
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "n2o_center_ppbv",
                "small_marker_x_px",
                "large_marker_x_px",
                "small_surface_diameter_nm",
                "large_surface_diameter_nm",
            )
        )
        for n2o in range(105, 350, 10):
            # Tick centers at y=37 and 766 correspond to 100 and 350 ppbv.
            y = round(37 + (n2o - 100) * 729 / 250)
            small_x = marker_center(image, y, 470, 589, "triangle")
            large_x = marker_center(image, y, 605, 680, "circle")
            writer.writerow(
                (
                    n2o,
                    small_x,
                    large_x,
                    f"{diameter_nm(small_x):.2f}",
                    f"{diameter_nm(large_x):.2f}",
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    extract(args.pdf, args.csv)
