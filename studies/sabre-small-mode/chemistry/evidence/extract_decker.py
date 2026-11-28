"""Extract tracer-bin marks from the preliminary Decker SABRE poster."""

import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[1] / "raw/Decker-2024-poster.pdf"
SOURCE_HASH = "4ea73c592a07de50e22cd0ad4b989018e812709e223e725c90639ff4ffc8cffa"


def main():
    """Read the long horizontal marks, excluding ticks and short error caps."""
    if hashlib.sha256(PDF.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("Poster checksum mismatch")
    with tempfile.TemporaryDirectory() as folder:
        prefix = Path(folder) / "image"
        subprocess.run(  # noqa: S603
            ["pdfimages", "-png", str(PDF), str(prefix)],  # noqa: S607
            check=True,
        )
        rgb = np.asarray(Image.open(str(prefix) + "-018.png").convert("RGB"))
        alpha = np.asarray(Image.open(str(prefix) + "-019.png"))
    if rgb.shape != (1487, 3456, 3):
        raise ValueError("Unexpected Air Mass Relation panel")
    blue = (rgb[:, :, 0] == 26) & (rgb[:, :, 1] > 78) & (alpha > 128)
    # Pixel x=927 and 3264 are 150 and 350 ppbv. Pixel y=143 and 623
    # are -1 and -3 on the log10(gamma) axis. The 250 ppbv and -2 ticks
    # fall at x=2095 and y=383, respectively.
    rows = []
    for lower in range(270, 330, 5):
        x = round(927 + (lower - 150) * (3264 - 927) / 200)
        counts = blue[141:638, x : x + 58].sum(axis=1)
        ys = np.flatnonzero(counts >= 48)
        if ys.size == 0:
            continue
        if np.ptp(ys) > 12:
            raise ValueError("Ambiguous horizontal mark")
        y = 141 + np.mean(ys)
        rows.append((lower, lower + 5, y, 10 ** (-1 - (y - 143) / 240)))
    if [row[0] for row in rows] != list(range(280, 320, 5)):
        raise ValueError("Unexpected marker bins")
    pd.DataFrame(
        rows, columns=["n2o_lo", "n2o_hi", "center_y_px", "gamma_marker"]
    ).to_csv(HERE / "decker-tracer-marks.csv", index=False)


if __name__ == "__main__":
    main()
