"""Reduce the NASA MCST Terra in-band workbook to two energy-response curves."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

SOURCE = "https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/Terra_RSR_in-band.xlsx"
DIGEST = "f14498183f9258ed691999313a13fc425edb2eb2d89ca676f79abbb6f48ad637"


def detector_curve(samples, wave):
    """Return one unit-area detector curve after excluding MCST fill rows."""
    # MCST uses -99 for missing responses, including edge samples.
    # Preserve the common gas-optics grid, but interpolate only data.
    samples = samples[samples[:, 3] != -99]
    if (
        len(samples) < 2
        or not np.all(np.isfinite(samples[:, 2:4]))
        or np.any(np.diff(samples[:, 2]) <= 0)
        or np.any((samples[:, 3] < 0) | (samples[:, 3] > 1))
    ):
        raise ValueError("Invalid in-band detector samples")
    curve = np.interp(wave, samples[:, 2], samples[:, 3], left=0, right=0)
    area = np.trapezoid(curve, wave)
    if area <= 0:
        raise ValueError("Empty detector response on the target grid")
    return curve / area


def main():
    """Average the forty area-normalized detector responses in each band."""
    import openpyxl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.workbook.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected source workbook checksum")
    workbook = openpyxl.load_workbook(args.workbook, read_only=True)
    # Independent band support avoids integrating across the interband gap.
    out = []
    for band in (1, 2):
        values = np.array(
            list(workbook[f"band_{band}"].values)[1:], dtype=float
        )
        wave = np.arange(
            np.floor(values[:, 2].min()) - 1, np.ceil(values[:, 2].max()) + 2
        )
        detectors = []
        for channel in np.unique(values[:, 1]):
            samples = values[values[:, 1] == channel]
            detectors.append(detector_curve(samples, wave))
        mean = np.mean(detectors, axis=0)
        out.extend(
            zip(np.full(wave.size, band), wave, mean / mean.max(), strict=True)
        )
    np.savetxt(
        args.output,
        out,
        fmt=("%d", "%.0f", "%.9f"),
        delimiter=",",
        header="band,wavelength_nm,relative_response",
        comments="",
    )


if __name__ == "__main__":
    main()
