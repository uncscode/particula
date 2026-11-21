"""Digitize Salignat et al. Figures 3b, 5a and S5a at native resolution."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2] / "docs/data/sea2cloud-events"


def extract(fig3, fig5, supp):
    """Use printed axes, marker centers, and the published SMPS color scale."""
    a = np.asarray(Image.open(supp).convert("RGB"), dtype=float)
    # Standard emissions are orange. Exclude the upper-right legend.
    red, green, blue = a.transpose(2, 0, 1)
    mask = (red > green + 70) & (green > blue + 25) & (red > 135)
    mask[:151, 743:] = False
    acid = []
    for h in range(97):
        x = round(147.5 + h * (1271.5 - 147.5) / 96)
        if h == 96:
            x = 1270
        ys = np.flatnonzero(mask[48:469, x]) + 48
        if not len(ys):
            for offset in (1, 2, 3, 4, 5):
                ys = (
                    np.where(
                        mask[
                            48:469,
                            max(147, x - offset) : min(1272, x + offset + 1),
                        ]
                    )[0]
                    + 48
                )
                if len(ys):
                    break
            if not len(ys):
                raise ValueError(f"Missing HNO3 curve at {h} hours")
        acid.append([h, max(0, (466.3 - np.median(ys)) * 1e7 / 419)])
    np.savetxt(
        ROOT / "hourly-acid.csv",
        acid,
        delimiter=",",
        fmt="%.6g",
        header="hours_since_20200319_00utc,HNO3_molecules_cm3",
        comments="",
    )

    # Erode each ~10-pixel black square so observations retain their centers.
    # Dense overlaps can only support ranges in half-hour windows.
    a = np.asarray(Image.open(fig5).convert("RGB"))
    dark = np.max(a, axis=2) < 90
    dark[:18] = False
    dark[457:] = False
    dark[:, :207] = False
    dark[:, 1524:] = False
    dark[:112, 1250:] = False
    y, x = np.nonzero(binary_erosion(dark, np.ones((9, 9))))
    hours = (x - 204.5) * 96 / (1526.5 - 204.5)
    number = np.maximum(0, (450.7 - y) * 2500 / (450.7 - 23.3))
    bins = np.floor(hours * 2).astype(int)
    obs = [
        [(i + 0.5) / 2, min(number[bins == i]), max(number[bins == i])]
        for i in np.unique(bins)
    ]
    np.savetxt(
        ROOT / "observations.csv",
        obs,
        delimiter=",",
        fmt="%.5f",
        header="hours_since_20200319_00utc,low_cm3,high_cm3",
        comments="",
    )

    a = np.asarray(Image.open(fig3).convert("RGB"), dtype=float)
    palette = np.median(a[816:826, 128:1407], axis=0)
    colors = cKDTree(palette)
    edges = np.geomspace(12, 400, 25)
    diameter = np.sqrt(edges[:-1] * edges[1:])
    rows = []
    for hour in np.arange(-3, 90, 0.5):
        x = round(260.5 + hour * 166 / 24)
        ys = np.rint(
            640 - np.log(diameter / 25) * (640 - 427) / np.log(400 / 25)
        ).astype(int)
        pixels = np.median(a[ys, x - 1 : x + 2], axis=1)
        dist, idx = colors.query(pixels)
        if max(dist) > 95:
            continue
        n = 10 ** (idx * 4 / (len(palette) - 1)) * np.diff(np.log10(edges))
        rows.extend(zip(np.full_like(n, hour), diameter, n, strict=True))
    np.savetxt(
        ROOT / "background.csv",
        rows,
        delimiter=",",
        fmt="%.6g",
        header="hours_since_20200319_00utc,diameter_nm,bin_number_cm3",
        comments="",
    )


if __name__ == "__main__":
    extract(*sys.argv[1:])
