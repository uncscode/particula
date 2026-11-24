"""Native optical transfer probabilities from the archived, refined setup."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from instruments import (
    WATER,
    BEADS,
    MARKERS,
    sobol,
    geometry,
    profiles,
    fssp,
    pulse_lookup,
)
from simulate import device_areas, bead_fit, corrected
from study import edges

ROOT = Path(__file__).resolve().parent


def run(out):
    draws = sobol(512)
    grid = np.arange(1 + 0.0625 / 2, 40, 0.0625)
    z, p, ix = geometry(20, draws)
    r = np.arange(0, 52, 0.04)

    def image(d, index, nm):
        field = profiles(d, nm, index, z, r, 80, "mie")
        return device_areas(field, d, p, draws, ix, 0.04, 0.4, 3)

    _, slope = bead_fit(image(MARKERS, BEADS, 660))
    cache = {nm: pulse_lookup(nm, 80) for nm in [633, 660]}
    sizes = {}
    for nm in [633, 660]:
        sizes["I", nm] = corrected(image(grid, WATER, nm), slope)
        sizes["F", nm] = fssp(grid, nm, 80, draws, 633, 0.12, cache)
    parts = []
    for (probe, nm), inferred in sizes.items():
        bins = edges(probe, "upper")
        for d, sample in zip(grid, inferred):
            idx = np.searchsorted(bins, sample, side="right") - 1
            valid = (idx >= 0) & (idx < len(bins) - 1)
            prob = np.bincount(idx[valid], minlength=len(bins) - 1) / len(
                sample
            )
            for j in np.flatnonzero(prob):
                parts.append((probe, nm, d, j, prob[j]))
    pd.DataFrame(
        parts, columns=["probe", "nm", "true_um", "channel", "fraction"]
    ).to_csv(
        out / "transfer.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
