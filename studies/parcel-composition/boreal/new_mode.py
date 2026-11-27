"""Measure new-mode size errors alongside dry absorption."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from metrics import compress
from model import SEEDS

from optics import Optics


def diagnostic(folder, opt):
    """Measure how far the cumulative new-particle size spectrum moves."""
    rows = []
    for event in ["burst-peak", "03600", "05400"]:
        path = folder / (event + ".npz")
        boxes, _, t = read(path)
        for parcel, b in enumerate(boxes):
            young = ~b.old
            if not np.any(young):
                continue
            dry = b.dry.copy()
            water = b.water.copy()
            d = np.sort(np.cbrt(dry[young].sum(1)))
            area = opt.areas(dry, water)[:, 0].sum()
            for label in ("16 per bin", "one new-particle row"):
                b.dry = dry.copy()
                b.water = water.copy()
                if label == "16 per bin":
                    compress(b, 10, 16, 2)
                else:
                    b.dry[young] = dry[young].mean(0)
                    b.water[young] = water[young].mean()
                reduced = np.sort(np.cbrt(b.dry[young].sum(1)))
                points = np.unique(np.r_[d, reduced])
                delta = np.stack(
                    [
                        (
                            np.searchsorted(d, points, side)
                            - np.searchsorted(reduced, points, side)
                        )
                        / len(d)
                        for side in ["left", "right"]
                    ]
                )
                j, k = np.unravel_index(abs(delta).argmax(), delta.shape)
                error = 100 * abs(
                    opt.areas(b.dry, b.water)[:, 0].sum() / area - 1
                )
                rows.append(
                    [
                        folder.parent.name,
                        t,
                        parcel,
                        label,
                        len(d),
                        points[k],
                        100 * abs(delta[j, k]),
                        error,
                    ]
                )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    opt = Optics(a.optics)
    rows = [
        r
        for s in SEEDS[8:]
        for r in diagnostic(a.base / str(s) / "c1b1n1", opt)
    ]
    pd.DataFrame(
        rows,
        columns=(
            "seed time parcel encoding cohort_count feature_nm "
            "cumulative_error_pp absorption_error_pct"
        ).split(),
    ).to_csv(a.out, index=False, float_format="%.8g")
