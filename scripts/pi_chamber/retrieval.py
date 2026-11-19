"""Recover shared and case-specific humidity from saved CUDA haze channels."""

import json
from pathlib import Path

import numpy as np


def fit_humidity():
    """Minimize the squared log-channel residual on the simulated grid."""
    data = Path(__file__).resolve().parents[2] / "docs/data/pi-chamber"
    table = np.genfromtxt(
        data / "haze-grid.csv",
        names=True,
        delimiter=",",
        dtype=None,
        encoding="utf-8",
    )
    target = json.loads((data / "haze-target.json").read_text())
    signals = {}
    grids = {}
    for case, obs in target.items():
        rows = np.sort(table[table["case"] == case], order="mean_pct")
        h = np.stack([rows[k] for k in ["h0", "h1", "h2"]], axis=1)
        f = np.asarray(obs["fractions"])
        signals[case] = np.column_stack(
            [
                np.log((h[:, [0, 2]] / h[:, [1]]) / (f[[0, 2]] / f[1])),
                np.log(rows["haze"] / obs["haze_cm3"]),
            ]
        )
        grids[case] = rows["mean_pct"]
    np.testing.assert_array_equal(grids["FW"], grids["SW"])
    x = grids["FW"]
    signals["shared"] = np.column_stack([signals["FW"], signals["SW"]])
    estimates = {}
    for key, r in signals.items():
        # The objective is quadratic on each linear interpolation interval.
        slope = np.diff(r, axis=0) / np.diff(x)[:, None]
        shift = np.clip(
            -(r[:-1] * slope).sum(axis=1) / (slope**2).sum(axis=1),
            0,
            np.diff(x),
        )
        loss = ((r[:-1] + shift[:, None] * slope) ** 2).sum(axis=1)
        best = np.argmin(loss)
        estimates[key] = float(x[best] + shift[best])
    return estimates
