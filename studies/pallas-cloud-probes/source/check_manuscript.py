"""Check the published concentration, spread and sign changes from replayed tables."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def check(directory):
    field = pd.read_csv(directory / "distributions.csv").query("high_um <= 13")
    totals = field.groupby("day")[
        ["FSSP", "ICEMET", "model_FSSP", "model_ICEMET"]
    ].sum()
    np.testing.assert_allclose(
        totals[["FSSP", "ICEMET"]], [[123.8, 173.5], [91.8, 75.1]], atol=0.05
    )
    for a, b, expected in [
        ("FSSP", "ICEMET", [-33.4, 20.1]),
        ("model_FSSP", "model_ICEMET", [14.7, 22.5]),
    ]:
        percent = 200 * (totals[a] - totals[b]) / (totals[a] + totals[b])
        np.testing.assert_allclose(percent, expected, atol=0.05)
    unc = pd.read_csv(directory / "uncertainty-budget.csv")
    intervals = unc.query(
        "scenario == 'total' and quantity == 'percent'"
    ).sort_values("day")
    np.testing.assert_allclose(
        intervals[["p025", "p975"]], [[-65.7, -5.3], [-14.8, 48.6]], atol=0.05
    )
    sim = (
        pd.read_csv(directory / "method-budget.csv")
        .query("day == 15 and nm == 633")
        .set_index(["probe", "setting"])
    )
    keys = [("F", "nominal"), ("I", "nominal"), ("I", "uncorrected")]
    np.testing.assert_allclose(
        sim.loc[keys, "reported_band_cm3"], [84.5, 74.2, 92.9], atol=0.05
    )
    np.testing.assert_allclose(
        sim.loc[keys[:2], "within_size_scatter_um"], [0.90, 0.71], atol=0.005
    )
    np.testing.assert_allclose(
        sim.loc[keys[:2], "total_sizing_rmse_um"], [1.42, 1.11], atol=0.005
    )
    print(
        "Verified field counts, model contrast, uncertainty intervals, sizing errors and correction reversal."
    )


if __name__ == "__main__":
    check(Path(sys.argv[1]))
