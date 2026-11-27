"""Reduce native scenario forecasts to the pooled envelopes used in Figure 2."""

import argparse
from pathlib import Path

import pandas as pd


def summarize(source: Path, output: Path):
    """Pool accepted trajectories across the five chamber settings at 20 h."""
    rows = pd.read_csv(source)
    half_bin = rows["resolution"] == "half_bin"
    full_bin_test = (
        (rows["resolution"] == "full_bin")
        & (rows["precursor"] == "ocimene")
        & (rows["diameter_nm"] == 60)
    )
    keep = (
        (rows["time_min"] == 1200)
        & (rows["accepted"] > 0)
        & (half_bin | full_bin_test)
    )
    keys = ["precursor", "mechanism", "resolution", "diameter_nm", "time_min"]
    table = (
        rows[keep]
        .groupby(keys, as_index=False)
        .agg(
            accepted=("accepted", "sum"),
            low=("low", "min"),
            high=("high", "max"),
        )
    )
    table.to_csv(output, index=False, float_format="%.10g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.source, args.output)
