"""Check the public Welas plot table against full-precision 1 s trajectories."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

PLOT_DATA = (
    Path(__file__).resolve().parents[2]
    / "website/public/study-data/jeju-chamber/figure-data.zip"
)
SERIES = ["measured", "shared", "individual"]


def check_bands(results, plot_data):
    """Compare integrated plots with independent sums of native bins."""
    with ZipFile(plot_data) as archive:
        bins = pd.read_csv(archive.open("welas-bins.csv")).set_index("bin")
        table = pd.read_csv(archive.open("welas-spectra.csv"))
    bins = bins.sort_index()
    np.testing.assert_array_equal(bins.index, np.arange(70))
    np.testing.assert_array_equal(bins.upper_um[:-1], bins.lower_um[1:])
    width = np.log10(bins.upper_um) - np.log10(bins.lower_um)
    center = np.sqrt(bins.upper_um * bins.lower_um)
    with np.load(results / "observations.npz") as observations:
        observed = observations["y"]
        valid = observations["valid"]
        diameter = observations["d"]
    with np.load(results / "pooled-full-1s.npz") as pooled:
        np.testing.assert_array_equal(pooled["runs"], np.arange(6))
        shared = pooled["y"]

    records = []
    np.testing.assert_array_equal(np.sort(table.run.unique()), np.arange(1, 7))
    for run, group in table.groupby("run"):
        i = run - 1
        with np.load(results / f"R{run}-full-1s.npz") as fit:
            np.testing.assert_array_equal(fit["runs"], [i])
            individual = fit["y"][0]
        minutes = np.flatnonzero(valid[i])
        expected_index = pd.MultiIndex.from_product(
            (minutes, bins.index), names=("minute", "bin")
        )
        group = group.set_index(["minute", "bin"]).sort_index()
        pd.testing.assert_index_equal(group.index, expected_index)
        spectra = group[SERIES].to_numpy().reshape(len(minutes), len(bins), 3)
        reference = np.stack(
            (observed[i, minutes], shared[i, minutes], individual[minutes]),
            axis=-1,
        )
        for low, high in [(2, 5), (5, 10), (10, 20)]:
            keep = (center >= low) & (center < high)
            native = (diameter >= low) & (diameter < high)
            np.testing.assert_array_equal(keep, native)
            actual = (
                spectra[:, keep] * width[keep].to_numpy()[None, :, None]
            ).sum(axis=1)
            expected = reference[:, native].sum(axis=1)
            # The published CSVs retain eight significant digits.
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
            delta = abs(actual - expected)
            relative = np.divide(
                delta, expected, out=np.zeros_like(delta), where=expected != 0
            )
            records.append((len(minutes), delta.max(), relative.max()))
            if run == 5 and low == 5:
                print(f"Run 5: {len(minutes)} valid minutes, {keep.sum()} bins")
                for label, public, full in zip(
                    SERIES,
                    actual.mean(axis=0),
                    expected.mean(axis=0),
                    strict=True,
                ):
                    print(
                        f"{label}: plot {public:.12g}, full {full:.12g} cm^-3"
                    )
    n, absolute, relative = np.asarray(records).T
    print(f"{int(n.sum()) * len(SERIES)} minute/band/series comparisons")
    print(
        f"max absolute {absolute.max():.12g} cm^-3, "
        f"max relative {relative.max():.12g}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, help="directory produced by replay.py"
    )
    parser.add_argument("--plot-data", type=Path, default=PLOT_DATA)
    args = parser.parse_args()
    check_bands(args.results, args.plot_data)
