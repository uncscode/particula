"""Compare the frozen replay and its matched outputs with the public references."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def verify(output):
    """Compare final arrays and tables after running compare.py and threshold_check.py."""
    reference = Path(__file__).resolve().parent
    for name in (
        "predictions-192.npz",
        "predictions-288.npz",
        "published-spectra.npz",
    ):
        with (
            np.load(reference / name) as expected,
            np.load(output / name) as actual,
        ):
            assert set(actual.files) == set(expected.files), name
            for key in actual.files:
                np.testing.assert_array_equal(
                    actual[key], expected[key], err_msg=f"{name}:{key}"
                )
    for name in (
        "stage-scores.csv",
        "matched-series.csv.gz",
        "threshold-check.csv",
    ):
        pd.testing.assert_frame_equal(
            pd.read_csv(reference / name),
            pd.read_csv(output / name),
            check_exact=True,
        )
    print(
        "Frozen spectra, measured and modeled counts, and summary tables reproduced."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="Frozen replay directory after rescoring"
    )
    verify(parser.parse_args().output)
