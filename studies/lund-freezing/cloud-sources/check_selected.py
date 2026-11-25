"""Check six physical outputs for each of the eleven representative GPU cases."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from campaign import Experiment, key

METRICS = [
    "primary",
    "resident_primary",
    "fresh_primary",
    "suspended",
    "LWP",
    "IWP",
]


def selected_rows(ledger, cases):
    requested = [key(Experiment(**case)) for case in cases]
    records = {}
    for record in ledger:
        experiment = Experiment(**record["experiment"])
        name = key(experiment)
        if name in requested:
            records[name] = {
                "case": requested.index(name) + 1,
                "sky": experiment.sky,
                "donor": experiment.donor,
                "seed": experiment.seed,
                **{metric: record["values"][metric] for metric in METRICS},
            }
    if set(records) != set(requested):
        raise ValueError(
            f"Expected {len(requested)} selected cases, found {len(records)}"
        )
    return pd.DataFrame([records[name] for name in requested])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()
    cases = json.loads(
        Path(__file__).with_name("replay-cases.json").read_text()
    )
    with args.ledger.open() as stream:
        actual = selected_rows(map(json.loads, stream), cases)
    expected = pd.read_csv(args.expected)
    keys = ["case", "sky", "donor", "seed"]
    pd.testing.assert_frame_equal(actual[keys], expected[keys])
    np.testing.assert_allclose(
        actual[METRICS], expected[METRICS], rtol=3e-10, atol=2e-10
    )
    print(
        f"Verified {len(expected)} selected GPU cases across {len(METRICS)} endpoint quantities."
    )
