"""Validate an independently regenerated matched pair or CPU/CUDA case."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "seasonal/results"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parity", action="store_true")
columns = ["activated_cm3", "median_um"]
if parser.parse_args().parity:
    values = [
        pd.read_csv(RESULTS / f"parity-{device}/m07_clean_ranked.csv")
        .set_index("w")
        .sort_index()[columns]
        for device in ["cpu", "cuda"]
    ]
    np.testing.assert_allclose(values[0], values[1], rtol=0, atol=1e-6)
    print("CPU and CUDA July cases agree within 1e-6 in cm^-3 and micrometres.")
else:
    for name, expected in [
        ("m00_clean_dark", [41.056666, 21.351928]),
        ("m07_clean_ranked", [35.62380723, 22.39863159]),
    ]:
        values = pd.read_csv(RESULTS / "smoke" / (name + ".csv"))
        np.testing.assert_allclose(
            values[columns].to_numpy(), [expected], rtol=0, atol=1e-5
        )
        print(name, values[columns].iloc[0].to_dict())
