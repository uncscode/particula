"""Extract normalized Figure 4A traces from the authors' IGOR experiment."""

import argparse
import hashlib
from pathlib import Path

import numpy as np


def extract(pxp, output):
    """Read the deposited wave values using the optional igor2 package."""
    from igor2 import packed

    expected = (
        "02b417be44a3cbce1705c6b7dedb64e30856820c2f492ef5b564d1d7e8c2bc92"
    )
    if hashlib.sha256(pxp.read_bytes()).hexdigest() != expected:
        raise ValueError("Expected the original Figure 4 PXP")
    _, tree = packed.load(str(pxp))

    def wave(name):
        return tree["root"][name].wave["wave"]["wData"].astype(float).ravel()

    time = wave(b"IMS_time_min")
    columns = [time]
    for name in (
        b"C9H14O4_MS_1min_avg",
        b"C9H14O4_isomer_1_Area_TS",
        b"C9H14O4_isomer_2_Area_TS",
    ):
        raw = wave(name)
        baseline = raw[time <= 0].mean()
        scale = raw[-5:].mean() - baseline
        columns.append((raw - baseline) / scale)
    np.savetxt(
        output,
        np.column_stack(columns),
        delimiter=",",
        fmt="%.10g",
        header="minutes,formula_normalized,cis_pinic_normalized,unknown_normalized",
        comments="",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pxp", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.pxp, args.output)
