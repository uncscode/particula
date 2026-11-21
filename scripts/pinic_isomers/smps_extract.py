"""Recover the Figure 4 clock alignment from the archived graph axes."""

import hashlib
import re
from pathlib import Path

import numpy as np


def extract(path, output):
    """Extract chamber minutes and SMPS-derived total SOA mass concentration."""
    from igor2 import packed

    raw = Path(path).read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "02b417be44a3cbce1705c6b7dedb64e30856820c2f492ef5b564d1d7e8c2bc92"
    ):
        raise ValueError("Expected original Figure 4 PXP")

    def limits(axis):
        match = re.search(rb"SetAxis " + axis + rb" ([0-9.]+),([0-9.]+)", raw)
        return np.array(match.groups(), dtype=float)

    absolute = limits(b"SMPS_x_C9")
    minutes = limits(b"C9_IMS_x")
    np.testing.assert_equal(np.diff(absolute), 60 * np.diff(minutes))
    _, tree = packed.load(path)

    def wave(name):
        return tree["root"][name].wave["wave"]["wData"].ravel()

    t = (wave(b"TotTime_10_10_2025") - absolute[0]) / 60 + minutes[0]
    mass = wave(b"SOA_Mass_Conc_10_10_2025")
    np.savetxt(
        output,
        np.column_stack((t, mass)),
        delimiter=",",
        header="minute,mass_ug_m3",
        comments="",
        fmt="%.10g",
    )


if __name__ == "__main__":
    import sys

    extract(sys.argv[1], sys.argv[2])
