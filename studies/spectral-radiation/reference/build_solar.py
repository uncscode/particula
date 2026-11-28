"""Rebuild the compact 1-nm TSIS-1 irradiance table from a LISIRD CSV."""

import argparse
import hashlib
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    wave, flux = np.loadtxt(
        args.source, delimiter=",", skiprows=1, usecols=(0, 1), unpack=True
    )
    keep = (wave >= 279) & (wave <= 4001)
    np.savetxt(
        args.output,
        np.column_stack((wave[keep], flux[keep])),
        header="TSIS-1 HSRS full spectrum | wavelength_nm irradiance_W_m-2_nm-1",
        fmt=("%.1f", "%.12g"),
    )
    print(hashlib.sha256(args.source.read_bytes()).hexdigest())
    print(
        "280-4000 nm irradiance [W/m2]:",
        flux[(wave > 280) & (wave < 4000)].sum(),
    )


if __name__ == "__main__":
    main()
