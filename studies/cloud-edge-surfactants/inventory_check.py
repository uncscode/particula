"""Calculate areal supply from the paired concentration and RH axes."""

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Publisher PDF, printed p. 14308; pdfimages -f 8 -l 8 -png, second image
# 2067 x 1857. Lower 70% RH tick x=1390.5; upper 5.2/5.0 M ticks
# x=1386.0/1450.5. Approximate binary-CA concentration used by the authors.
ACID_M = np.interp(1390.5, [1386.0, 1450.5], [5.2, 5.0])
AVOGADRO = 6.02214076e23


def main():
    """Calculate areal coverage ceilings from wet diameter and composition."""
    path = HERE / "inventory_check.csv"
    with path.open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(
            [
                "recipe",
                "wet_diameter_nm",
                "acid_M",
                "surfactant_molecules",
                "maximum_surface_excess_umol_m2",
            ]
        )
        for recipe, ratio in [("octyl", 100), ("dodecyl", 1000)]:
            c = 1000 * ACID_M / ratio  # mol/m³
            for d_nm in [50, 100, 200, 500, 1000, 5000, 7500, 10000]:
                d = d_nm * 1e-9
                writer.writerow(
                    [
                        recipe,
                        d_nm,
                        ACID_M,
                        AVOGADRO * c * np.pi * d**3 / 6,
                        c * d / 6 * 1e6,
                    ]
                )


if __name__ == "__main__":
    main()
