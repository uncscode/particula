"""Rebuild the native integration-test fixture with external C DISORT.

Run from the checkout root with PYTHONPATH=. and pass the libRadtran prefix.
Each monochromatic run uses solar normal irradiance in mW/m²/nm. The layer
files supply extinction per km, albedo, and normalized Legendre moments.
"""

import argparse
import csv
import io
import tempfile
from pathlib import Path

import numpy as np

from disort import run_uvspec
from particula.integration_tests.spectral_radiative_transfer_test import (
    WAVE,
    native_scene,
)
from particula.optics.surface import dusty_ground, ocean_surface


def write_layer_files(folder, column, index, wavelength_nm):
    thickness_km = -np.diff(column.interface_height_m) / 1000
    for layer, name in enumerate(("upper", "lower")):
        moments = column.phase_moments[layer, index]
        row = np.r_[
            column.optical_depth[layer, index] / thickness_km[layer],
            column.single_scattering_albedo[layer, index],
            np.pad(moments, (0, 65 - len(moments))),
        ]
        # uvspec trims trailing zeros when inferring its phase-array size.
        row[-1] = max(row[-1], 1e-25)
        table = [np.r_[wavelength_nm + offset, row] for offset in (-1, 1)]
        np.savetxt(folder / f"{name}.txt", table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    reference, aerosol, _ = native_scene()
    cases = (
        ("base", reference.column),
        ("case", reference.column.plus(aerosol)),
    )
    surfaces = (("ocean", ocean_surface(WAVE)), ("soil", dusty_ground(WAVE)))
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "surface",
            "state",
            "wavelength_nm",
            "up_top_W_m2_nm",
            "down_ground_W_m2_nm",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "profile.txt").write_text(
            "120 NULL\n1.2 upper.txt\n0 lower.txt\n"
        )
        for label, surface in surfaces:
            for state, column in cases:
                for index, wavelength_nm in enumerate(WAVE * 1e9):
                    write_layer_files(folder, column, index, wavelength_nm)
                    beam = reference.sun.direct_normal[index] * 1e-6
                    np.savetxt(
                        folder / "beam.txt",
                        [
                            [wavelength_nm + offset, beam]
                            for offset in (-1, 0, 1)
                        ],
                    )
                    settings = f"""source solar beam.txt
mol_abs_param crs
no_absorption mol
no_scattering mol
aerosol_default
aerosol_file explicit profile.txt
rte_solver disort
number_of_streams 32
disort_intcor off
sza {np.degrees(np.arccos(reference.sun.mu0))}
albedo {surface.coefficients[0][index]}
wavelength {wavelength_nm:g} {wavelength_nm:g}
zout 0 120
output_user lambda zout edir edn eup"""
                    rows = run_uvspec(args.prefix, folder, settings)
                    bottom, top = rows[np.argsort(rows[:, 1])]
                    writer.writerow(
                        [
                            label,
                            state,
                            round(wavelength_nm),
                            top[4] * 0.001,
                            (bottom[2] + bottom[3]) * 0.001,
                        ]
                    )
    target = Path("particula/integration_tests/data/shortwave_disort.csv")
    target.write_text(output.getvalue())


if __name__ == "__main__":
    main()
