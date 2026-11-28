"""Interpolate REPTRAN gas depth to the footprint water column and height."""

import argparse
import json
from pathlib import Path

import numpy as np
from particula.optics.solar import (
    _gas_band_equivalent,
    _gas_reference,
    _profile_grid,
)
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent


def interfaces(row):
    """Return the three-layer altitude grid in metres above sea level."""
    ground = row["elevation_m"]
    return np.array([120000.0, 10000.0, ground + 2000.0, ground])


def stage(directory, target):
    """Preserve native band and broadband optical depths for each footprint."""
    rows = json.loads((HERE / "footprints.json").read_text())["footprints"]
    water_tables = []
    for path in sorted(directory.glob("water-*.npz")):
        with np.load(path) as data:
            water_tables.append(dict(data))
    first = water_tables[0]
    wave = _gas_reference("reduced")["wavelength_m"]
    nodes = np.searchsorted(first["wavelength_m"], wave)
    np.testing.assert_allclose(
        first["wavelength_m"][nodes], wave, rtol=0, atol=1e-15
    )
    rsr = np.loadtxt(HERE / "terra-response.csv", delimiter=",", skiprows=1)
    band_wave = rsr[:, 1] * 1e-9
    pressures, wide, bands = [], [], []
    for row in rows:
        height = interfaces(row)
        columns, spectra = [], []
        for table in water_tables:
            z = table["interface_height_m"]
            # One g/cm2 of integrated vapor mass is one cm liquid water.
            amount = (
                -cumulative_trapezoid(
                    table["water_number_cm3"], z * 100, initial=0
                )
                * 18.01528
                / 6.02214076e23
            )
            columns.append(np.interp(height[-1], z[::-1], amount[::-1]))
            tau, pressure = _profile_grid(table, height)
            spectra.append(tau)
        target_cm = row["precipitable_water_cm"]
        if not columns[0] <= target_cm <= columns[-1]:
            raise ValueError(
                f"Water profile range excludes footprint {row['footprint']}"
            )
        hi = np.searchsorted(columns, target_cm)
        lo = max(0, hi - 1)
        fraction = (
            (target_cm - columns[lo]) / (columns[hi] - columns[lo])
            if hi != lo
            else 0
        )
        gas = (1 - fraction) * spectra[lo] + fraction * spectra[hi]
        pressures.append(pressure * (row["pressure_hpa"] * 100 / pressure[-1]))
        wide.append(gas[:, nodes])
        bands.append(_gas_band_equivalent(first, gas, band_wave)[0])
    np.savez_compressed(
        target,
        footprint=np.array([r["footprint"] for r in rows]),
        source=np.array([r["source"] for r in rows]),
        pressure_pa=pressures,
        broad_gas_tau=np.array(wide, dtype=np.float32),
        band_gas_tau=np.array(bands, dtype=np.float32),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.directory, args.output)
