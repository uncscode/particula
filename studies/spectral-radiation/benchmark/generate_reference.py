"""Derive AFGLUS layer absorption at REPTRAN nodes using libRadtran 2.0.6.

Run from the repository with the prefix of a libRadtran install as argument.
Only the generated optical depths and quadrature are used at runtime.
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "particula" / "optics" / "reference_data"
WAVE = re.compile(r"\*\*\* wavelength: iv = \d+, ([\d.]+) nm$")
ROW = re.compile(r"^\s*\d+ \|")
# Snap these desired boundaries to existing REPTRAN band boundaries.
MACRO_NM = [
    280,
    315,
    360,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    760,
    800,
    880,
    1000,
    1150,
    1350,
    1500,
    1750,
    2000,
    2250,
    2500,
    3000,
    4000,
]


def gas_log(prefix, data, low_nm, high_nm, directory):
    """Parse the native layer setup, before adding clouds or scattering."""
    setup = f"""data_files_path {data}
atmosphere_file {data}/atmmod/afglus.dat
mol_abs_param reptran coarse
source solar
sza 0
wavelength {low_nm:.6f} {high_nm:.6f}
rte_solver null
verbose
"""
    run = subprocess.run(  # noqa: S603 - user-selected local executable
        [prefix / "bin" / "uvspec"],
        input=setup,
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode:
        raise RuntimeError(run.stderr[-2000:])
    parsed = {}
    layer_rows = None
    wave = None
    for line in run.stderr.splitlines():
        if "redistributing molecular optical properties" in line:
            break
        match = WAVE.match(line)
        if match:
            wave = float(match[1])
        if "setup_gases(), layer properties" in line:
            layer_rows = []
        if layer_rows is not None and ROW.match(line):
            row = line.split("|")
            layer_rows.append(float(row[3]))
        if layer_rows and "--------" in line:
            parsed[wave] = layer_rows
            layer_rows = None
    return parsed


def full_reference(prefix, data):
    """Collect layer depths, quadrature, and the native AFGLUS profile."""
    model = np.loadtxt(data / "atmmod" / "afglus.dat")
    with netcdf_file(
        data / "correlated_k/reptran/reptran_solar_coarse.cdf", mmap=False
    ) as nc:
        v = nc.variables
        lo, hi = v["wvlmin"][:], v["wvlmax"][:]
        ids = np.flatnonzero((lo >= 280) & (hi <= 4000))
        wavelength, bounds, weights, extra = [], [], [], []
        for band in ids:
            n = v["nwvl_in_band"][band]
            for index, weight in zip(
                v["iwvl"][:n, band], v["iwvl_weight"][:n, band], strict=True
            ):
                if weight > 0:
                    wavelength.append(v["wvl"][index - 1])
                    bounds.append([lo[band], hi[band]])
                    weights.append(weight * v["wvl_integral"][band])
                    extra.append(v["extra"][index - 1])
        low, high = lo[ids].copy(), hi[ids].copy()

    diagnostics = {}
    with tempfile.TemporaryDirectory() as tmp:
        for chunk in np.array_split(np.arange(len(ids)), 38):
            first, last = chunk[[0, -1]]
            diagnostics.update(
                gas_log(
                    prefix,
                    data,
                    (low[first] + high[first]) / 2,
                    (low[last] + high[last]) / 2,
                    tmp,
                )
            )
    # uvspec's internal wavelength axis is single precision and printed to 6
    # decimals. Match at that precision, then verify uniqueness and distance.
    keys = np.array(sorted(diagnostics))
    if not keys.size:
        raise ValueError("No uvspec layer diagnostics found")
    wave = np.asarray(wavelength)
    order = np.argsort(wave)
    wave = wave[order]
    near = np.searchsorted(keys, wave).clip(0, len(keys) - 1)
    left = (near - 1).clip(0)
    near = np.where(abs(keys[left] - wave) < abs(keys[near] - wave), left, near)
    if (
        len(np.unique(near)) != len(near)
        or np.max(abs(keys[near] - wave)) > 0.0003
    ):
        raise ValueError("Representative wavelength match failed")
    tau = np.asarray([diagnostics[k] for k in keys[near]]).T
    if tau.shape != (len(model) - 1, len(wave)) or np.min(tau) < 0:
        raise ValueError("Layer array differs from AFGLUS heights")
    return (
        dict(
            wavelength_m=wave * 1e-9,
            band_bounds_m=np.asarray(bounds)[order] * 1e-9,
            weight_m=np.asarray(weights)[order] * 1e-9,
            absorption_tau=tau.astype(np.float32),
        ),
        np.asarray(extra)[order],
        model,
    )


def reduced_reference(full, extra, model):
    """Preserve gas transmission moments using positive sparse quadrature."""
    w = full["wavelength_m"] * 1e9
    edges = np.unique(full["band_bounds_m"])
    edges = edges[
        np.abs(edges[:, None] - np.asarray(MACRO_NM) * 1e-9).argmin(axis=0)
    ]
    nodes, widths, bounds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        band = np.flatnonzero(
            (full["wavelength_m"] >= lo) & (full["wavelength_m"] < hi)
        )
        wave = w[band]
        weight = full["weight_m"][band] / (hi - lo)
        normal = extra[band] / np.max(extra[band])
        gas = full["absorption_tau"][:, band].astype(float)
        mean = (wave - wave.min()) / np.ptp(wave)
        # Moments reproduce the reference's incident energy, smooth spectral
        # slope and integrated direct gas transmission at three air masses.
        columns = [
            np.ones_like(wave),
            normal,
            mean,
            normal * mean,
            normal * (wave[0] / wave) ** 4,
        ]
        for altitude, cosines in [
            (0, (1, 0.5, 0.2)),
            (2, (0.5,)),
            (10, (0.5,)),
        ]:
            depth = gas[model[1:, 0] >= altitude].sum(axis=0)
            columns.extend(normal * np.exp(-depth / mu) for mu in cosines)
        a = np.array(columns)
        # Low optical depths produce near-dependent moments. Normalize and
        # discard redundant rows by SVD so the linear program is well scaled.
        _, singular, vectors = np.linalg.svd(a, full_matrices=False)
        a = vectors[singular > 1e-8 * singular[0]]
        sol = linprog(
            (mean - 0.5) ** 2,
            A_eq=a,
            b_eq=a @ weight,
            bounds=(0, None),
            method="highs",
        )
        if not sol.success:
            raise ValueError(sol.message)
        take = sol.x > 1e-11
        nodes.extend(band[take])
        widths.extend(sol.x[take] * (hi - lo))
        bounds.extend([[lo, hi]] * np.count_nonzero(take))
    return dict(
        wavelength_m=full["wavelength_m"][nodes],
        band_bounds_m=np.asarray(bounds),
        weight_m=np.asarray(widths),
        absorption_tau=full["absorption_tau"][:, nodes],
    )


def main():
    """Write full and reduced quadrature data to the chosen directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    data = prefix / "share" / "libRadtran" / "data"
    full, extra, model = full_reference(prefix, data)
    reduced = reduced_reference(full, extra, model)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, grid in [("full", full), ("reduced", reduced)]:
        np.savez_compressed(
            args.output / f"gas_{name}.npz",
            **grid,
            interface_height_m=model[:, 0] * 1e3,
            pressure_pa=model[:, 1] * 100,
            temperature_k=model[:, 2],
            water_number_m3=model[:, 6] * 1e6,
            ozone_number_m3=model[:, 4] * 1e6,
        )
        print(
            name,
            grid["absorption_tau"].shape,
            "covered width [nm]",
            grid["weight_m"].sum() * 1e9,
        )


if __name__ == "__main__":
    main()
