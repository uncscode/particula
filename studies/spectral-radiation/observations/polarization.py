"""Compare scalar and polarized MYSTIC radiances at two observed geometries."""

import argparse
import csv
import json
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def run(job):
    """Run a clear molecular column at one band center."""
    prefix, row, wave, polarized = job
    data = prefix / "share/libRadtran/data"
    ocean = row["surface"] == "ocean"
    boundary = (
        f"brdf_cam u10 {row['wind_m_s']:.6f}"
        if ocean
        else f"albedo {row['land_surface_reflectance'][2]:.6f}"
    )
    angle = (180 - row["relative_azimuth_deg"]) % 360
    # Distinct seeds allow independent simulation standard errors to combine.
    seed = 48311 + 1000 * int(polarized) + int(wave) + int(ocean)
    inp = f"""data_files_path {data}
atmosphere_file {data}/atmmod/afglus.dat
source solar solar.dat
wavelength {wave}
mol_abs_param crs
no_absorption mol
sza {row["solar_zenith_deg"]:.7f}
phi0 0
umu {np.cos(np.deg2rad(row["view_zenith_deg"])):.10f}
phi {angle:.7f}
{boundary}
zout toa
rte_solver mystic
mc_escape on
mc_photons 2000000
mc_randomseed {seed}
mc_std
{"mc_polarisation" if polarized else ""}
quiet
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        solar = np.loadtxt(
            HERE.parents[2] / "particula/optics/reference_data/tsis1.txt"
        )
        window = np.array([wave - 1, wave, wave + 1])
        local = np.column_stack(
            [window, np.interp(window, solar[:, 0], solar[:, 1])]
        )
        local[:, 1] *= 1000  # uvspec expects mW m-2 nm-1.
        np.savetxt(root / "solar.dat", local, fmt="%.11g")
        proc = subprocess.run(  # noqa: S603 - user-selected local executable
            [prefix / "bin/uvspec"],
            input=inp,
            text=True,
            cwd=root,
            capture_output=True,
            check=False,
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr[-2000:])
        i = np.loadtxt(root / "mc.rad.spc", ndmin=2)[0, -1]
        se = np.loadtxt(root / "mc.rad.std.spc", ndmin=2)[0, -1]
    return dict(
        source=row["source"],
        footprint=row["footprint"],
        surface=row["surface"],
        wavelength_nm=wave,
        polarization=int(polarized),
        seed=seed,
        intensity_w_m2_sr_um=i,
        mc_standard_error=se,
    )


def main():
    """Compare a high-view land case and the nearest-to-specular ocean case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads((HERE / "footprints.json").read_text())["footprints"]
    land = max(
        (r for r in rows if r["surface"] == "bare"),
        key=lambda r: r["view_zenith_deg"],
    )
    water = min(
        (r for r in rows if r["surface"] == "ocean"),
        key=lambda r: r["glint_angle_deg"],
    )
    jobs = list(
        product(
            [args.prefix.resolve()], [land, water], [645, 858], [False, True]
        )
    )
    with ProcessPoolExecutor(4) as pool:
        result = list(pool.map(run, jobs))
    with args.output.open("w") as out:
        writer = csv.DictWriter(out, fieldnames=result[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()
