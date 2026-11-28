"""Compare paired flux and directional radiance against libRadtran's C DISORT.

Usage: PYTHONPATH=. python studies/spectral-radiation/benchmark/disort.py PREFIX
"""

import argparse
import csv
import subprocess
import tempfile
from io import StringIO
from itertools import product
from pathlib import Path

import numpy as np
from particula.optics.column import SpectralColumn
from particula.optics.radiative_transfer import solve_shortwave
from particula.optics.solar import SolarInput
from particula.optics.surface import lambertian_surface

HERE = Path(__file__).resolve().parent
WAVE = np.array([450.0, 550.0, 850.0]) * 1e-9
DIRECT = np.array([2.0, 1.8, 1.2]) * 1e9  # W m-2 per metre
VIEW_MU = 0.8
AZIMUTH = 60.0


def write_csv(path, rows):
    """Save a small table with its physical quantities as column names."""
    with path.open("w", newline="") as stream:
        out = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        out.writeheader()
        out.writerows(rows)


def run_uvspec(prefix, directory, setup):
    """Run the specified local libRadtran executable and parse its table."""
    data = prefix / "share/libRadtran/data"
    settings = (
        f"data_files_path {data}\natmosphere_file {data}/atmmod/afglus.dat\n"
        f"{setup}\nquiet\n"
    )
    proc = subprocess.run(  # noqa: S603 - user-selected local executable
        [prefix / "bin/uvspec"],
        input=settings,
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if (
        proc.returncode
        or not proc.stdout.strip()
        or "ERROR" in proc.stderr.upper()
    ):
        raise RuntimeError(
            f"{settings}\n{proc.stderr[-3000:]}\n{proc.stdout[-1000:]}"
        )
    return np.loadtxt(StringIO(proc.stdout), ndmin=2)


def slab_depth(tau550):
    """Return dimensionless vertical extinction depth at the test nodes."""
    return tau550 * (WAVE / 550e-9) ** -1.2


def disort_slab(
    prefix,
    directory,
    mu,
    albedo,
    tau550,
    ssa,
    g,
    *,
    boundary=None,
    azimuth=AZIMUTH,
):
    """Evaluate one external DISORT column at the three spectral nodes."""
    # Each profile row gives the LOWER altitude of a slab, in km. The upper
    # 119 km of this calculation is a vacuum. DISORT sees a 1 km uniform slab.
    directory = Path(directory)
    (directory / "profile.txt").write_text(
        "120 NULL\n1 vacuum.txt\n0 layer.txt\n"
    )
    # Explicit profile sets extinction and SSA. libRadtran 2.0.6 requires
    # aerosol_file moments to set the global DISORT phase expansion length.
    moment = g ** np.arange(65)
    np.savetxt(directory / "moments.txt", moment)
    wl = WAVE * 1e9
    np.savetxt(directory / "beam.txt", np.column_stack([wl, DIRECT * 1e-6]))
    np.savetxt(
        directory / "layer.txt",
        np.column_stack(
            [
                wl,
                slab_depth(tau550),
                np.full_like(wl, ssa),
                np.tile(moment, (len(wl), 1)),
            ]
        ),
    )
    np.savetxt(
        directory / "vacuum.txt",
        np.column_stack(
            [
                wl,
                np.zeros_like(wl),
                np.ones_like(wl),
                np.tile(moment, (len(wl), 1)),
            ]
        ),
    )
    setup = f"""source solar beam.txt
mol_abs_param crs
no_absorption mol
no_scattering mol
aerosol_default
aerosol_file explicit profile.txt
aerosol_file moments moments.txt
rte_solver disort
number_of_streams 32
disort_intcor off
sza {np.degrees(np.arccos(mu))}
phi0 0
umu {VIEW_MU}
phi {180 - azimuth}
{boundary if boundary else f"albedo {albedo}"}
wavelength 450 850
wavelength_grid_file beam_wavelengths.txt
zout 0 1
output_user lambda zout edir edn eup uu"""
    np.savetxt(directory / "beam_wavelengths.txt", wl)
    out = run_uvspec(prefix, directory, setup)
    out = out[np.lexsort((out[:, 0], -out[:, 1]))]
    if out.shape != (6, 6):
        raise ValueError(out.shape)
    return out[:, 2:].reshape(2, 3, 4) * 1e6  # mW/nm to W/m


def native_slab(mu, albedo, tau550, ssa, g, streams=12):
    """Solve the same column and viewing geometry using Particula."""
    depth = slab_depth(tau550)[None, :]
    column = SpectralColumn(
        WAVE,
        [1000.0, 0.0],
        depth,
        np.full_like(depth, ssa),
        np.tile(g ** np.arange(65), (1, 3, 1)),
    )
    beam = SolarInput(WAVE, mu, DIRECT, np.zeros_like(WAVE))
    return solve_shortwave(
        column,
        beam,
        lambertian_surface(WAVE, albedo),
        streams=streams,
        view_zenith_deg=np.degrees(np.arccos(VIEW_MU)),
        relative_azimuth_deg=AZIMUTH,
    )


def reference_sweep(prefix):
    """Generate the matched external case and baseline fixtures."""
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        for mu, albedo, baseline, ssa in product(
            (0.3, 0.65, 0.9), (0.04, 0.4, 0.8), (0.02, 0.4), (0.72, 0.98)
        ):
            for case, tau in [
                ("reference", baseline),
                ("case", baseline + 0.01),
            ]:
                ref = disort_slab(prefix, temp, mu, albedo, tau, ssa, 0.55)
                for k, wave in enumerate(WAVE):
                    rows.append(
                        dict(
                            mu=mu,
                            albedo=albedo,
                            baseline=baseline,
                            ssa=ssa,
                            case=case,
                            tau550=tau,
                            wavelength_nm=round(wave * 1e9),
                            toa_up_W_m2_nm=ref[0, k, 2] * 1e-9,
                            surface_down_W_m2_nm=ref[1, k, :2].sum() * 1e-9,
                            toa_radiance_W_m2_nm_sr=ref[0, k, 3] * 1e-9,
                        )
                    )
    write_csv(HERE / "disort_reference.csv", rows)


def compare():
    """Replay native fluxes and calculate signed integrated residuals."""
    from collections import defaultdict

    groups = defaultdict(list)
    with (HERE / "disort_reference.csv").open() as stream:
        for r in csv.DictReader(stream):
            key = tuple(
                float(r[k]) for k in ("mu", "albedo", "baseline", "ssa")
            )
            groups[key].append(r)
    rows = []
    bands = [
        "toa_up_W_m2_nm",
        "surface_down_W_m2_nm",
        "toa_radiance_W_m2_nm_sr",
    ]
    for (mu, albedo, baseline, ssa), pair in groups.items():
        results = {}
        for case in ("reference", "case"):
            data = [r for r in pair if r["case"] == case]
            native = native_slab(
                mu, albedo, float(data[0]["tau550"]), ssa, 0.55
            )
            benchmark = (
                np.array([[float(r[key]) for r in data] for key in bands]) * 1e9
            )
            model = np.array(
                [
                    native.up[0],
                    native.direct_down[-1] + native.diffuse_down[-1],
                    native.radiance,
                ]
            )
            results[case] = np.trapezoid(model - benchmark, x=WAVE, axis=-1)
        delta = results["case"] - results["reference"]
        rows.append(
            dict(
                mu=mu,
                albedo=albedo,
                baseline_tau=baseline,
                ssa=ssa,
                absolute_toa_up_error_W_m2=results["case"][0],
                delta_toa_up_error_W_m2=delta[0],
                delta_surface_down_error_W_m2=delta[1],
                delta_toa_radiance_error_W_m2_sr=delta[2],
                absolute_toa_radiance_error_W_m2_sr=results["case"][2],
            )
        )
    write_csv(HERE / "disort_flux_errors.csv", rows)
    print({k: max(abs(r[k]) for r in rows) for k in list(rows[0])[4:]})


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("prefix", type=Path, nargs="?")
    prefix = args.parse_args().prefix
    if prefix:
        reference_sweep(prefix.resolve())
    compare()
