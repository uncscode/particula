"""Compare AFGLUS/TSIS-1 broadband paired flux with DISORT/REPTRAN.

Generate the external result once with a libRadtran 2.0.6 prefix including
the fine solar REPTRAN lookups. Running without a prefix replays the native
part against the compact stored integrated reference.
"""

import argparse
import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from disort import HERE, run_uvspec, write_csv
from particula.optics.column import SpectralColumn
from particula.optics.radiative_transfer import solve_shortwave
from particula.optics.solar import SolarInput, earth_reference
from particula.optics.surface import lambertian_surface
from scipy.io import netcdf_file

DATA = HERE.parents[2] / "particula/optics/reference_data"
MU = 0.65
ALBEDO = 0.2


def compare_bands(prefix, temp, scale, resolution):
    """Return TSIS-weighted DISORT fluxes for the requested REPTRAN grid."""
    data = prefix / "share/libRadtran/data"
    filename = data / f"correlated_k/reptran/reptran_solar_{resolution}.cdf"
    with netcdf_file(filename, mmap=False) as ds:
        low, high = (
            ds.variables["wvlmin"][:].copy(),
            ds.variables["wvlmax"][:].copy(),
        )
    # Integrate exactly the common spectral interval on the TSIS-1 1 nm bins.
    lo, hi = (
        np.load(DATA / "gas_full.npz")["band_bounds_m"][[0, -1], [0, 1]] * 1e9
    )
    idx = np.flatnonzero((high > lo + 1e-5) & (low < hi - 1e-5))
    source = np.loadtxt(DATA / "tsis1.txt")
    edges = np.r_[source[:, 0] - 0.5, source[-1, 0] + 0.5]
    energy = np.r_[0.0, np.cumsum(source[:, 1])]
    total = np.zeros(4)
    # Provide the native aerosol strictly in the bottom kilometre.
    (temp / "profile.txt").write_text("120 NULL\n1 vacuum.txt\n0 layer.txt\n")
    wave = np.arange(270.0, 4021.0)
    moment = 0.55 ** np.arange(33)
    np.savetxt(temp / "moments.txt", moment)
    for name, factor in [("layer", scale), ("vacuum", 0.0)]:
        np.savetxt(
            temp / f"{name}.txt",
            np.column_stack(
                [
                    wave,
                    factor * (wave / 550.0) ** -1.2,
                    np.full_like(wave, 0.90),
                    np.tile(moment, (len(wave), 1)),
                ]
            ),
            fmt="%.9g",
        )
    # Bounded runs keep the full-resolution external calculation portable.
    for subset in np.array_split(idx, max(1, int(np.ceil(len(idx) / 400)))):
        nominal = (low[subset] + high[subset]) / 2
        result = run_uvspec(
            prefix,
            temp,
            f"""mol_abs_param reptran {resolution}
source solar
rte_solver disort
number_of_streams 16
disort_intcor off
aerosol_default
aerosol_file explicit profile.txt
aerosol_file moments moments.txt
sza {np.degrees(np.arccos(MU))}
albedo {ALBEDO}
zout 0 toa
wavelength {nominal[0]:.6f} {nominal[-1]:.6f}
output_process per_band
output_user lambda zout edir edn eup""",
        )
        ground = result[result[:, 1] == 0]
        top = result[result[:, 1] > 0]
        if (
            len(ground) != len(subset)
            or np.max(abs(ground[:, 0] - nominal)) > 0.05
        ):
            raise ValueError(
                (
                    len(subset),
                    ground.shape,
                    nominal[[0, -1]],
                    ground[[0, -1], 0],
                )
            )
        band_energy = np.interp(
            np.minimum(high[subset], hi), edges, energy
        ) - np.interp(np.maximum(low[subset], lo), edges, energy)
        # A ratio removes the original Kurucz in-band illumination and
        # multiplies by the matching TSIS integrated extraterrestrial energy.
        factor = band_energy * MU / top[:, 2]
        total += (
            np.array([ground[:, 2], ground[:, 3], ground[:, 4], top[:, 4]])
            @ factor
        )
    return total


def native_case(grid, scale):
    """Return matching surface and TOA fluxes for one native integration."""
    model = np.load(DATA / f"gas_{grid}.npz")
    z = model["interface_height_m"]
    earth = earth_reference(
        0.0, 0.0, datetime(2024, 3, 20, 12, tzinfo=timezone.utc), z, grid=grid
    )
    direct = earth.sun.direct_normal * earth.position.distance_au**2
    beam = SolarInput(
        earth.sun.wavelength_m,
        MU,
        direct,
        0.0 * direct,
        earth.sun.quadrature_weights_m,
    )
    depth = np.zeros_like(earth.column.optical_depth)
    depth[-1] = scale * (earth.sun.wavelength_m / 550e-9) ** -1.2
    moment = np.broadcast_to(0.55 ** np.arange(17), (*depth.shape, 17))
    column = earth.column.plus(
        SpectralColumn(
            beam.wavelength_m, z, depth, np.full_like(depth, 0.90), moment
        )
    )
    result = solve_shortwave(
        column, beam, lambertian_surface(beam.wavelength_m, ALBEDO), streams=4
    )
    return np.array(
        [
            result.integrate(field)
            for field in (
                result.direct_down[-1],
                result.diffuse_down[-1],
                result.up[-1],
                result.up[0],
            )
        ]
    )


def main():
    """Generate reference values or replay the native Earth calculation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", nargs="?", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    fields = ["surface_direct", "surface_diffuse", "surface_up", "toa_up"]
    if args.prefix:
        rows = []
        for mode in ["coarse", "fine"]:
            for loading in [0.0, 0.01]:
                with tempfile.TemporaryDirectory() as directory:
                    result = compare_bands(
                        args.prefix.resolve(), Path(directory), loading, mode
                    )
                rows.append(
                    dict(
                        grid=mode,
                        tau550=loading,
                        **dict(zip(fields, result, strict=True)),
                    )
                )
                print(mode, loading, result, flush=True)
        write_csv(HERE / "earth_disort.csv", rows)
    if not args.reference_only:
        with (HERE / "earth_disort.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for grid in ["reduced", "full"]:
            for loading in [0.0, 0.01]:
                result = native_case(grid, loading)
                rows.append(
                    dict(
                        grid=f"native_{grid}",
                        tau550=loading,
                        **dict(zip(fields, result, strict=True)),
                    )
                )
                print(grid, loading, result, flush=True)
        write_csv(HERE / "earth_flux.csv", rows)


if __name__ == "__main__":
    main()
