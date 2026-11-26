"""Calculate action-weighted Mie absorption of hydrated limonene spheres."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from particula.optics import get_mie_efficiencies, mix_refractive_index
from particula.optics.gpu.workspace import prepare_mie
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "solar" / "spectrum.csv"
AVOGADRO = 6.02214076e23
WATER_INDEX = 1.345
FIELDS = ("mvos", "lvos")


@dataclass(frozen=True)
class Material:
    """Optical and water-uptake settings for one dry material."""

    n: float = 1.5
    rh: float = 0.8
    kappa: float = 0.045
    absorption_scale: float = 1.0
    density_kg_m3: float = 1400.0
    kelvin: bool = True
    liu_index: bool = False
    liu_absorption: bool = False


def sizes(dry_nm, settings):
    """Sub-saturated kappa-Köhler equilibrium including Kelvin curvature."""
    planar_volume = 1 + settings.kappa * settings.rh / (1 - settings.rh)
    if settings.rh == 0 or not settings.kelvin:
        return dry_nm * planar_volume ** (1 / 3)
    kelvin_length_nm = 4 * 0.072 * 0.01801528 / (997 * 8.314462618 * 294) * 1e9
    ratios = [
        brentq(
            lambda v, d=d: (v - 1)
            / (v - 1 + settings.kappa)
            * np.exp(kelvin_length_nm / (d * v ** (1 / 3)))
            - settings.rh,
            1,
            planar_volume,
        )
        for d in dry_nm
    ]
    return dry_nm * np.cbrt(ratios)


def material(dry_nm, nm, settings):
    """Absolute wet n+ik and dry exponential attenuation per metre."""
    wet_nm = sizes(dry_nm, settings)
    dry_vol_fraction = (dry_nm / wet_nm) ** 3
    number_density = settings.density_kg_m3 / 0.250 * AVOGADRO
    # Bulk absorption, separately from each class' photochemical action.
    sigma_m2 = (
        1.3e-19 * 1e-4 * np.exp(np.log(9.1e-21 / 1.3e-19) * (nm - 300) / 45)
    )
    alpha = number_density * sigma_m2 * settings.absorption_scale
    real_part = settings.n
    if settings.liu_index or settings.liu_absorption:
        data = np.genfromtxt(HERE / "liu_uv.csv", delimiter=",", names=True)
        if settings.liu_index:
            real_part = np.interp(
                nm, data["wavelength_nm"], data["n_dry_limonene"]
            )
        if settings.liu_absorption:
            # Resolve the exp(-alpha z) normalization as k tends to zero where
            # the published k rounds to zero at four decimal places.
            imag = np.interp(
                nm, data["wavelength_nm"], data["k_abs_dry_limonene"]
            )
            alpha = 4 * np.pi * np.maximum(imag, 1e-8) / (nm * 1e-9)
    dry_index = real_part + 1j * alpha * (nm * 1e-9) / (4 * np.pi)
    index = np.stack((dry_index, np.full_like(dry_index, WATER_INDEX)), axis=-1)
    fraction = np.stack((dry_vol_fraction, 1 - dry_vol_fraction), axis=-1)
    wet_index = mix_refractive_index(index[None, :, :], fraction[:, None, :])
    return wet_nm, wet_index, alpha


def solve(dry_nm, nm, settings, device):
    """Run the spherical scattering series and check energy in float64."""
    wet_nm, index, alpha = material(dry_nm, nm, settings)
    size_param = np.pi * wet_nm[:, None] / nm
    if device == "numpy":
        series = get_mie_efficiencies(
            wet_nm[:, None] * 0.5e-9, nm * 1e-9, index
        )
        ext, sca, absorb, g = (
            series.extinction,
            series.scattering,
            series.absorption,
            series.asymmetry,
        )
    else:
        shape = size_param.shape
        pair = np.stack(
            (np.real(index).ravel(), np.imag(index).ravel()), axis=1
        )
        workspace = prepare_mie(
            wp.array(pair, dtype=wp.vec2d, device=device),
            wp.array(size_param.ravel(), dtype=wp.float64, device=device),
            coefficient_capacity=192,
            case_tile=4096,
        )
        workspace.enqueue()
        workspace.check()
        # Warp public workspace layout: Qext, Qsca, Qabs, g, Qback, Qpr, valid.
        values = workspace.efficiencies.numpy().reshape(*shape, 7)
        ext, sca, absorb, g = (values[..., i] for i in range(4))
    if not (
        np.isfinite(ext + sca + absorb + g).all()
        and (absorb > 0).all()
        and (sca >= 0).all()
        and np.max(np.abs(ext - sca - absorb)) < 1e-11
    ):
        raise ValueError("Physical Mie energy or finite-state check failed")
    cross_area = np.pi * (wet_nm[:, None] * 0.5e-9) ** 2
    dry_volume = np.pi / 6 * (dry_nm[:, None] * 1e-9) ** 3
    factors = absorb * cross_area / (alpha * dry_volume)
    return factors, (ext * cross_area, sca * cross_area, g)


def integrated(table, factors, source="solar"):
    """Normalize the action-weighted Mie response by the same incident dose."""
    nm = table["wavelength_nm"]
    photons = table[source + "_photons_cm2_s_nm"]
    res = {}
    for kind in FIELDS:
        action_flux = photons * table["action_" + kind + "_cm2"]
        res[kind] = np.trapezoid(
            factors * action_flux, nm, axis=1
        ) / np.trapezoid(action_flux, nm)
    return res


def save_table(path, columns):
    """Write equal-length named arrays as a CSV with a stable column order."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


def central(table, target, device, dry):
    """Produce the nominal solar and lamp tables plus scattering inputs."""
    factors, cross = solve(dry, table["wavelength_nm"], Material(), device)
    ratios = integrated(table, factors)
    save_table(
        target / "dose.csv",
        dict(
            diameter_nm=dry,
            ratio_mvos=ratios["mvos"],
            ratio_lvos=ratios["lvos"],
        ),
    )

    # The airborne residue assay needs the optical response at 330 nm only.
    w = int(np.flatnonzero(table["wavelength_nm"] == 330)[0])
    np.savez(
        target / "cross_330nm.npz",
        diameter_nm=dry,
        extinction_m2=cross[0][:, w],
        scattering_m2=cross[1][:, w],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0", choices=["cuda:0", "numpy"]
    )
    args = parser.parse_args()
    data = np.genfromtxt(SOURCE, delimiter=",", names=True)
    central(data, HERE, args.device, np.geomspace(3, 3000, 361))
