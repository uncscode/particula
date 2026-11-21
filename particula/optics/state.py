"""Bulk and individual optics from native particle state."""

from dataclasses import dataclass

import numpy as np

from particula.optics.cross_sections import get_cross_sections
from particula.optics.mie import get_mie_efficiencies
from particula.optics.population import (
    PopulationOptics,
    get_population_optics,
    get_population_weights,
)


def _index(values):
    result = np.asarray(values, dtype=np.complex128)
    if (
        not np.all(np.isfinite(result))
        or np.any(result.real <= 0)
        or np.any(result.imag < 0)
    ):
        raise ValueError("Require finite optical indices with n>0 and k>=0")
    return result


def lorentz_lorenz(index):
    """Convert absolute refractive index to Lorentz–Lorenz polarizability."""
    dielectric = _index(index) ** 2
    return (dielectric - 1) / (dielectric + 2)


def mix_refractive_index(index, volume_fraction):
    """Mix absolute indices along the last axis on a volume basis.

    The fractions are dimensionless, nonnegative and sum to one.
    Both inputs broadcast with their common final material axis.
    """
    fraction = np.asarray(volume_fraction, dtype=np.float64)
    if (
        np.any(~np.isfinite(fraction))
        or np.any(fraction < 0)
        or not np.allclose(np.sum(fraction, axis=-1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Optical volume fractions must sum to one")
    mixed = np.sum(fraction * lorentz_lorenz(index), axis=-1)
    return _index(np.sqrt((1 + 2 * mixed) / (1 - mixed)))


@dataclass(frozen=True, init=False)
class OpticalConfig:
    """Prescribed optical inputs for homogeneous spheres.

    Wavelengths are ordered vacuum wavelengths [m]. Supply an absolute
    particle index (scalar or per wavelength), or a species index table
    (wavelength, material) in native particle-material order. Medium index
    is real, positive and constant across the spectrum.
    """

    wavelength: tuple[float, ...]
    medium_index: float
    particle_index: tuple[complex, ...] | None
    species_index: tuple[tuple[complex, ...], ...] | None

    def __init__(
        self,
        wavelength,
        *,
        particle_index=None,
        species_index=None,
        medium_index=1.0,
    ):
        """Validate and freeze wavelengths and the chosen index model."""
        if np.iscomplexobj(wavelength) or np.iscomplexobj(medium_index):
            raise ValueError("Wavelength and medium index must be real")
        wave = np.atleast_1d(np.asarray(wavelength, dtype=np.float64))
        medium = float(medium_index)
        if (
            wave.ndim != 1
            or not wave.size
            or np.any(~np.isfinite(wave))
            or np.any(wave <= 0)
            or np.any(np.diff(wave) <= 0)
            or not np.isfinite(medium)
            or medium <= 0
        ):
            raise ValueError(
                "Require increasing positive wavelengths and medium"
            )
        if (particle_index is None) == (species_index is None):
            raise ValueError(
                "Specify either particle or species optical indices"
            )
        prescribed = None
        species = None
        if particle_index is not None:
            prescribed = tuple(
                np.broadcast_to(_index(particle_index), wave.shape)
            )
        else:
            table = _index(species_index)
            if (
                table.ndim != 2
                or table.shape[0] != wave.size
                or not table.shape[1]
            ):
                raise ValueError(
                    "Species indices require (wavelength, material)"
                )
            species = tuple(map(tuple, table))
            if not np.all(np.isfinite(lorentz_lorenz(table))):
                raise ValueError("Species mixing polarizability must be finite")
        object.__setattr__(self, "wavelength", tuple(wave))
        object.__setattr__(self, "medium_index", medium)
        object.__setattr__(self, "particle_index", prescribed)
        object.__setattr__(self, "species_index", species)

    def validate_materials(self, materials):
        """Check the species table against the native material axis."""
        if (
            self.species_index is not None
            and len(self.species_index[0]) != materials
        ):
            raise ValueError("Optical table and particle material axes differ")

    def metadata(self):
        """Return JSON-compatible optical inputs."""
        values = (
            self.particle_index
            if self.species_index is None
            else self.species_index
        )
        index = np.asarray(values)
        return {
            "wavelength_m": list(self.wavelength),
            "medium_index": self.medium_index,
            "index_type": "prescribed"
            if self.species_index is None
            else "species",
            "index_real": index.real.tolist(),
            "index_imag": index.imag.tolist(),
        }


def particle_optics(particles, config):
    """Return per-particle Mie efficiencies and cross sections.

    Arrays have axes (box, wavelength, slot). Empty slots have zero radius
    for the optical calculation. Discrete and resolved sizes come from
    material mass and density. Continuous PDFs use their explicit nodes.
    """
    particles.validate_representation()
    config.validate_materials(particles.n_species)
    active = particles.concentration > 0
    native_radius = (
        particles.radius_grid
        if particles.distribution_type == "continuous_pdf"
        else particles.radii
    )
    radius = np.where(active, native_radius, 0)[:, None, :]
    if config.species_index is None:
        index = np.asarray(config.particle_index)[None, :, None]
    else:
        volumes = particles.masses / particles.density
        volumes = np.where(particles.concentration[..., None] > 0, volumes, 0)
        total = volumes.sum(axis=-1, keepdims=True)
        if np.any(active & (total[..., 0] <= 0)):
            raise ValueError(
                "Species optics require positive active material volume"
            )
        fraction = np.divide(
            volumes, total, out=np.zeros_like(volumes), where=total > 0
        )
        fraction[..., 0] += total[..., 0] == 0
        index = mix_refractive_index(
            np.asarray(config.species_index)[None, :, None, :],
            fraction[:, None, :, :],
        )
    efficiency = get_mie_efficiencies(
        radius,
        np.asarray(config.wavelength)[None, :, None],
        index,
        config.medium_index,
    )
    return efficiency, get_cross_sections(efficiency, radius)


def population_optics(particles, config) -> PopulationOptics:
    """Evaluate live CPU particle state into (box, wavelength) results."""
    measure = particles.distribution_type
    weights = get_population_weights(
        particles.concentration,
        measure,
        volume=particles.volume if measure == "particle_resolved" else None,
        radius_grid=particles.radius_grid
        if measure == "continuous_pdf"
        else None,
    )
    if not particles.n_particles:
        zero = np.zeros((particles.n_boxes, len(config.wavelength)))
        config.validate_materials(particles.n_species)
        particles.validate_representation()
        return PopulationOptics(
            zero,
            zero.copy(),
            zero.copy(),
            zero.copy(),
            zero.copy(),
            zero.copy(),
            zero.copy(),
            zero.astype(bool),
            zero.astype(bool),
        )
    efficiency, cross = particle_optics(particles, config)
    return get_population_optics(
        cross, efficiency.asymmetry, weights[:, None, :]
    )
