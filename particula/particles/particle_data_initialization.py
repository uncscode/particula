"""Native particle initialization without strategy-bearing facades.

Initialization allocates owned float64 arrays. Aggregate wrapping is a separate,
identity-preserving operation. Binned densities use fixed volume 1 m^3.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from particula.particles.particle_data import ParticleData
from particula.particles.particle_data_builder import ParticleDataBuilder
from particula.particles.properties.lognormal_size_distribution import (
    get_lognormal_pdf_distribution,
    get_lognormal_pmf_distribution,
)
from particula.util.convert_units import get_unit_conversion


def _vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Copy a scalar, vector, or singleton row/column into one dimension."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim > 2 or (array.ndim == 2 and 1 not in array.shape):
        raise ValueError(
            f"{name} must be scalar, vector, or singleton row/column"
        )
    return array.reshape(-1).copy()


def _single_density(value: ArrayLike, units: str) -> float:
    """Require a single material; redundant equal density values are allowed."""
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.all(array == array.flat[0]):
        raise ValueError("radius initialization requires one material density")
    result = float(array.flat[0]) * get_unit_conversion(units, "kg/m^3")
    if not np.isfinite(result) or result <= 0:
        raise ValueError("density must be positive and finite")
    return float(result)


def _bin_values(value: ArrayLike, count: int, name: str) -> NDArray[np.float64]:
    """Broadcast scalar slot values without discarding any supplied lanes."""
    array = _vector(value, name)
    if array.size == 1:
        return np.full(count, array[0], dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{name} must have one value per radius")
    return array


def _modal_parameters(
    mode: ArrayLike | None,
    geometric_standard_deviation: ArrayLike | None,
    number_concentration: ArrayLike | None,
    radius_units: str,
    number_concentration_units: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Normalize existing modal defaults and validate generator inputs."""
    modes = (
        np.array([100e-9, 1e-6])
        if mode is None
        else _vector(mode, "mode") * get_unit_conversion(radius_units, "m")
    )
    deviations = _vector(
        [1.2, 1.4]
        if geometric_standard_deviation is None
        else geometric_standard_deviation,
        "geometric_standard_deviation",
    )
    numbers = (
        np.array([1e10, 1e9])
        if number_concentration is None
        else _vector(number_concentration, "number_concentration")
        * get_unit_conversion(number_concentration_units, "1/m^3")
    )
    if not (modes.shape == deviations.shape == numbers.shape) or not modes.size:
        raise ValueError("mode, deviation, and concentration shapes must match")
    if (
        not np.all(np.isfinite(modes))
        or np.any(modes <= 0)
        or not np.all(np.isfinite(deviations))
        or np.any(deviations <= 1)
        or not np.all(np.isfinite(numbers))
        or np.any(numbers <= 0)
    ):
        raise ValueError("modes/counts must be positive finite and GSD > 1")
    return modes, deviations, numbers


def _reference_volume(value: float, units: str) -> float:
    """Canonicalize conversion roundoff only for non-SI reference volumes."""
    volume = float(value) * get_unit_conversion(units, "m^3")
    if units != "m^3" and abs(volume - 1.0) <= 4 * np.spacing(1.0):
        volume = 1.0
    return float(volume)


def create_particle_data_from_radii(
    *,
    radii: ArrayLike,
    concentration: ArrayLike,
    distribution_type: str,
    density: ArrayLike = 1000.0,
    charge: ArrayLike = 0.0,
    volume: float = 1.0,
    radius_units: str = "m",
    concentration_units: str | None = None,
    density_units: str = "kg/m^3",
    volume_units: str = "m^3",
) -> ParticleData:
    """Create owned single-box, single-material data from particle radii.

    Args:
        radii: Scalar, vector or singleton row/column of positive radii.
        concentration: Scalar or matching per-radius densities.
        distribution_type: Required ``discrete`` or ``continuous_pdf`` kind.
        density: One positive material density; redundant equal values allowed.
        charge: Scalar or per-radius elementary-charge counts, default zero.
        volume: Reference volume, which must convert to exactly 1 m^3.
        radius_units: Radius units, default metres.
        concentration_units: Defaults to 1/m^3 for bins, 1/m^4 for PDFs.
            Discrete inputs also accept 1/cm^3, as in ParticleDataBuilder.
        density_units: Material-density units, default kg/m^3.
        volume_units: Reference-volume units, default m^3. Non-SI conversion
            roundoff within four float64 ulps of one is canonicalized to one.

    Returns:
        Owned float64 data with masses (1, N, 1), density (1,), and volume
        (1,). PDF data retains its explicit radius grid. Mass is 4*pi*r^3*rho/3.

    Raises:
        ValueError: If kind, units, shapes, physical inputs or PDF grid fail
            admission. No caller arrays are modified. Empty discrete input is
            supported; PDF grids require at least two increasing nodes.
    """
    if distribution_type not in ("discrete", "continuous_pdf"):
        raise ValueError("radius bins require discrete or continuous_pdf")
    radius = _vector(radii, "radii") * get_unit_conversion(radius_units, "m")
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("radii must be positive and finite")
    material = _single_density(density, density_units)
    with np.errstate(over="ignore", invalid="ignore"):
        masses = (4.0 / 3.0 * np.pi * radius**3 * material)[:, None]
    builder = (
        ParticleDataBuilder()
        .set_distribution_type(distribution_type)
        .set_masses(masses)
        .set_density(np.array([material]))
        .set_concentration(
            _bin_values(concentration, radius.size, "concentration"),
            units=concentration_units
            if concentration_units is not None
            else (
                "1/m^4" if distribution_type == "continuous_pdf" else "1/m^3"
            ),
        )
        .set_charge(_bin_values(charge, radius.size, "charge"))
        .set_volume(np.asarray(_reference_volume(volume, volume_units)))
    )
    if distribution_type == "continuous_pdf":
        builder.set_radius_grid(radius)
    return builder.build().copy()


def create_lognormal_binned_particle_data(
    *,
    distribution_type: str,
    radius_bins: ArrayLike | None = None,
    mode: ArrayLike | None = None,
    geometric_standard_deviation: ArrayLike | None = None,
    number_concentration: ArrayLike | None = None,
    density: ArrayLike = 1000.0,
    charge: ArrayLike = 0.0,
    volume: float = 1.0,
    radius_units: str = "m",
    number_concentration_units: str = "1/m^3",
    density_units: str = "kg/m^3",
    volume_units: str = "m^3",
) -> ParticleData:
    """Initialize native bins with the existing lognormal PDF/PMF generators.

    Args:
        distribution_type: Required ``discrete`` or ``continuous_pdf``.
        radius_bins: Radius nodes; default logspace(-9, -4, 250) metres.
        mode: Lognormal scale radii, default [100e-9, 1e-6] metres.
        geometric_standard_deviation: GSD values > 1, default [1.2, 1.4].
        number_concentration: Modal number densities, default [1e10, 1e9] m^-3.
        density: Single material density, default 1000 kg/m^3.
        charge: Scalar or per-node charge, default zero.
        volume: Reference volume; must convert to exactly 1 m^3.
        radius_units: Units of supplied nodes/modes; defaults remain SI.
        number_concentration_units: Units of supplied modal densities.
        density_units: Material-density units.
        volume_units: Reference-volume units.

    Returns:
        Owned single-box ParticleData. The existing generator normalizes modes
        on the supplied radius interval; PDF density is integrated over radius,
        whereas discrete bin densities are summed. No facade is constructed.

    Raises:
        ValueError: If inputs do not define a valid distribution. Both generator
            paths require at least two positive, strictly increasing nodes.
    """
    radius = (
        np.logspace(-9, -4, 250)
        if radius_bins is None
        else _vector(radius_bins, "radius_bins")
        * get_unit_conversion(radius_units, "m")
    )
    if radius.size < 2 or np.any(np.diff(radius) <= 0):
        raise ValueError("lognormal radius grid must have increasing nodes")
    modes, deviations, numbers = _modal_parameters(
        mode,
        geometric_standard_deviation,
        number_concentration,
        radius_units,
        number_concentration_units,
    )
    # Preflight shape, volume, grid and material before evaluating a generator.
    result = create_particle_data_from_radii(
        radii=radius,
        concentration=0.0,
        distribution_type=distribution_type,
        density=density,
        charge=charge,
        volume=volume,
        density_units=density_units,
        volume_units=volume_units,
    )
    generator = (
        get_lognormal_pdf_distribution
        if distribution_type == "continuous_pdf"
        else get_lognormal_pmf_distribution
    )
    result.concentration[0] = generator(
        x_values=radius,
        mode=modes,
        geometric_standard_deviation=deviations,
        number_of_particles=numbers,
    )
    result.validate_representation()
    return result
