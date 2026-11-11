"""Native particle initialization without strategy-bearing facades.

Initialization allocates owned float64 arrays. Aggregate wrapping is a separate,
identity-preserving operation. Binned densities use fixed volume 1 m^3.
Canonical SI inputs do not require Pint; other unit spellings/conversions do.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from particula.particles.particle_data import ParticleData
from particula.particles.particle_data_builder import ParticleDataBuilder
from particula.particles.properties.lognormal_size_distribution import (
    get_lognormal_pdf_distribution,
    get_lognormal_pmf_distribution,
    get_lognormal_sample_distribution,
)
from particula.util.convert_units import get_unit_conversion


def _unit_factor(units: str, target: str) -> float:
    """Avoid the optional converter when inputs already use canonical units."""
    return 1.0 if units == target else float(get_unit_conversion(units, target))


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
    result = float(array.flat[0]) * _unit_factor(units, "kg/m^3")
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
        else _vector(mode, "mode") * _unit_factor(radius_units, "m")
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
        * _unit_factor(number_concentration_units, "1/m^3")
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
    volume = float(value) * _unit_factor(units, "m^3")
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
    radius = _vector(radii, "radii") * _unit_factor(radius_units, "m")
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
        ValueError: If inputs do not define a valid distribution. Discrete PMFs
            require at least three positive, strictly increasing nodes; PDFs
            require at least two. The existing PMF converter needs two
            intervals to extrapolate its last bin width.
    """
    radius = (
        np.logspace(-9, -4, 250)
        if radius_bins is None
        else _vector(radius_bins, "radius_bins")
        * _unit_factor(radius_units, "m")
    )
    minimum_nodes = 3 if distribution_type == "discrete" else 2
    if radius.size < minimum_nodes:
        raise ValueError(
            f"{distribution_type} lognormal radius grid requires at least "
            f"{minimum_nodes} nodes"
        )
    if np.any(np.diff(radius) <= 0):
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


def _species_density(
    value: ArrayLike, shape: tuple[int, int, int]
) -> NDArray[np.float64]:
    """Reduce only scalar, species or redundant population density forms."""
    array = np.asarray(value, dtype=np.float64)
    boxes, slots, species = shape
    squeezed = np.squeeze(array)
    if squeezed.ndim == 0:
        return np.full(species, float(squeezed))
    if squeezed.shape == (species,):
        return squeezed.copy()
    if (
        array.shape in ((slots, species), (boxes, slots, species))
        and array.size
    ):
        rows = array.reshape(-1, species)
        if np.all(rows == rows[0]):
            return rows[0].copy()
    raise ValueError("density must retain one value per species without loss")


def _slot_values(
    value: ArrayLike,
    shape: tuple[int, int, int],
    name: str,
    *,
    reduce_species: bool = False,
) -> NDArray[np.float64]:
    """Copy scalar/slot values; reduce redundant charge species lanes only."""
    array = np.asarray(value, dtype=np.float64)
    boxes, slots, species = shape
    target = (boxes, slots)
    if array.size == 1:
        return np.full(target, float(array.flat[0]))
    if array.shape == target:
        return array.copy()
    if array.shape in ((slots,), (1, slots), (slots, 1)):
        return np.broadcast_to(array.reshape(1, slots), target).copy()
    if reduce_species and array.shape in (
        (slots, species),
        (boxes, slots, species),
    ):
        if np.all(array == array[..., :1]):
            return np.broadcast_to(array[..., 0], target).copy()
    raise ValueError(f"{name} must retain one value per particle without loss")


def create_resolved_particle_data(
    *,
    masses: ArrayLike,
    density: ArrayLike,
    concentration: ArrayLike | None = None,
    charge: ArrayLike | None = 0.0,
    volume: ArrayLike = 1.0,
    mass_units: str = "kg",
    density_units: str = "kg/m^3",
    concentration_units: str = "count",
    volume_units: str = "m^3",
) -> ParticleData:
    """Create owned resolved data with explicit count or density input units.

    Args:
        masses: Scalar, (N,) single-species, (N,S), or (B,N,S) masses.
        density: Scalar or (S,) densities; singleton forms and identical
            population rows are accepted. Differing population rows reject.
        concentration: Scalar, (N,), or (B,N) weights; default one count/slot.
            Singleton row/column forms broadcast without dropping values.
        charge: Scalar, (N,), or (B,N) charge counts; None means zero.
            Redundant (N,S) or (B,N,S) species lanes may reduce only when equal.
            Canonical (B,N) shape takes precedence over a redundant shape.
        volume: Positive finite physical volume, scalar or (B,).
        mass_units: Per-species mass units, default kg.
        density_units: Material-density units, default kg/m^3.
        concentration_units: ``count`` preserves weights; ``1/m^3`` or
            ``1/cm^3`` explicitly converts density to counts using volume once.
            Omitted concentration requires ``count`` units.
        volume_units: Physical volume units, default m^3.

    Returns:
        Owned float64 ParticleData, explicitly particle_resolved. Empty slot
        axes and fractional/zero weights are supported. This does not authorize
        scientific algorithms that require unit weights or a single box.

    Raises:
        ValueError: If units, shapes or representation values are invalid.
            No caller arrays are changed and no species/particle lane is lost.
    """
    mass = np.asarray(masses, dtype=np.float64)
    if mass.ndim == 0:
        mass = mass.reshape(1, 1, 1)
    elif mass.ndim == 1:
        mass = mass[None, :, None]
    elif mass.ndim == 2:
        mass = mass[None, ...]
    if mass.ndim != 3 or mass.shape[0] == 0 or mass.shape[2] == 0:
        raise ValueError(
            "masses require nonempty box/species axes and rank <= 3"
        )
    shape = (mass.shape[0], mass.shape[1], mass.shape[2])
    volumes = _vector(volume, "volume") * _unit_factor(volume_units, "m^3")
    if volumes.size == 1:
        volumes = np.full(shape[0], volumes[0])
    if (
        volumes.shape != (shape[0],)
        or not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0)
    ):
        raise ValueError(
            "volume must be positive finite with one value per box"
        )
    if concentration_units not in ("count", "1/m^3", "1/cm^3"):
        raise ValueError(
            "resolved concentration units must be count or density"
        )
    if concentration is None and concentration_units != "count":
        raise ValueError("density units require explicit concentration")
    counts = _slot_values(
        1.0 if concentration is None else concentration, shape, "concentration"
    )
    if concentration_units != "count":
        with np.errstate(over="ignore", invalid="ignore"):
            counts *= _unit_factor(concentration_units, "1/m^3")
            counts *= volumes[:, None]
    builder = (
        ParticleDataBuilder()
        .set_distribution_type("particle_resolved")
        .set_masses(mass.copy(), units=mass_units)
        .set_density(_species_density(density, shape), units=density_units)
        .set_concentration(counts, units="count")
        .set_charge(
            _slot_values(
                0.0 if charge is None else charge,
                shape,
                "charge",
                reduce_species=True,
            )
        )
        .set_volume(volumes)
    )
    return builder.build().copy()


def create_lognormal_resolved_particle_data(
    *,
    particle_resolved_count: int = 20_000,
    mode: ArrayLike | None = None,
    geometric_standard_deviation: ArrayLike | None = None,
    number_concentration: ArrayLike | None = None,
    density: ArrayLike = 1000.0,
    charge: ArrayLike | None = 0.0,
    volume: float = 1.0,
    radius_units: str = "m",
    number_concentration_units: str = "1/m^3",
    density_units: str = "kg/m^3",
    volume_units: str = "m^3",
) -> ParticleData:
    """Sample the existing lognormal mixture with one count per sampled slot.

    Args:
        particle_resolved_count: Positive integer sample count, default 20,000.
        mode: Scale radii; default [100e-9, 1e-6] metres.
        geometric_standard_deviation: GSD values > 1; default [1.2, 1.4].
        number_concentration: Modal densities; default [1e10, 1e9] m^-3.
            These set relative modal sampling weights, not final number density.
        density: Single material density, default 1000 kg/m^3.
        charge: Scalar or per-particle charge; default zero.
        volume: Positive finite physical volume, default 1 m^3.
        radius_units: Units of supplied modes; omitted defaults are SI.
        number_concentration_units: Units of supplied modal densities.
        density_units: Material-density units.
        volume_units: Physical-volume units.

    Returns:
        Owned single-box data with unit count weights, hence number density
        equal to sample count divided by physical volume. The existing sampler
        and NumPy/SciPy global random-state behavior are preserved; no new seed
        policy or cross-backend replay guarantee is introduced.

    Raises:
        ValueError: If parameters, count, shape or physical values are invalid.
            Deterministic input admission precedes random sampling.
    """
    if (
        isinstance(particle_resolved_count, (bool, np.bool_))
        or not isinstance(particle_resolved_count, (int, np.integer))
        or particle_resolved_count <= 0
    ):
        raise ValueError("particle_resolved_count must be a positive integer")
    modes, deviations, numbers = _modal_parameters(
        mode,
        geometric_standard_deviation,
        number_concentration,
        radius_units,
        number_concentration_units,
    )
    material = _single_density(density, density_units)
    result = create_resolved_particle_data(
        masses=np.zeros(particle_resolved_count),
        density=material,
        charge=charge,
        volume=volume,
        volume_units=volume_units,
    )
    radii = get_lognormal_sample_distribution(
        mode=modes,
        geometric_standard_deviation=deviations,
        number_of_particles=numbers,
        number_of_samples=particle_resolved_count,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result.masses[0, :, 0] = 4.0 / 3.0 * np.pi * radii**3 * material
    result.validate_representation()
    return result
