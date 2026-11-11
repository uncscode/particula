"""Native CPU wall-loss admission and staged updates, without facade state."""

from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from particula.particles.particle_data import ParticleData


def _scalar(value: Any, name: str, *, positive: bool = False) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or (positive and value <= 0)
    ):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}real scalar")


def _controls(strategy: Any) -> None:
    """Revalidate mutable concrete geometry and field configuration."""
    _scalar(
        strategy.wall_eddy_diffusivity, "wall_eddy_diffusivity", positive=True
    )
    geometry = getattr(strategy, "chamber_geometry", None)
    if geometry is not None and geometry not in ("spherical", "rectangular"):
        raise ValueError("unsupported wall-loss chamber_geometry")
    if geometry == "spherical" or (
        geometry is None and hasattr(strategy, "chamber_radius")
    ):
        _scalar(strategy.chamber_radius, "chamber_radius", positive=True)
    if geometry == "rectangular" or (
        geometry is None and hasattr(strategy, "chamber_dimensions")
    ):
        dimensions = strategy.chamber_dimensions
        if (
            not isinstance(dimensions, (tuple, list, np.ndarray))
            or np.ndim(dimensions) != 1
            or len(dimensions) != 3
        ):
            raise ValueError(
                "chamber_dimensions must have three positive lengths"
            )
        for dimension in dimensions:
            _scalar(dimension, "chamber_dimensions", positive=True)
    if hasattr(strategy, "wall_potential"):
        _scalar(strategy.wall_potential, "wall_potential")
        field = strategy.wall_electric_field
        if isinstance(field, (tuple, list, np.ndarray)):
            if np.ndim(field) != 1 or len(field) != 3:
                raise ValueError(
                    "wall_electric_field must be scalar or length three"
                )
            for component in field:
                _scalar(component, "wall_electric_field")
        else:
            _scalar(field, "wall_electric_field")


def validate(
    strategy: Any, particle: ParticleData, temperature: float, pressure: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Check current native storage, physics, representation and controls."""
    particle.validate_representation()
    if particle.distribution_type not in (
        "discrete",
        "continuous_pdf",
        "particle_resolved",
    ):
        raise ValueError("native wall loss requires explicit distribution_type")
    if particle.n_boxes != 1:
        raise ValueError("ParticleData must have n_boxes=1 for CPU wall loss")
    if strategy.distribution_type != particle.distribution_type:
        raise ValueError("wall-loss distribution_type must match particle data")
    _scalar(temperature, "temperature", positive=True)
    _scalar(pressure, "pressure", positive=True)
    _controls(strategy)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mass = particle.masses.sum(axis=2)
        material_volume = (particle.masses / particle.density).sum(axis=2)
        population = particle.concentration / particle.volume[:, None]
        radius = particle.radii[0]
        density = particle.effective_density[0]
    if not all(
        np.all(np.isfinite(value))
        for value in (mass, material_volume, population, radius, density)
    ):
        raise ValueError("native wall-loss derived quantities must be finite")
    if np.any((mass > 0) & (material_volume <= 0)):
        raise ValueError("positive mass requires representable particle volume")
    if np.any((particle.concentration > 0) & (mass <= 0)):
        raise ValueError("occupied wall-loss particles require positive mass")
    return radius, density


def coefficient(
    strategy: Any,
    particle: ParticleData,
    temperature: float,
    pressure: float,
    *,
    include_inactive: bool = False,
) -> NDArray[np.float64]:
    """Apply existing coefficient laws to native material and charge arrays."""
    radius, density = validate(strategy, particle, temperature, pressure)
    selected = radius > 0
    if (
        particle.distribution_type == "particle_resolved"
        and not include_inactive
    ):
        selected &= particle.concentration[0] > 0
    result = np.zeros(particle.n_particles, dtype=np.float64)
    if not np.any(selected):
        return result
    if hasattr(strategy, "compute_coefficient_from_arrays"):
        values = strategy.compute_coefficient_from_arrays(
            radius[selected],
            density[selected],
            particle.charge[0, selected],
            temperature,
            pressure,
        )
    else:
        values = strategy.loss_coefficient_for_particles(
            radius[selected], density[selected], temperature, pressure
        )
    values = np.asarray(values)
    if (
        values.shape != radius[selected].shape
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
    ):
        raise ValueError(
            "wall-loss coefficient must be finite nonnegative slot data"
        )
    result[selected] = values
    return result


def rate(
    strategy: Any, particle: ParticleData, temperature: float, pressure: float
) -> NDArray[np.float64]:
    """Return number-density (or dn/dr) loss per second."""
    values = coefficient(strategy, particle, temperature, pressure)
    with np.errstate(over="ignore", invalid="ignore"):
        result = -values * (particle.concentration[0] / particle.volume[0])
    if not np.all(np.isfinite(result)):
        raise ValueError("native wall-loss rate must remain finite")
    return result


def _storage(strategy: Any, particle: ParticleData, resolved: bool) -> None:
    fields = [
        particle.concentration,
        particle.masses,
        particle.charge,
        particle.density,
        particle.volume,
    ]
    if particle.radius_grid is not None:
        fields.append(particle.radius_grid)
    for name in ("chamber_dimensions", "wall_electric_field"):
        control = getattr(strategy, name, None)
        if isinstance(control, np.ndarray):
            fields.append(control)
    for index in (0, 1) if resolved else (0,):
        target = fields[index]
        if not target.flags.writeable:
            raise ValueError("native wall-loss destinations must be writable")
        if any(
            np.shares_memory(target, other)
            for other_index, other in enumerate(fields)
            if other_index != index
        ):
            raise ValueError(
                "native wall-loss destination aliases protected storage"
            )


def step(
    strategy: Any,
    particle: ParticleData,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleData:
    """Stage the existing CPU deterministic or Bernoulli removal law.

    Resolved removal clears selected masses/counts and retains charge storage,
    including inactive lanes, as on the existing CPU path. GPU slot clearing
    and its random lifecycle are separate contracts.
    """
    radius, density = validate(strategy, particle, temperature, pressure)
    _scalar(time_step, "time_step")
    if time_step < 0:
        raise ValueError("time_step must be nonnegative")
    resolved = particle.distribution_type == "particle_resolved"
    _storage(strategy, particle, resolved)
    charged = hasattr(strategy, "compute_coefficient_from_arrays")
    # Neutral CPU helper draws at positive-radius slots, including unoccupied
    # material carriers. Charged CPU removal draws only at occupied slots.
    values = coefficient(
        strategy,
        particle,
        temperature,
        pressure,
        include_inactive=resolved and not charged,
    )
    if resolved and not callable(
        getattr(strategy.random_generator, "binomial", None)
    ):
        raise ValueError(
            "wall-loss random_generator must support binomial draws"
        )
    if time_step == 0 or not np.any(particle.concentration):
        return particle
    if not resolved:
        with np.errstate(over="ignore", invalid="ignore"):
            updated = (
                particle.concentration
                - values[None] * particle.concentration * time_step
            )
        if not np.all(np.isfinite(updated)) or np.any(updated < 0):
            raise ValueError(
                "wall-loss step requires a smaller finite time_step"
            )
        np.copyto(particle.concentration, updated)
        return particle
    active = radius > 0
    if charged:
        active &= particle.concentration[0] > 0
        with np.errstate(over="ignore"):
            probability = np.exp(-values[active] * time_step)
        survived = np.zeros(particle.n_particles)
        survived[active] = strategy.random_generator.binomial(
            n=1, p=probability
        )
    else:
        from .wall_loss_strategies import get_particle_resolved_wall_loss_step

        with np.errstate(over="ignore"):
            survived = get_particle_resolved_wall_loss_step(
                radius,
                density,
                particle.concentration[0] / particle.volume[0],
                lambda r, d: values[active],
                time_step,
                strategy.random_generator,
            )
    survived = np.asarray(survived)
    if survived.shape != (particle.n_particles,) or not np.all(
        (survived == 0) | (survived == 1)
    ):
        raise ValueError("wall-loss draws must return binary slot values")
    updated = particle.concentration * survived[None]
    masses = particle.masses.copy()
    lost = (particle.concentration[0] > 0) & (survived == 0)
    masses[0, lost] = 0
    np.copyto(particle.masses, masses)
    np.copyto(particle.concentration, updated)
    return particle
