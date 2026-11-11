"""Read-only admission for explicitly tagged native coagulation state.

Legacy untagged/facade inputs keep their existing behavior. Kernel evaluation
uses a private material-only view for positive-mass lanes, then restores the
full slot matrix with zero rows/columns for zero-mass lanes. This does not
select collisions, mutate a population, or assign it a different representation.
"""

from functools import wraps
from numbers import Real
from typing import Any, Callable, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from particula.particles.particle_data import ParticleData

_KernelMethod = TypeVar("_KernelMethod", bound=Callable[..., Any])


def is_native(particle: Any) -> bool:
    """Identify explicit metadata without inferring legacy meaning."""
    return (
        isinstance(particle, ParticleData)
        and particle.distribution_type is not None
    )


def positive_scalar(
    value: Any, name: str, *, zero_allowed: bool = False
) -> None:
    """Require a finite physical scalar, excluding boolean/array coercions."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or (value < 0 if zero_allowed else value <= 0)
    ):
        qualifier = "nonnegative" if zero_allowed else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} real scalar")


def validate_particle(particle: ParticleData, distribution_type: str) -> None:
    """Check current representation, layout, physics and strategy agreement."""
    particle.validate_representation()
    if particle.n_boxes != 1:
        raise ValueError(
            "ParticleData must have n_boxes=1 for CPU coagulation strategies"
        )
    if distribution_type != particle.distribution_type:
        raise ValueError(
            "coagulation strategy distribution_type must match particle data"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mass = np.sum(particle.masses, axis=2)
        material_volume = np.sum(particle.masses / particle.density, axis=2)
        density = particle.concentration / particle.volume[:, None]
    if not all(
        np.all(np.isfinite(values))
        for values in (mass, material_volume, density)
    ):
        raise ValueError("native coagulation derived quantities must be finite")
    if np.any((mass > 0) & (material_volume <= 0)):
        raise ValueError(
            "positive mass must have representable particle volume"
        )
    if np.any((particle.concentration > 0) & (mass == 0)):
        raise ValueError("occupied coagulation particles require positive mass")


def validate_controls(  # noqa: C901
    strategy: Any, kind: str, visiting: set[int] | None = None
) -> None:
    """Revalidate mutable mechanism controls and composed distribution modes."""
    if strategy.distribution_type != kind:
        raise ValueError(
            "combined coagulation distributions must match native data"
        )
    for name in ("fluid_density", "reynolds_lambda", "turbulent_dissipation"):
        if hasattr(strategy, name):
            positive_scalar(
                getattr(strategy, name),
                name,
                zero_allowed=name == "turbulent_dissipation"
                and not hasattr(strategy, "reynolds_lambda"),
            )
    if hasattr(strategy, "relative_velocity"):
        positive_scalar(
            strategy.relative_velocity, "relative_velocity", zero_allowed=True
        )
    if hasattr(strategy, "kernel_strategy"):
        from particula.dynamics.coagulation.charged_kernel_strategy import (
            ChargedKernelStrategyABC,
        )

        if not isinstance(strategy.kernel_strategy, ChargedKernelStrategyABC):
            raise ValueError(
                "kernel_strategy must be a charged kernel strategy"
            )
    if hasattr(strategy, "strategies"):
        children = strategy.strategies
        if not isinstance(children, (list, tuple)) or not children:
            raise ValueError(
                "combined coagulation requires nonempty strategies"
            )
        visiting = set() if visiting is None else visiting
        if id(strategy) in visiting:
            raise ValueError("combined coagulation must not contain a cycle")
        visiting.add(id(strategy))
        for child in children:
            if not callable(getattr(child, "kernel", None)) or not hasattr(
                child, "distribution_type"
            ):
                raise ValueError(
                    "combined coagulation requires compatible strategies"
                )
            validate_controls(child, kind, visiting)
        visiting.remove(id(strategy))


def native_kernel(method: _KernelMethod) -> _KernelMethod:
    """Admit native state and retain the full kernel slot layout."""

    @wraps(method)
    def evaluate(
        self: Any, particle: Any, temperature: float, pressure: float
    ) -> Any:
        if not is_native(particle):
            return method(self, particle, temperature, pressure)
        validate_particle(particle, self.distribution_type)
        positive_scalar(temperature, "temperature")
        positive_scalar(pressure, "pressure")
        validate_controls(self, particle.distribution_type)
        indices = np.flatnonzero(np.sum(particle.masses[0], axis=1) > 0)
        matrix = np.zeros(
            (particle.n_particles, particle.n_particles), dtype=np.float64
        )
        if indices.size == 0:
            return matrix
        # This view is only a material argument to the existing kernel law.
        # Admission above owns population semantics; no density or weight is
        # inferred from this temporary untagged, non-authoritative carrier.
        material = ParticleData(
            masses=particle.masses[:, indices, :],
            concentration=particle.concentration[:, indices],
            charge=particle.charge[:, indices],
            density=particle.density,
            volume=particle.volume,
        )
        values = np.asarray(
            method(self, material, temperature, pressure), dtype=np.float64
        )
        if (
            values.shape != (indices.size, indices.size)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError(
                "native coagulation kernel must be finite, nonnegative "
                "and square"
            )
        matrix[np.ix_(indices, indices)] = values
        return matrix

    return cast(_KernelMethod, evaluate)


def validate_rate_kernel(
    particle: ParticleData, kernel: Any
) -> NDArray[np.float64]:
    """Require one finite nonnegative pair value per original slot pair."""
    values = np.asarray(kernel)
    if (
        values.shape != (particle.n_particles, particle.n_particles)
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
    ):
        raise ValueError(
            "native coagulation kernel must be a real square matrix"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            "native coagulation kernel must be finite and nonnegative"
        )
    return np.asarray(values, dtype=np.float64)


def rate_radius(particle: ParticleData) -> NDArray[np.float64]:
    """Use the explicit PDF coordinate, otherwise the physical particle size."""
    if particle.distribution_type == "continuous_pdf":
        if particle.radius_grid is None:
            raise ValueError("native PDF requires radius_grid")
        return particle.radius_grid
    return particle.radii[0]
