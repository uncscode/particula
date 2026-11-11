"""Staged native CPU coagulation using the existing rate/selector algorithms.

PMF/PDF concentrations are densities at unit volume. Resolved slots carry
equal positive counts during a merge; their common multiplicity scales the
selector's duration, leaving the physical volume and raw counts unchanged.
No caller array is written until the complete proposed state is validated.
RNG draws are not rolled back if selection or proposal validation fails.
"""

import logging
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.coagulation import charged_dimensional_kernel as laws
from particula.dynamics.coagulation import charged_kernel_strategy as models
from particula.dynamics.coagulation.particle_resolved_step import (
    particle_resolved_method,
)
from particula.particles.particle_data import ParticleData

from ._native_validation import (
    positive_scalar,
    validate_controls,
    validate_particle,
    validate_rate_kernel,
)


def validate_edges(edges: Any) -> NDArray[np.float64]:
    """Require real, positive, strictly increasing kernel radius edges."""
    values = np.asarray(edges)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError("kernel radius edges must be positive and increasing")
    return np.asarray(values, dtype=np.float64)


def validate_bin_controls(
    edges: Any, total_bins: Any, bins_per_decade: Any
) -> None:
    """Validate mutable bin controls even on a zero-work step."""
    if edges is not None:
        validate_edges(edges)
    for value, name, minimum in (
        (total_bins, "total_bins", 2),
        (bins_per_decade, "bins_per_radius_decade", 1),
    ):
        if value is None and name == "total_bins":
            continue
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")


def kernel_edges(
    particle: ParticleData,
    edges: Any,
    total_bins: Any,
    bins_per_decade: Any,
) -> NDArray[np.float64]:
    """Generate the existing logarithmic grid from active native slots."""
    validate_particle(particle, "particle_resolved")
    validate_bin_controls(edges, total_bins, bins_per_decade)
    if edges is not None:
        return validate_edges(edges).copy()
    radii = particle.radii[0][particle.concentration[0] > 0]
    if not radii.size:
        raise ValueError("kernel binning requires an active particle")
    lower, upper = float(np.min(radii)) * 0.5, float(np.max(radii)) * 2
    positive_scalar(lower, "minimum kernel radius")
    positive_scalar(upper, "maximum kernel radius")
    count = total_bins
    if count is None:
        count = max(
            2,
            int(np.ceil(bins_per_decade * (np.log10(upper) - np.log10(lower)))),
        )
    return np.logspace(np.log10(lower), np.log10(upper), int(count))


def binned_kernel_data(
    particle: ParticleData, edges: NDArray[np.float64]
) -> ParticleData:
    """Build temporary count-weighted material representatives, not new state.

    Counts and species inventories are preserved across occupied bins. Charge
    retains the existing median approximation for kernel evaluation only; this
    temporary charge is never used for the authoritative merge bookkeeping.
    """
    validate_particle(particle, "particle_resolved")
    edges = validate_edges(edges)
    active = particle.concentration[0] > 0
    radii = particle.radii[0][active]
    indices = np.digitize(radii, edges)
    if np.any(indices >= edges.size):
        raise ValueError("kernel radius edges must cover every active particle")
    weights = particle.concentration[0][active]
    mass = particle.masses[0][active]
    charge = particle.charge[0][active]
    groups = np.unique(indices)
    new_mass = np.zeros((1, groups.size, particle.n_species))
    new_count = np.zeros((1, groups.size))
    new_charge = np.zeros((1, groups.size))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for lane, group in enumerate(groups):
            selected = indices == group
            count = np.sum(weights[selected])
            new_count[0, lane] = count
            # Normalize weights first, avoiding overflow in count * mass.
            new_mass[0, lane] = np.sum(
                mass[selected] * (weights[selected] / count)[:, None], axis=0
            )
            # Preserve an exactly monodisperse material point. A rounded
            # average can otherwise move its radius outside a one-point grid.
            if np.all(mass[selected] == mass[selected][0]):
                new_mass[0, lane] = mass[selected][0]
            new_charge[0, lane] = np.median(charge[selected])
    result = ParticleData(
        masses=new_mass,
        concentration=new_count,
        charge=new_charge,
        density=particle.density.copy(),
        volume=particle.volume.copy(),
        distribution_type="particle_resolved",
    )
    validate_particle(result, "particle_resolved")
    return result


def _validate_storage(particle: ParticleData, resolved: bool) -> None:
    """Reject unwritable or overlapping destinations before draws."""
    fields = [
        particle.masses,
        particle.concentration,
        particle.charge,
        particle.density,
        particle.volume,
    ]
    if particle.radius_grid is not None:
        fields.append(particle.radius_grid)
    for index in [0, 1, 2] if resolved else [1]:
        destination = fields[index]
        if not destination.flags.writeable:
            raise ValueError("native coagulation destinations must be writable")
        if any(
            np.shares_memory(destination, field)
            for other, field in enumerate(fields)
            if other != index
        ):
            raise ValueError("native coagulation fields must not alias writers")


def _direct_kernel(
    strategy: Any, particle: ParticleData, temp: float, press: float
) -> Any:
    """Retain the existing charged-model direct callback and fallback policy."""
    if not strategy.use_direct_kernel:
        return None
    from .coagulation_strategy_abc import _build_direct_kernel_index_func

    mapping: dict[Any, Any] = {
        models.HardSphereKernelStrategy: (
            laws.get_hard_sphere_kernel_via_system_state
        ),
        models.CoulombDyachkov2007KernelStrategy: (
            laws.get_coulomb_kernel_dyachkov2007_via_system_state
        ),
        models.CoulombGatti2008KernelStrategy: (
            laws.get_coulomb_kernel_gatti2008_via_system_state
        ),
        models.CoulombGopalakrishnan2012KernelStrategy: (
            laws.get_coulomb_kernel_gopalakrishnan2012_via_system_state
        ),
        models.CoulumbChahl2019KernelStrategy: (
            laws.get_coulomb_kernel_chahl2019_via_system_state
        ),
    }
    builder = mapping.get(type(getattr(strategy, "kernel_strategy", None)))
    if builder is None:
        logging.getLogger("particula").warning(
            "Direct kernel disabled for unsupported kernel strategy."
        )
        return None
    return _build_direct_kernel_index_func(
        particle.radii[0],
        particle.total_mass[0],
        particle.charge[0],
        builder,
        temp,
        press,
    )


def _merge(
    particle: ParticleData, active: NDArray[np.int64], pairs: Any
) -> None:
    """Validate selector output, stage the merge, then publish in place."""
    pairs = np.asarray(pairs)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or not np.issubdtype(pairs.dtype, np.integer)
        or np.any(pairs < 0)
        or np.any(pairs >= active.size)
    ):
        raise ValueError("coagulation selector must return valid integer pairs")
    # A self pair is a no-op, including the root left by chain resolution.
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not pairs.size:
        return
    donor, recipient = active[pairs[:, 0]], active[pairs[:, 1]]
    if (
        np.unique(donor).size != donor.size
        or np.intersect1d(donor, recipient).size
    ):
        raise ValueError(
            "coagulation selector must return resolved merge pairs"
        )
    masses, counts, charge = (
        particle.masses.copy(),
        particle.concentration.copy(),
        particle.charge.copy(),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        np.add.at(masses[0], recipient, masses[0, donor].copy())
        np.add.at(charge[0], recipient, charge[0, donor].copy())
    masses[0, donor] = 0
    counts[0, donor] = 0
    charge[0, donor] = 0
    candidate = ParticleData(
        masses=masses,
        concentration=counts,
        charge=charge,
        density=particle.density,
        volume=particle.volume,
        distribution_type="particle_resolved",
    )
    validate_particle(candidate, "particle_resolved")
    np.copyto(particle.masses, masses)
    np.copyto(particle.concentration, counts)
    np.copyto(particle.charge, charge)


def _density_step(
    strategy: Any,
    particle: ParticleData,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleData:
    """Stage the existing explicit density update without clipping losses."""
    rate = np.asarray(strategy.net_rate(particle, temperature, pressure))
    if (
        rate.shape != (particle.n_particles,)
        or not np.issubdtype(rate.dtype, np.number)
        or np.iscomplexobj(rate)
        or not np.all(np.isfinite(rate))
    ):
        raise ValueError("native coagulation rate must have finite slot shape")
    if time_step == 0:
        return particle
    with np.errstate(over="ignore", invalid="ignore"):
        updated = particle.concentration + rate[None, :] * time_step
    if not np.all(np.isfinite(updated)) or np.any(updated < 0):
        raise ValueError("coagulation step requires a smaller finite time_step")
    np.copyto(particle.concentration, updated)
    return particle


def step_native(
    strategy: Any,
    particle: ParticleData,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleData:
    """Apply an admitted native step without changing process/RNG ownership."""
    kind = strategy.distribution_type
    validate_particle(particle, kind)
    positive_scalar(temperature, "temperature")
    positive_scalar(pressure, "pressure")
    positive_scalar(time_step, "time_step", zero_allowed=True)
    validate_controls(strategy, kind)
    resolved = kind == "particle_resolved"
    _validate_storage(particle, resolved)
    if not resolved:
        return _density_step(
            strategy, particle, temperature, pressure, time_step
        )
    validate_bin_controls(
        strategy.particle_resolved_radius,
        strategy.particle_resolved_bins_number,
        strategy.particle_resolved_bins_per_decade,
    )
    if not isinstance(strategy.use_direct_kernel, (bool, np.bool_)):
        raise ValueError("use_direct_kernel must be Boolean")
    if not isinstance(strategy.random_generator, np.random.Generator):
        raise ValueError("random_generator must be a NumPy Generator")
    active = np.flatnonzero(particle.concentration[0] > 0)
    weights = particle.concentration[0, active]
    if weights.size and np.any(weights != weights[0]):
        raise ValueError(
            "resolved coagulation requires equal positive active weights"
        )
    if active.size < 2:
        return particle
    scaled_time = float(time_step) * float(weights[0])
    positive_scalar(scaled_time, "weighted time_step", zero_allowed=True)
    material = ParticleData(
        masses=particle.masses[:, active].copy(),
        concentration=particle.concentration[:, active].copy(),
        charge=particle.charge[:, active].copy(),
        density=particle.density.copy(),
        volume=particle.volume.copy(),
        distribution_type="particle_resolved",
    )
    edges = kernel_edges(
        material,
        strategy.particle_resolved_radius,
        strategy.particle_resolved_bins_number,
        strategy.particle_resolved_bins_per_decade,
    )
    binned = binned_kernel_data(material, edges)
    kernel = validate_rate_kernel(
        binned, strategy.kernel(binned, temperature, pressure)
    )
    radii = binned.radii[0]
    order = np.argsort(radii)
    radii, kernel = radii[order], kernel[np.ix_(order, order)]
    if np.any(np.diff(radii) <= 0):
        raise ValueError("binned kernel radii must be strictly increasing")
    direct = _direct_kernel(strategy, material, temperature, pressure)
    if time_step == 0:
        return particle
    point_kernel: Any = None
    if radii.size == 1 and direct is None:
        # The linear interpolator has no finite 2D single-point value. Define
        # its value at that point only; retain zero outside the tabulated range.
        def at_point(small: Any, large: Any) -> Any:
            return np.where(
                (small == radii[0]) & (large == radii[0]), kernel[0, 0], 0.0
            )

        point_kernel = at_point

    pairs = particle_resolved_method.get_particle_resolved_coagulation_step(
        particle_radius=material.radii[0],
        kernel=kernel,
        kernel_radius=radii,
        volume=float(particle.volume[0]),
        time_step=scaled_time,
        random_generator=strategy.random_generator,
        kernel_func=point_kernel,
        kernel_index_func=direct,
    )
    _merge(particle, active, pairs)
    return particle
