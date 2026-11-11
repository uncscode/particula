"""Staged native condensation inventories and storage preflight.

Particle changes are per-particle kg. Weights are already representation-aware
number densities, including radius quadrature for PDFs. Gas stays in its full
ordered kg/m3 storage. No helper here mutates caller-owned arrays.
"""

from math import fsum
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from particula.gas.gas_data import GasData
from particula.particles.particle_data import ParticleData


def validate_native_time_step(time_step: float) -> None:
    """Require a finite scalar duration, excluding Boolean control values."""
    if (
        isinstance(time_step, (bool, np.bool_))
        or not isinstance(time_step, Real)
        or not np.isfinite(time_step)
        or time_step < 0
    ):
        raise ValueError("native time_step must be a finite nonnegative scalar")


def validate_native_step_storage(
    particle: ParticleData, gas: GasData, update_gases: bool
) -> None:
    """Reject read-only destinations and aliases with any protected field."""
    fields = [
        particle.masses,
        particle.concentration,
        particle.charge,
        particle.density,
        particle.volume,
        gas.concentration,
        gas.molar_mass,
        gas.partitioning,
    ]
    if particle.radius_grid is not None:
        fields.append(particle.radius_grid)
    writers = [0, 5] if update_gases else [0]
    if particle.distribution_type == "particle_resolved":
        writers.extend([1, 2])
    for index in writers:
        destination = fields[index]
        if not destination.flags.writeable:
            raise ValueError(
                "native condensation destinations must be writable"
            )
        if any(
            np.shares_memory(destination, field)
            for other, field in enumerate(fields)
            if other != index
        ):
            raise ValueError(
                "native condensation fields must not alias writers"
            )


def retire_evaporated_slots(particle: ParticleData) -> None:
    """Release fully evaporated resolved slots after inventory accounting.

    The particle no longer exists: clear its count and charge, preserving
    fixed storage and every surviving particle. No gas-phase charge model is
    implied. Density-grid populations retain their prescribed bin semantics.
    """
    if particle.distribution_type == "particle_resolved":
        empty = np.all(particle.masses == 0, axis=2) & (
            particle.concentration > 0
        )
        particle.concentration[empty] = 0
        particle.charge[empty] = 0


def _weighted_sum(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    """Sum species inventory independently of cancellation/reduction order."""
    with np.errstate(over="raise", invalid="raise"):
        products = values * weights
    if not np.all(np.isfinite(products)):
        raise ValueError("native condensation inventory must remain finite")
    try:
        result = fsum(products)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "native condensation inventory must remain finite"
        ) from error
    if not np.isfinite(result):
        raise ValueError("native condensation inventory must remain finite")
    return result


def _finalize_species(
    masses: NDArray[np.float64],
    weights: NDArray[np.float64],
    proposal: NDArray[np.float64],
    gas_concentration: float,
) -> tuple[NDArray[np.float64], float]:
    """Limit evaporation first, then uptake against gas plus actual release."""
    delta = np.where(weights > 0, np.maximum(proposal, -masses), 0.0)
    evaporating = delta < 0
    evaporated_mass = np.where(
        evaporating, np.maximum(masses + delta, 0), masses
    )
    release = -_weighted_sum(evaporated_mass - masses, weights)
    uptake = np.maximum(delta, 0)
    demand = _weighted_sum(uptake, weights)
    available = gas_concentration + release
    if not np.isfinite(available):
        raise ValueError(
            "native condensation available inventory must be finite"
        )
    if demand > available:
        scale = np.nextafter(available / demand, 0.0)
        uptake *= max(scale, 0.0)
    updated = evaporated_mass + uptake
    if not np.all(np.isfinite(updated)):
        raise ValueError("native condensation masses must remain finite")
    # Couple actual representable mass changes, not pre-addition proposals.
    # At a gas limit a rounded particle addition can overshoot by an ulp.
    # Move only positive additions toward their old masses before coupling.
    for _ in range(4):
        transferred = _weighted_sum(updated - masses, weights)
        if transferred <= gas_concentration:
            return updated, gas_concentration - transferred
        gaining = updated > masses
        updated[gaining] = np.nextafter(updated[gaining], masses[gaining])
    raise ValueError(
        "native condensation transfer is not representable conservatively"
    )


def finalize_native_transfer(
    masses: NDArray[np.float64],
    weights: NDArray[np.float64],
    gas_concentration: NDArray[np.float64],
    proposal: NDArray[np.float64],
    sources: NDArray[np.intp],
    targets: NDArray[np.intp],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return staged particle/gas states for a validated explicit lane map.

    Each proposal is per-particle mass change in kg, already multiplied by
    its local timestep. Zero-weight nodes, unmapped particle material, and all
    unmapped gas remain unchanged. Actual per-particle evaporation is clipped
    before its release is made available to uptake. This avoids crediting a
    request to evaporate more mass than a particle owns.

    Args:
        masses: Validated nonnegative particle masses, shape (nodes, species).
        weights: Validated inventory density per node, shape (nodes,).
        gas_concentration: Validated full ordered gas concentration, kg/m3.
        proposal: Finite per-particle changes with exactly the mass-array shape.
        sources: Participating full-gas indices from native state admission.
        targets: Matching unique particle indices from native state admission.

    Returns:
        Fresh particle masses and full gas concentration; input arrays retain
        their values and identities. The caller decides whether to commit gas
        changes or retain its externally maintained reservoir.

    Raises:
        ValueError: If proposals have invalid shape/values or a transfer cannot
            be represented conservatively with finite floating-point storage.
    """
    if proposal.shape != masses.shape or not np.all(np.isfinite(proposal)):
        raise ValueError("native mass proposal must have finite particle shape")
    new_masses = masses.copy()
    new_gas = gas_concentration.copy()
    for source, target in zip(sources, targets, strict=True):
        try:
            with np.errstate(over="raise", invalid="raise"):
                new_masses[:, target], new_gas[source] = _finalize_species(
                    masses[:, target],
                    weights,
                    proposal[:, target],
                    float(gas_concentration[source]),
                )
        except FloatingPointError as error:
            raise ValueError(
                "native condensation inventory overflow"
            ) from error
    return new_masses, new_gas
