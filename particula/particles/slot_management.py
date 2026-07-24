"""Discover fixed particle slots without modifying particle data.

This module defines the CPU reference classification for active and free
particle-resolved slots. It deliberately exposes only fixed-shape discovery
diagnostics so callers can preserve stable particle storage.
"""

import numpy as np
from numpy.typing import NDArray

from particula.particles.particle_data import ParticleData


def get_slot_diagnostics(
    data: ParticleData,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Return fixed-shape diagnostics for active and free particle slots.

    A slot is active when its concentration is positive and finite, each mass
    lane is finite and nonnegative, and its total mass is positive and finite.
    A slot is free when its concentration, all mass lanes, and its charge are
    exactly zero. All other slot states are invalid.

    Args:
        data: Fixed-shape particle data to inspect without mutation.

    Returns:
        A tuple containing free indices with ``-1`` tails, active counts, and
        free counts. All arrays are newly allocated ``np.int32`` arrays.

    Raises:
        ValueError: If any particle slot has an invalid state.
    """
    masses = data.masses
    concentration = data.concentration
    charge = data.charge

    mass_valid = np.all(np.isfinite(masses) & (masses >= 0.0), axis=-1)
    mass_zero = np.all(masses == 0.0, axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        total_mass = np.sum(masses, axis=-1)

    active = (
        np.isfinite(concentration)
        & (concentration > 0.0)
        & mass_valid
        & np.isfinite(total_mass)
        & (total_mass > 0.0)
        & np.isfinite(charge)
    )
    free = (
        (concentration == 0.0)
        & mass_zero
        & (charge == 0.0)
        & np.isfinite(charge)
    )

    if np.any(~(active | free)):
        raise ValueError("Invalid particle slot state.")

    free_indices = np.full((data.n_boxes, data.n_particles), -1, dtype=np.int32)
    free_counts = np.sum(free, axis=1, dtype=np.int32)
    active_counts = np.sum(active, axis=1, dtype=np.int32)
    particle_indices = np.arange(data.n_particles, dtype=np.int32)

    for box_index in range(data.n_boxes):
        count = free_counts[box_index]
        free_indices[box_index, :count] = particle_indices[free[box_index]]

    return free_indices, active_counts, free_counts
