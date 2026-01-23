"""Helpers for converting particle-resolved representations to binned forms.

This module provides utilities for defining kernel bins from particle-resolved
radii and for converting a particle-resolved representation into a
SpeciatedMassMovingBin representation that is compatible with kernel-based
calculations.
"""

from copy import deepcopy
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.condensation.condensation_strategies import (
    MIN_PARTICLE_RADIUS_M,
)
from particula.particles.distribution_strategies import (
    SpeciatedMassMovingBin,
)
from particula.particles.representation import ParticleRepresentation


def get_particle_resolved_binned_radius(
    particle: ParticleRepresentation,
    bin_radius: Optional[NDArray[np.float64]] = None,
    total_bins: Optional[int] = None,
    bins_per_radius_decade: int = 10,
) -> NDArray[np.float64]:
    """Compute radius bin edges for kernel calculations.

    Args:
        particle: ParticleRepresentation used to derive radius statistics.
        bin_radius: Explicit radius bin edges in metres, if already defined.
        total_bins: Number of log-spaced bins to generate when provided.
        bins_per_radius_decade: Bin density per radius decade when
            ``total_bins`` is None.

    Returns:
        NDArray[np.float64]: Radius bin edges in metres.

    Raises:
        ValueError: When particle radii cannot be determined or are not finite.
    """
    # if the bin radius is set, return it
    if bin_radius is not None:
        return bin_radius
    # else find the non-zero min and max radii, the log space them
    particle_radius = particle.get_radius()
    min_radius = np.min(particle_radius[particle_radius > 0]) * 0.5
    max_radius = np.max(particle_radius[particle_radius > 0]) * 2
    if not np.isfinite(min_radius) or not np.isfinite(max_radius):
        raise ValueError(
            "Particle radius must be finite. Check the particles,"
            "they may all be zero and the kernel cannot be calculated."
        )
    if min_radius == 0:
        min_radius = np.float64(1e-10)
    if total_bins is not None:
        return np.logspace(
            np.log10(min_radius),
            np.log10(max_radius),
            num=total_bins,
            base=10,
            dtype=np.float64,
        )
    # else kernel bins per decade
    num = np.ceil(
        bins_per_radius_decade * np.log10(max_radius / min_radius),
    )
    return np.logspace(
        np.log10(min_radius),
        np.log10(max_radius),
        num=int(num),
        base=10,
        dtype=np.float64,
    )


def get_speciated_mass_representation_from_particle_resolved(
    particle: ParticleRepresentation,
    bin_radius: NDArray[np.float64],
) -> ParticleRepresentation:
    """Convert a particle-resolved representation into a moving-bin format.

    Args:
        particle: ParticleResolved representation to convert.
        bin_radius: Radius bin edges in metres used for grouping particles.

    Returns:
        ParticleRepresentation: Copy of the original representation with
            SpeciatedMassMovingBin strategy applied and mass/concentration
            rebinned onto the provided radii.
    """
    # deep copy the particle to avoid modifying the original
    new_particle = deepcopy(particle)
    new_particle.strategy = SpeciatedMassMovingBin()

    # add the concentration by bin_indexes
    new_concentration = np.zeros_like(bin_radius)
    old_concentration = particle.get_concentration()

    # get the radius to bin the indexes
    bin_indexes = np.digitize(particle.get_radius(), bin_radius)
    # add the distribution by bin_indexes
    old_distribution = particle.get_distribution()
    if old_distribution.ndim == 1:
        new_distribution = np.zeros_like(bin_radius)
    else:
        new_distribution = np.zeros(
            (len(bin_radius), np.shape(old_distribution)[1])
        )

    # add the charge by bin_indexes
    new_charge = np.zeros(len(bin_radius))
    old_charge = particle.get_charge()
    if np.shape(old_charge) != np.shape(old_concentration):
        aligned_charge = np.zeros_like(old_concentration)
        flat_charge = np.reshape(old_charge, -1)
        copy_len = min(flat_charge.size, aligned_charge.size)
        aligned_charge[:copy_len] = flat_charge[:copy_len]
        old_charge = aligned_charge

    # loop through the bins and get the median
    for index, _ in enumerate(bin_radius):
        mask = bin_indexes == index
        if np.any(mask):
            if old_distribution.ndim == 1:
                new_distribution[index] = np.median(old_distribution[mask])
            else:
                new_distribution[index, :] = np.mean(old_distribution[mask, :])
            new_charge[index] = np.median(old_charge[mask])
            new_concentration[index] = np.sum(old_concentration[mask])
        else:
            # Default behavior when the bin is empty:
            if old_distribution.ndim == 1:
                new_distribution[index] = np.nan
            else:
                new_distribution[index, :] = np.nan
            new_charge[index] = np.nan
            new_concentration[index] = 0

    # Replace NaNs with zeros so kernel steps see valid bins
    new_distribution = np.where(np.isnan(new_distribution), 0, new_distribution)

    new_charge = np.where(np.isnan(new_charge), 0, new_charge)
    new_concentration = np.where(
        np.isnan(new_concentration), 0, new_concentration
    )

    # Remove empty bins to keep the kernel radius grid strictly ordered
    valid_bins = new_concentration > 0
    if not np.any(valid_bins):
        valid_bins = np.ones_like(valid_bins, dtype=bool)
    if new_distribution.ndim == 1:
        new_distribution = np.maximum(
            new_distribution[valid_bins], MIN_PARTICLE_RADIUS_M
        )
    else:
        new_distribution = np.maximum(
            new_distribution[valid_bins, :], MIN_PARTICLE_RADIUS_M
        )
    new_charge = new_charge[valid_bins]
    new_concentration = new_concentration[valid_bins]

    new_particle.distribution = new_distribution
    new_particle.charge = new_charge
    new_particle.concentration = new_concentration
    return new_particle
