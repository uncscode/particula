"""Change the particle-resolved representation to a binned representation.
A binning approach is used to calculate the kernel.
This creates a simple particle representation to pass to the kernel function.
"""

from copy import deepcopy
from typing import Optional

import numpy as np
from numpy.typing import NDArray

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
    """Determine binned radii for kernel calculations.

    If bin_radius is provided, those edges are used directly. Otherwise,
    a log-spaced array is generated based on the particle's minimum and
    maximum radii and either a total number of bins or bins per radius
    decade.

    Arguments:
        - particle : The ParticleRepresentation instance for radius binning.
        - bin_radius : Optional array of radius bin edges in meters.
        - total_bins : Exact number of bins to generate, if set.
        - bins_per_radius_decade : Number of bins per decade of radius,
          used only if total_bins is None.

    Returns:
        - NDArray[np.float64] : The bin edges (radii) in meters.

    Raises:
        - ValueError : If finite radii cannot be determined for binning.
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
    """Convert a ParticleResolvedSpeciatedMass to a SpeciatedMassMovingBin.

    This function bins the mass and charge distributions for each species
    according to the provided bin_radius array, using median or mean
    values in each bin. The distribution_strategy is switched to
    SpeciatedMassMovingBin.

    Arguments:
        - particle : The ParticleRepresentation to convert.
        - bin_radius : Array of radius bin edges in meters.

    Returns:
        - ParticleRepresentation : A new representation with binned
          mass and concentration for each species.
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
        old_charge = np.zeros_like(old_concentration) + old_charge

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

    # check for nans and all zeros in the new distribution
    mask_nan_zeros = np.isnan(new_distribution) | (new_distribution == 0)

    new_charge = np.where(np.isnan(new_charge), 0, new_charge)
    new_concentration = np.where(
        np.isnan(new_concentration), 0, new_concentration
    )

    # filter out the nans and zeros
    if new_distribution.ndim == 1:
        new_particle.distribution = new_distribution[~mask_nan_zeros]
        new_particle.charge = new_charge[~mask_nan_zeros]
        new_particle.concentration = new_concentration[~mask_nan_zeros]
        return new_particle
    mask_nan_zeros = np.any(mask_nan_zeros, axis=1)
    new_particle.distribution = new_distribution[~mask_nan_zeros, :]
    new_particle.charge = new_charge[~mask_nan_zeros]
    new_particle.concentration = new_concentration[~mask_nan_zeros]
    return new_particle
