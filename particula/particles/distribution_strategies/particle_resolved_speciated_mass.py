"""Particle resolved speciated mass strategy."""

import logging
import numpy as np
from numpy.typing import NDArray

from .base import DistributionStrategy

logger = logging.getLogger("particula")


class ParticleResolvedSpeciatedMass(DistributionStrategy):
    """
    Strategy for particle-resolved masses with multiple species.

    Allows each particle to have separate masses for each species, with
    individualized densities. This strategy provides a more detailed
    approach when each particle's composition must be modeled explicitly.

    Methods:
    - get_name : Return the type of the distribution strategy.
    - get_species_mass : Calculate the mass per species.
    - get_mass : Calculate the mass of the particles or bin.
    - get_total_mass : Calculate the total mass of particles.
    - get_radius : Calculate the radius of particles.
    - add_mass : Add mass to the particle distribution.
    - add_concentration : Add concentration to the distribution.
    - collide_pairs : Perform collision logic on specified particle pairs.
    """

    def get_species_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return distribution

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if distribution.ndim == 1:
            volumes = distribution / density
        else:
            volumes = np.sum(distribution / density, axis=1)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(  # pylint: disable=R0801
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if distribution.ndim == 2:
            concentration_expand = concentration[:, np.newaxis]
        else:
            concentration_expand = concentration
        new_mass = np.divide(
            np.maximum(distribution * concentration_expand + added_mass, 0),
            concentration_expand,
            out=np.zeros_like(distribution),
            where=concentration_expand != 0,
        )
        if new_mass.ndim == 1:
            new_mass_sum = np.sum(new_mass)
        else:
            new_mass_sum = np.sum(new_mass, axis=1)
        concentration = np.where(new_mass_sum > 0, concentration, 0)
        return new_mass, concentration

    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rescaled = False
        if np.all(added_concentration == 1):
            rescaled = True
        max_concentration = np.max(concentration)
        if np.allclose(added_concentration, max_concentration, atol=1e-2) or np.all(
            concentration == 0
        ):
            if max_concentration > 0:
                added_concentration = added_concentration / max_concentration
            rescaled = True
        if not rescaled:
            message = (
                "When adding concentration to ParticleResolvedSpeciatedMass, the added "
                "concentration should be all ones or all the same value."
            )
            logger.error(message)
            raise ValueError(message)

        concentration = np.divide(
            concentration,
            concentration,
            out=np.zeros_like(concentration),
            where=concentration != 0,
        )

        empty_bins = np.flatnonzero(concentration == 0)
        empty_bins_count = len(empty_bins)
        added_bins_count = len(added_concentration)
        if empty_bins_count >= added_bins_count:
            distribution[empty_bins] = added_distribution
            concentration[empty_bins] = added_concentration
            return distribution, concentration
        if empty_bins_count > 0:
            distribution[empty_bins] = added_distribution[:empty_bins_count]
            concentration[empty_bins] = added_concentration[:empty_bins_count]
        distribution = np.concatenate(
            (distribution, added_distribution[empty_bins_count:]), axis=0
        )
        concentration = np.concatenate(
            (concentration, added_concentration[empty_bins_count:]), axis=0
        )
        return distribution, concentration

    def collide_pairs(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        small_index = indices[:, 0]
        large_index = indices[:, 1]
        if distribution.ndim == 1:
            distribution[large_index] += distribution[small_index]
            distribution[small_index] = 0
            concentration[small_index] = 0
            return distribution, concentration
        distribution[large_index, :] += distribution[small_index, :]
        distribution[small_index, :] = 0
        concentration[small_index] = 0
        return distribution, concentration
