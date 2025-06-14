"""Speciated mass moving bin strategy."""

import logging
import numpy as np
from numpy.typing import NDArray

from .base import DistributionStrategy

logger = logging.getLogger("particula")


class SpeciatedMassMovingBin(DistributionStrategy):
    """
    Strategy for particles with speciated mass distribution.

    Each particle may contain multiple species, each with a unique
    density. This strategy calculates mass, radius, and total mass from
    the species-level masses and overall particle concentrations.

    Methods:
    - get_name : Return the type of the distribution strategy.
    - get_species_mass : Calculate the mass per species.
    - get_mass : Calculate the mass of the particles or bin.
    - get_total_mass : Calculate the total mass of particles.
    - get_radius : Calculate the radius of particles.
    - add_mass : Add mass to the particle distribution.
    - add_concentration : Add concentration to the distribution.
    """

    def get_species_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return distribution

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        volumes = np.sum(distribution / density, axis=1)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(
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
        mass_per_particle = np.where(
            concentration_expand > 0, added_mass / concentration_expand, 0
        )
        new_distribution = np.maximum(distribution + mass_per_particle, 0)
        return new_distribution, concentration

    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to SpeciatedMassMovingBin, the distribution "
                "and added distribution should match."
            )
            logger.error(message)
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to SpeciatedMassMovingBin, the arrays "
                "should have the same shape."
            )
            logger.error(message)
            raise ValueError(message)
        concentration += added_concentration
        return distribution, concentration

    def collide_pairs(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        message = (
            "Colliding pairs in SpeciatedMassMovingBin is not physically meaningful"
        )
        logger.warning(message)
        raise NotImplementedError(message)
