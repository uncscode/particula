"""Radii-based moving bin strategy."""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import DistributionStrategy

logger = logging.getLogger("particula")


class RadiiBasedMovingBin(DistributionStrategy):
    """Strategy for particles represented by their radius distribution.

    Calculates particle mass, radius, and total mass based on particle
    radius, number concentration, and density. This moving-bin approach
    recalculates radii when mass is added.

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
        """Calculate the mass per species from radius and density.

        Returns:
            Mass per species array.
        """
        volumes = 4 / 3 * np.pi * distribution**3
        return volumes * density

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return particle radius from the distribution.

        Returns:
            Particle radius in meters.
        """
        return distribution

    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Add mass to the particle distribution and update radii.

        Returns:
            Updated distribution (radii) and concentration arrays.
        """
        mass_per_particle = np.where(
            concentration > 0, added_mass / concentration, 0
        )
        initial_volumes = (4 / 3) * np.pi * np.power(distribution, 3)
        new_volumes = initial_volumes + mass_per_particle / density
        new_radii = np.power(3 * new_volumes / (4 * np.pi), 1 / 3)
        return new_radii, concentration

    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
        charge: Optional[NDArray[np.float64]] = None,
        added_charge: Optional[NDArray[np.float64]] = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        Optional[NDArray[np.float64]],
    ]:
        """Add concentration to the distribution.

        Returns:
            Updated distribution, concentration, and charge arrays
            (charge unchanged for this strategy).
        """
        if (distribution.shape != added_distribution.shape) or (
            not np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to RadiiBasedMovingBin, "
                "distribution and added distribution should match."
            )
            logger.error(message)
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to RadiiBasedMovingBin, the arrays "
                "should have the same shape."
            )
            logger.error(message)
            raise ValueError(message)
        concentration += added_concentration
        return distribution, concentration, charge

    def collide_pairs(  # pylint: disable=too-many-positional-arguments
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
        charge: Optional[NDArray[np.float64]] = None,
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]
    ]:
        """Collide particle pairs (not implemented for this strategy).

        This method is not implemented for RadiiBasedMovingBin because particle
        pair collisions are not physically valid for bin-based strategies
        where particles are represented by fixed radius bins with
        concentrations.

        Arguments:
            - distribution : The radius distribution array.
            - concentration : The concentration array.
            - density : The density array.
            - indices : Collision pair indices array of shape (K, 2).
            - charge : Optional charge array (unused in this strategy).

        Raises:
            NotImplementedError: Always raised as method is not applicable.
        """
        message = "Colliding pairs in RadiiBasedMovingBin not physically valid"
        logger.warning(message)
        raise NotImplementedError(message)
