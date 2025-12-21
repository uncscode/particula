"""Mass-based moving bin strategy."""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import DistributionStrategy

logger = logging.getLogger("particula")


class MassBasedMovingBin(DistributionStrategy):
    """Strategy for particles represented by their mass distribution.

    Calculates particle mass, radius, and total mass based on the
    particle mass, number concentration, and density. This moving-bin
    approach adjusts mass bins on mass addition events.

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
        """Calculate the mass per species for the distribution.

        Returns:
            Mass per species array.
        """
        return distribution

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate particle radius from mass and density.

        Returns:
            Particle radius in meters.
        """
        volumes = distribution / density
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Add mass to the particle distribution.

        Returns:
            Updated distribution and concentration arrays.
        """
        return distribution + added_mass, concentration

    def add_concentration(  # pylint: disable=R0801
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
                "When adding concentration to MassBasedMovingBin, "
                "distribution and added distribution should match."
            )
            logger.error(message)
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to MassBasedMovingBin, the arrays "
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

        This method is not implemented for MassBasedMovingBin because particle
        pair collisions are not physically meaningful for bin-based strategies
        where particles are represented by fixed mass bins with concentrations.

        Arguments:
            - distribution : The mass distribution array.
            - concentration : The concentration array.
            - density : The density array.
            - indices : Collision pair indices array of shape (K, 2).
            - charge : Optional charge array (unused in this strategy).

        Raises:
            NotImplementedError: Always raised as method is not applicable.
        """
        message = (
            "Colliding pairs in MassBasedMovingBin is not physically meaningful"
        )
        logger.warning(message)
        raise NotImplementedError(message)
