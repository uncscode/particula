"""Particle Distribution Strategies

Defines various distribution strategies (mass-based, radii-based, etc.)
for representing particles in a distribution. Each strategy handles
mass, radius, total mass, and concentration updates differently.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class DistributionStrategy(ABC):
    """Abstract base class defining common interfaces for
    mass, radius, and total mass calculations across
    different particle distribution representations.

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

    def get_name(self) -> str:
        """Return the type of the distribution strategy."""
        return self.__class__.__name__

    @abstractmethod
    def get_species_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Return the mass per species in the distribution.

        Arguments:
            - distribution : The distribution of particle sizes or masses.
            - density : The density of the particles.

        Returns:
            - The mass of the particles (per species).
        """

    def get_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculate the mass of the particles or bin.

        Arguments:
            - distribution : The distribution of particle sizes or masses.
            - density : The density of the particles.

        Returns:
            - The mass of the particles.
        """
        if distribution.ndim == 1:
            return self.get_species_mass(distribution, density)
        return np.sum(self.get_species_mass(distribution, density), axis=1)

    def get_total_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> np.float64:
        """
        Calculate the total mass of all particles (or bin).

        Arguments:
            - distribution : The distribution of particle sizes or masses.
            - concentration : The concentration of each particle
              size or mass in the distribution.
            - density : The density of the particles.

        Returns:
            - The total mass of the particles.
        """
        masses = self.get_mass(distribution, density)
        return np.sum(masses * concentration)

    @abstractmethod
    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculate the radius of the particles.

        Arguments:
            - distribution : The distribution of particle sizes or masses.
            - density : The density of the particles.

        Returns:
            - The radius of the particles in meters.
        """

    @abstractmethod
    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Add mass to the distribution of particles.

        Arguments:
            - distribution : The distribution of particle sizes or masses.
            - concentration : The concentration of each particle
              size or mass.
            - density : The density of the particles.
            - added_mass : The mass to be added per distribution bin.

        Returns:
            - The updated distribution array.
            - The updated concentration array.
        """

    @abstractmethod
    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Add concentration to the distribution of particles.

        Arguments:
            - distribution : The distribution of particle sizes or masses.
            - concentration : The concentration of each particle
              size or mass.
            - added_distribution : The distribution to be added.
            - added_concentration : The concentration to be added.

        Returns:
            - The updated distribution array
            - The updated concentration array.
        """

    @abstractmethod
    def collide_pairs(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Collide index pairs in the distribution.

        Arguments:
            - distribution : The distribution of particle sizes or masses.
            - concentration : The concentration of each particle size or mass.
            - density : The density of the particles.
            - indices : The indices of the particles to collide.

        Returns:
            - The updated distribution array
            - The updated concentration array.
        """
