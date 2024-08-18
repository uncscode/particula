"""
Particle Distribution Strategies, how to represent particles in a distribution.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class DistributionStrategy(ABC):
    """
    Abstract base class for particle strategy, defining the common
    interface for mass, radius, and total mass calculations for different
    particle representations.

    Methods:
        get_mass: Calculates the mass of particles.
        get_radius: Calculates the radius of particles.
        get_total_mass: Calculates the total mass of particles.
        add_mass: Adds mass to the distribution of particles.
    """

    @abstractmethod
    def get_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculates the mass of the particles.

        Args:
            distribution: The distribution of particle sizes or masses.
            density: The density of the particles.

        Returns:
            NDArray[np.float64]: The mass of the particles.
        """

    @abstractmethod
    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculates the radius of the particles.

        Args:
            distribution: The distribution of particle sizes or masses.
            density: The density of the particles.

        Returns:
            NDArray[np.float64]: The radius of the particles.
        """

    @abstractmethod
    def get_total_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> np.float64:
        """Calculates the total mass of particles.

        Args:
            distribution: The distribution of particle sizes or masses.
            concentration: The concentration of each particle size or mass in
            the distribution.
            density: The density of the particles.

        Returns:
            np.float64: The total mass of the particles.
        """

    @abstractmethod
    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Adds mass to the distribution of particles.

        Args:
            distribution: The distribution of particle sizes or masses.
            concentration: The concentration of each particle size or mass in
            the distribution.
            density: The density of the particles.
            added_mass: The mass to be added per distribution bin.

        Returns:
            NDArray[np.float64]: The new concentration array.
            NDArray[np.float64]: The new distribution array.
        """


class MassBasedMovingBin(DistributionStrategy):
    """A strategy for particles represented by their mass distribution.

    This strategy calculates particle mass, radius, and total mass based on
    the particle's mass, number concentration, and density. It also moves the
    bins when adding mass to the distribution.
    """

    def get_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # In a mass-based strategy, the mass distribution is directly returned.
        return distribution

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Calculate the volume of each particle from its mass and density,
        # then calculate the radius.
        volumes = distribution / density
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def get_total_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> np.float64:
        # Calculate the total mass by summing the product of the mass
        # distribution and its concentration.
        return np.sum(distribution * concentration)

    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Add the mass to the distribution moving the bins
        return (distribution + added_mass, concentration)


class RadiiBasedMovingBin(DistributionStrategy):
    """A strategy for particles represented by their radius.

    This strategy calculates particle mass, radius, and total mass based on
    the particle's radius, number concentration, and density.
    """

    def get_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Calculate the volume of each particle
        volumes = 4 / 3 * np.pi * distribution**3
        return volumes * density  # Mass is volume multiplied by density

    def get_radius(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return distribution  # Radii are directly available

    def get_total_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> np.float64:
        # Calculate individual masses
        masses = self.get_mass(distribution, density)
        # Total mass is the sum of individual masses times their concentrations
        return np.sum(masses * concentration)

    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Add the the volume of the added mass to the distribution
        new_volume = 4 / 3 * np.pi * (distribution**3 + added_mass / density)
        return (new_volume ** (1 / 3), concentration)


class SpeciatedMassMovingBin(DistributionStrategy):
    """Strategy for particles with speciated mass distribution.

    Strategy for particles with speciated mass distribution.
    Some particles may have different densities and their mass is
    distributed across different species. This strategy calculates mass,
    radius, and total mass based on the species at each mass, density,
    the particle concentration.
    """

    def get_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Broadcasting works natively as each column represents a species
        if distribution.ndim == 1:
            return distribution
        return distribution

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Calculate volume from mass and density, then derive radius
        volumes = np.sum(distribution / density, axis=1)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def get_total_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> np.float64:
        # Calculate mass for each bin and species, then sum for total mass
        # mass_per_species = self.get_mass(distribution, density)
        mass_per_species = np.sum(distribution, axis=1)
        return np.sum(mass_per_species * concentration)

    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Add the mass to the distribution moving the bins
        # limit add to zero, total mass cannot be negative
        if distribution.ndim == 2:
            concentration_expand = concentration[:, np.newaxis]
        else:
            concentration_expand = concentration
        new_mass = (
            np.maximum(distribution * concentration_expand + added_mass, 0)
            / concentration_expand
        )
        return (new_mass, concentration)
