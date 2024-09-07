"""
Particle Distribution Strategies, how to represent particles in a distribution.
"""

from abc import ABC, abstractmethod
import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("particula")


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

    @abstractmethod
    def collide_pairs(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Collides index pairs.

        Args:
            distribution: The distribution of particle sizes or masses.
            concentration: The concentration of each particle size or mass in
                the distribution.
            density: The density of the particles.
            indices: The indices of the particles to collide.

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

    def collide_pairs(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        message = (
            "Colliding pairs in MassBasedMovingBin is not physically"
            + "meaningful, change dyanmic or particle strategy."
        )
        logger.warning(message)
        raise NotImplementedError(message)


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
        # Step 1: Calculate mass added per particle
        mass_per_particle = np.where(
            concentration > 0,
            added_mass / concentration,
            0
        )
        # Step 2: Calculate new volumes
        initial_volumes = (4 / 3) * np.pi * np.power(distribution, 3)
        new_volumes = initial_volumes + mass_per_particle / density
        # Step 3: Convert new volumes back to radii
        new_radii = np.power(3 * new_volumes / (4 * np.pi), 1 / 3)
        return (new_radii, concentration)

    def collide_pairs(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        message = (
            "Colliding pairs in RadiiBasedMovingBin is not physically"
            + "meaningful, change dyanmic or particle strategy."
        )
        logger.warning(message)
        raise NotImplementedError(message)


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
        # Step 1: Expand concentration if necessary for 2D distributions
        if distribution.ndim == 2:
            concentration_expand = concentration[:, np.newaxis]
        else:
            concentration_expand = concentration
        # Step 2: Calculate mass added per particle (handle zero concentration)
        mass_per_particle = np.where(
            concentration_expand > 0,
            added_mass / concentration_expand,
            0
        )
        # Step 3: Update the distribution by adding the mass per particle
        new_distribution = np.maximum(distribution + mass_per_particle, 0)
        return new_distribution, concentration

    def collide_pairs(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        message = (
            "Colliding pairs in SpeciatedMassMovingBin is not physically"
            + "meaningful, change dyanmic or particle strategy."
        )
        logger.warning(message)
        raise NotImplementedError(message)


class ParticleResolvedSpeciatedMass(DistributionStrategy):
    """Strategy for resolved particles via speciated mass.

    Strategy for resolved particles with speciated mass.
    Particles may have different densities and their mass is
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
        if distribution.ndim == 1:
            volumes = distribution / density
        else:
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
        mass_per_species = np.sum(distribution, axis=-1)
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

    def collide_pairs(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Get the indices of the smaller and larger particles
        small_index = indices[:, 0]
        large_index = indices[:, 1]
        # Check if the distribution is 1D or 2D
        if distribution.ndim == 1:
            # Step 1: Transfer all species mass from smaller to larger particles
            distribution[large_index] += distribution[small_index]
            # Step 2: Zero out the mass of the smaller particles
            distribution[small_index] = 0
            # Step 3: Zero out the concentration of the smaller particles
            concentration[small_index] = 0
            return distribution, concentration

        # Step 1: Transfer all species mass from smaller to larger particles
        distribution[large_index, :] += distribution[small_index, :]
        # Step 2: Zero out the mass of the smaller particles
        distribution[small_index, :] = 0
        # Step 3: Zero out the concentration of the smaller particles
        concentration[small_index] = 0
        # Return the updated distribution (masses) and concentration
        return distribution, concentration
