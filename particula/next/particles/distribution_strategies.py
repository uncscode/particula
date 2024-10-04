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
        get_name: Returns the type of the distribution strategy.
        get_mass: Calculates the mass of particles.
        get_radius: Calculates the radius of particles.
        get_total_mass: Calculates the total mass of particles.
        add_mass: Adds mass to the distribution of particles.
    """

    def get_name(self) -> str:
        """Return the type of the distribution strategy."""
        return self.__class__.__name__

    @abstractmethod
    def get_species_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The mass per species in the particles (or bin).

        Args:
            distribution: The distribution of particle sizes or masses.
            density: The density of the particles.

        Returns:
            NDArray[np.float64]: The mass of the particles
        """

    def get_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculates the mass of the particles (or bin).

        Args:
            distribution: The distribution of particle sizes or masses.
            density: The density of the particles.

        Returns:
            NDArray[np.float64]: The mass of the particles.
        """
        # one species present
        if distribution.ndim == 1:
            return self.get_species_mass(distribution, density)
        # each column represents a species
        return np.sum(self.get_species_mass(distribution, density), axis=1)

    def get_total_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
    ) -> np.float64:
        """Calculates the total mass of all particles (or bin).

        Args:
            distribution: The distribution of particle sizes or masses.
            concentration: The concentration of each particle size or mass in
            the distribution.
            density: The density of the particles.

        Returns:
            np.float64: The total mass of the particles.
        """
        # Calculate the mass of each particle and multiply by the concentration
        masses = self.get_mass(distribution, density)
        return np.sum(masses * concentration)

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
            (distribution, concentration): The new distribution array and the
                new concentration array.
        """

    @abstractmethod
    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Adds concentration to the distribution of particles.

        Args:
            distribution: The distribution of particle sizes or masses.
            concentration: The concentration of each particle size or mass in
                the distribution.
            added_distribution: The distribution to be added.
            added_concentration: The concentration to be added.

        Returns:
            (distribution, concentration): The new distribution array and the
                new concentration array.
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
            (distribution, concentration): The new distribution array and the
                new concentration array.
        """


class MassBasedMovingBin(DistributionStrategy):
    """A strategy for particles represented by their mass distribution.

    This strategy calculates particle mass, radius, and total mass based on
    the particle's mass, number concentration, and density. It also moves the
    bins when adding mass to the distribution.
    """

    def get_species_mass(
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

    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Add the mass to the distribution moving the bins
        return (distribution + added_mass, concentration)

    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # check if distribution and added distribution have same elements
        if (
            (distribution.shape != added_distribution.shape) and
            (np.allclose(distribution, added_distribution, rtol=1e-6))
        ):
            message = (
                "When adding concentration to MassBasedMovingBin,"
                + " the distribution and added distribution should have "
                "the same elements. The current distribution shape is "
                f"{distribution.shape} and the added distribution shape is "
                f"{added_distribution.shape}."
            )
            logger.error(message)
            raise ValueError(message)
        # concentration shape should be equal
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to MassBasedMovingBin,"
                " the concentration and added concentration should have "
                "the same shape. The current concentration shape is "
                f"{concentration.shape} and the added concentration shape is "
                f"{added_concentration.shape}."
            )
            logger.error(message)
            raise ValueError(message)

        # add the concentration in place
        concentration += added_concentration
        return (distribution, concentration)

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

    def get_species_mass(
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

    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Step 1: Calculate mass added per particle
        mass_per_particle = np.where(
            concentration > 0, added_mass / concentration, 0
        )
        # Step 2: Calculate new volumes
        initial_volumes = (4 / 3) * np.pi * np.power(distribution, 3)
        new_volumes = initial_volumes + mass_per_particle / density
        # Step 3: Convert new volumes back to radii
        new_radii = np.power(3 * new_volumes / (4 * np.pi), 1 / 3)
        return (new_radii, concentration)

    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # check if distribution and added distribution have same elements
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to RadiiBasedMovingBin,"
                " the distribution and added distribution should have "
                "the same elements. The current distribution shape is "
                f"{distribution.shape} and the added distribution shape is "
                f"{added_distribution.shape}."
            )
            logger.error(message)
            raise ValueError(message)
        # concentration shape should be equal
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to RadiiBasedMovingBin,"
                " the concentration and added concentration should have "
                "the same shape. The current concentration shape is "
                f"{concentration.shape} and the added concentration shape is "
                f"{added_concentration.shape}."
            )
            logger.error(message)
            raise ValueError(message)
        # add the concentration in place
        concentration += added_concentration
        return (distribution, concentration)

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

    def get_species_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return distribution

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Calculate volume from mass and density, then derive radius
        volumes = np.sum(distribution / density, axis=1)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

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
            concentration_expand > 0, added_mass / concentration_expand, 0
        )
        # Step 3: Update the distribution by adding the mass per particle
        new_distribution = np.maximum(distribution + mass_per_particle, 0)
        return new_distribution, concentration

    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # check if distribution and added distribution have same elements
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to SpeciatedMassMovingBin,"
                " the distribution and added distribution should have "
                "the same elements. The current distribution shape is "
                f"{distribution.shape} and the added distribution shape is "
                f"{added_distribution.shape}."
            )
            logger.error(message)
            raise ValueError(message)
        # concentration shape should be equal
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to SpeciatedMassMovingBin,"
                " the concentration and added concentration should have "
                "the same shape. The current concentration shape is "
                f"{concentration.shape} and the added concentration shape is "
                f"{added_concentration.shape}."
            )
            logger.error(message)
            raise ValueError(message)
        # add the concentration in place
        concentration += added_concentration
        return (distribution, concentration)

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

    def get_species_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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

        new_mass = np.divide(
            np.maximum(distribution * concentration_expand + added_mass, 0),
            concentration_expand,
            out=np.zeros_like(distribution),
            where=concentration_expand != 0
        )
        return (new_mass, concentration)

    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Check if added concentration is all ones
        rescaled = False
        if np.all(added_concentration == 1):
            rescaled = True
        if (
            np.allclose(
                added_concentration, np.max(concentration), atol=1e-2) or
            np.all(concentration == 0)
           ):
            # then rescale the added concentration
            added_concentration = added_concentration / np.max(concentration)
            rescaled = True
        if not rescaled:
            message = (
                "When adding concentration to ParticleResolvedSpeciatedMass,"
                + " the added concentration should be all ones or all the same"
                + " value of 1/volume."
            )
            logger.error(message)
            raise ValueError(message)

        # replace concentration with ones, as it is not defined by volume,
        # when it is stored in the class
        concentration = np.divide(
            concentration,
            concentration,
            out=np.zeros_like(concentration),
            where=concentration != 0
        )

        # find empty distribution bins
        empty_bins = np.flatnonzero(np.all(concentration == 0))
        # count of empty bins
        empty_bins_count = len(empty_bins)
        added_bins_count = len(added_concentration)
        if empty_bins_count >= added_bins_count:
            # add all added bins to empty bins
            distribution[empty_bins] = added_distribution
            concentration[empty_bins] = added_concentration
            return distribution, concentration
        # add added bins to empty bins
        if empty_bins_count > 0:
            distribution[empty_bins] = added_distribution[:empty_bins_count]
            concentration[empty_bins] = added_concentration[:empty_bins_count]
        # add the rest of the added bins to the end of the distribution
        distribution = np.concatenate(
            (distribution, added_distribution[empty_bins_count:]),
            axis=0
        )
        concentration = np.concatenate(
            (concentration, added_concentration[empty_bins_count:]),
            axis=0
        )
        return distribution, concentration

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
            # Step 1: Transfer all species mass from smaller to larger
            # particles
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
