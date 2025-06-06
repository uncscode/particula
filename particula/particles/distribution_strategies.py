"""
Particle Distribution Strategies

Defines various distribution strategies (mass-based, radii-based, etc.)
for representing particles in a distribution. Each strategy handles
mass, radius, total mass, and concentration updates differently.
"""

from abc import ABC, abstractmethod
import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("particula")


class DistributionStrategy(ABC):
    """
    Abstract base class defining common interfaces for
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
        # Calculate the mass of each particle and multiply by the concentration
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


class MassBasedMovingBin(DistributionStrategy):
    """
    Strategy for particles represented by their mass distribution.

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
        if (distribution.shape != added_distribution.shape) and (
            np.allclose(distribution, added_distribution, rtol=1e-6)
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
            + "meaningful, change dynamic or particle strategy."
        )
        logger.warning(message)
        raise NotImplementedError(message)


class RadiiBasedMovingBin(DistributionStrategy):
    """
    Strategy for particles represented by their radius distribution.

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
            + "meaningful, change dynamic or particle strategy."
        )
        logger.warning(message)
        raise NotImplementedError(message)


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
            + "meaningful, change dynamic or particle strategy."
        )
        logger.warning(message)
        raise NotImplementedError(message)


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
            where=concentration_expand != 0,
        )
        # where total new_mass is zero, set concentration to zero
        if new_mass.ndim == 1:
            new_mass_sum = np.sum(new_mass)
        else:
            new_mass_sum = np.sum(new_mass, axis=1)
        concentration = np.where(new_mass_sum > 0, concentration, 0)
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
        if np.allclose(
            added_concentration, np.max(concentration), atol=1e-2
        ) or np.all(concentration == 0):
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
            where=concentration != 0,
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
