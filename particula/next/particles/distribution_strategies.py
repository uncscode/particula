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
    """

    @abstractmethod
    def get_mass(
        self,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        """
        Calculates the mass of particles based on their distribution and
        density.

        Parameters:
        - distribution (NDArray[np.float_]): The distribution of particle
            sizes or masses.
        - density (NDArray[np.float_]): The density of the particles.

        Returns:
        - NDArray[np.float_]: The mass of the particles.
        """

    @abstractmethod
    def get_radius(
        self,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        """
        Calculates the radius of particles based on their distribution and
        density.

        Parameters:
        - distribution (NDArray[np.float_]): The distribution of particle
            sizes or masses.
        - density (NDArray[np.float_]): The density of the particles.

        Returns:
        - NDArray[np.float_]: The radius of the particles.
        """

    @abstractmethod
    def get_total_mass(
        self,
        distribution: NDArray[np.float_],
        concentration: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> np.float_:
        """
        Calculates the total mass of particles based on their distribution,
        concentration, and density.

        Parameters:
        - distribution (NDArray[np.float_]): The distribution of particle
            sizes or masses.
        - concentration (NDArray[np.float_]): The concentration of each
            particle size or mass in the distribution.
        - density (NDArray[np.float_]): The density of the particles.

        Returns:
        - np.float_: The total mass of the particles.
        """

    @abstractmethod
    def add_mass(
        self,
        distribution: NDArray[np.float_],
        concentration: NDArray[np.float_],
        density: NDArray[np.float_],
        added_mass: NDArray[np.float_]
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """
        Adds mass to the distribution of particles based on their distribution,
        concentration, and density.

        Parameters:
        - distribution (NDArray[np.float_]): The distribution representation
        of particles
        - concentration (NDArray[np.float_]): The concentration of each
        particle in the distribution.
        - density (NDArray[np.float_]): The density of the particles.
        - added_mass (NDArray[np.float_]): The mass to be added per
        distribution bin.

        Returns:
        - NDArray[np.float_]: The new concentration array.
        - NDArray[np.float_]: The new distribution array.
        """


class MassBasedMovingBin(DistributionStrategy):
    """
    A strategy for particles represented by their mass distribution, and
    particle number concentration. Moving the bins when adding mass.
    """

    def get_mass(
        self,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        # In a mass-based strategy, the mass distribution is directly returned.
        return distribution

    def get_radius(
        self,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        # Calculate the volume of each particle from its mass and density,
        # then calculate the radius.
        volumes = distribution / density
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def get_total_mass(
        self,
        distribution: NDArray[np.float_],
        concentration: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> np.float_:
        # Calculate the total mass by summing the product of the mass
        # distribution and its concentration.
        return np.sum(distribution * concentration)

    def add_mass(
        self,
        distribution: NDArray[np.float_],
        concentration: NDArray[np.float_],
        density: NDArray[np.float_],
        added_mass: NDArray[np.float_]
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        # Add the mass to the distribution moving the bins
        return (distribution + added_mass, concentration)


class RadiiBasedMovingBin(DistributionStrategy):
    """
    A strategy for particles represented by their radius (distribution),
    and particle concentration. Implementing the DistributionStrategy
    interface.
    This strategy calculates particle mass, radius, and total mass based on
    the particle's radius, number concentration, and density.
    """

    def get_mass(
        self,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        # Calculate the volume of each particle
        volumes = 4 / 3 * np.pi * distribution ** 3
        return volumes * density  # Mass is volume multiplied by density

    def get_radius(
        self,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        return distribution  # Radii are directly available

    def get_total_mass(
        self,
        distribution: NDArray[np.float_],
        concentration: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> np.float_:
        # Calculate individual masses
        masses = self.get_mass(distribution, density)
        # Total mass is the sum of individual masses times their concentrations
        return np.sum(masses * concentration)

    def add_mass(
        self,
        distribution: NDArray[np.float_],
        concentration: NDArray[np.float_],
        density: NDArray[np.float_],
        added_mass: NDArray[np.float_]
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        # Add the the volume of the added mass to the distribution
        new_volume = 4 / 3 * np.pi * (distribution ** 3 + added_mass / density)
        return (new_volume ** (1 / 3), concentration)


class SpeciatedMassMovingBin(DistributionStrategy):
    """Strategy for particles with speciated mass distribution.
    Some particles may have different densities and their mass is
    distributed across different species. This strategy calculates mass,
    radius, and total mass based on the species at each mass, density,
    the particle concentration."""

    def get_mass(
        self,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        """
        Calculates the mass for each mass and species, leveraging densities
        for adjustment.

        Parameters:
        - distribution (NDArray[np.float_]): A 2D array with rows
            representing mass bins and columns representing species.
        - densities (NDArray[np.float_]): An array of densities for each
            species.

        Returns:
        - NDArray[np.float_]: A 1D array of calculated masses for each mass
            bin. The sum of each column (species) in the distribution matrix.
        """
        # Broadcasting works natively as each column represents a species
        if distribution.ndim == 1:
            return distribution
        return np.sum(distribution, axis=1)

    def get_radius(
        self,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        """
        Calculates the radius for each mass bin and species, based on the
        volume derived from mass and density.

        Parameters:
        - distribution (NDArray[np.float_]): A 2D array with rows representing
            mass bins and columns representing species.
        - density (NDArray[np.float_]): An array of densities for each
            species.

        Returns:
        - NDArray[np.float_]: A 1D array of calculated radii for each mass
            bin.
        """
        # Calculate volume from mass and density, then derive radius
        volumes = np.sum(distribution / density, axis=0)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def get_total_mass(
        self,
        distribution: NDArray[np.float_],
        concentration: NDArray[np.float_],
        density: NDArray[np.float_]
    ) -> np.float_:
        """
        Calculates the total mass of all species, incorporating the
        concentration of particles per species.

        Parameters:
        - distribution (NDArray[np.float_]): The mass distribution matrix.
        - counts (NDArray[np.float_]): A 1D array with elements representing
            the count of particles for each species.

        Returns:
        - np.float_: The total mass of all particles.
        """
        # Calculate mass for each bin and species, then sum for total mass
        mass_per_species = self.get_mass(distribution, density)
        return np.sum(mass_per_species * concentration)

    def add_mass(
        self,
        distribution: NDArray[np.float_],
        concentration: NDArray[np.float_],
        density: NDArray[np.float_],
        added_mass: NDArray[np.float_]
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        # Add the mass to the distribution moving the bins
        # limit add to zero, total mass cannot be negative
        new_mass = np.maximum(
            distribution * concentration + added_mass, 0) / concentration
        return (concentration, new_mass)
