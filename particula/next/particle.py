"""Particle distribution classes and factory."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class ParticleStrategy(ABC):
    """
    Abstract base class for particle strategy, defining the common
    interface for mass, radius, and total mass calculations for different
    particle representations.
    """

    @abstractmethod
    def get_mass(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculates the mass of particles based on their distribution and
        density.

        Parameters:
        - distribution (NDArray[np.float64]): The distribution of particle
            sizes or masses.
        - density (NDArray[np.float64]): The density of the particles.

        Returns:
        - NDArray[np.float64]: The mass of the particles.
        """

    @abstractmethod
    def get_radius(
        self,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculates the radius of particles based on their distribution and
        density.

        Parameters:
        - distribution (NDArray[np.float64]): The distribution of particle
            sizes or masses.
        - density (NDArray[np.float64]): The density of the particles.

        Returns:
        - NDArray[np.float64]: The radius of the particles.
        """

    @abstractmethod
    def get_total_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64]
    ) -> np.float64:
        """
        Calculates the total mass of particles based on their distribution,
        concentration, and density.

        Parameters:
        - distribution (NDArray[np.float64]): The distribution of particle
            sizes or masses.
        - concentration (NDArray[np.float64]): The concentration of each
            particle size or mass in the distribution.
        - density (NDArray[np.float64]): The density of the particles.

        Returns:
        - np.float64: The total mass of the particles.
        """


class MassBasedStrategy(ParticleStrategy):
    """
    A strategy for particles represented by their mass distribution, and
    particle number concentration. This class provides the implementation
    of the methods for ParticleStrategy.
    """

    def get_mass(
            self,
            distribution: NDArray[np.float64],
            density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # In a mass-based strategy, the mass distribution is directly returned.
        return distribution

    def get_radius(
            self,
            distribution: NDArray[np.float64],
            density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Calculate the volume of each particle from its mass and density,
        # then calculate the radius.
        volumes = distribution / density
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def get_total_mass(
            self,
            distribution: NDArray[np.float64],
            concentration: NDArray[np.float64],
            density: NDArray[np.float64]
    ) -> np.float64:
        # Calculate the total mass by summing the product of the mass
        # distribution and its concentration.
        return np.sum(distribution * concentration)


class RadiiBasedStrategy(ParticleStrategy):
    """
    A strategy for particles represented by their radius (distribution),
    and particle conentraiton. Implementing the ParticleStrategy interface.
    This strategy calculates particle mass, radius, and total mass based on
    the particle's radius, number concentraiton, and density.
    """

    def get_mass(
            self,
            distribution: NDArray[np.float64],
            density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Calculate the volume of each particle
        volumes = 4 / 3 * np.pi * distribution ** 3
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
            density: NDArray[np.float64]
    ) -> np.float64:
        # Calculate individual masses
        masses = self.get_mass(distribution, density)
        # Total mass is the sum of individual masses times their concentrations
        return np.sum(masses * concentration)


class SpeciatedMassStrategy(ParticleStrategy):
    """Strategy for particles with speciated mass distribution.
    Some particles may have different densities and their mass is
    distributed across different species. This strategy calculates mass,
    radius, and total mass based on the species at each mass, density,
    the particle concentration."""

    def get_mass(
            self,
            distribution: NDArray[np.float64],
            density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculates the mass for each mass and species, leveraging densities
        for adjustment.

        Parameters:
        - distribution (NDArray[np.float64]): A 2D array with rows
            representing mass bins and columns representing species.
        - densities (NDArray[np.float64]): An array of densities for each
            species.

        Returns:
        - NDArray[np.float64]: A 1D array of calculated masses for each mass
            bin. The sum of each column (species) in the distribution matrix.
        """
        # Broadcasting works natively as each column represents a species
        return np.sum(distribution, axis=1)

    def get_radius(
            self,
            distribution: NDArray[np.float64],
            density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculates the radius for each mass bin and species, based on the
        volume derived from mass and density.

        Parameters:
        - distribution (NDArray[np.float64]): A 2D array with rows representing
            mass bins and columns representing species.
        - dinsity (NDArray[np.float64]): An array of densities for each
            species.

        Returns:
        - NDArray[np.float64]: A 1D array of calculated radii for each mass
            bin.
        """
        # Calculate volume from mass and density, then derive radius
        volumes = np.sum(distribution / density, axis=0)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def get_total_mass(
            self,
            distribution: NDArray[np.float64],
            concentration: NDArray[np.float64],
            density: NDArray[np.float64]
    ) -> np.float64:
        """
        Calculates the total mass of all species, incorporating the
        concentration of particles per species.

        Parameters:
        - distribution (NDArray[np.float64]): The mass distribution matrix.
        - counts (NDArray[np.float64]): A 1D array with elements representing
            the count of particles for each species.

        Returns:
        - np.float64: The total mass of all particles.
        """
        # Calculate mass for each bin and species, then sum for total mass
        mass_per_species = self.get_mass(distribution, density)
        return np.sum(mass_per_species * concentration)


def create_particle_strategy(particle_representation: str) -> ParticleStrategy:
    """
    Factory function for creating instances of particle strategies based on
    the specified representation type.

    Parameters:
    - particle_type (str): The type of particle representation, determining
    which strategy instance to create.
    [mass_based, radii_based, speciated_mass]

    Returns:
    - An instance of ParticleStrategy corresponding to the specified
        particle type.

    Raises:
    - ValueError: If an unknown particle type is specified.
    """
    if particle_representation == "mass_based":
        return MassBasedStrategy()
    if particle_representation == "radii_based":
        return RadiiBasedStrategy()
    if particle_representation == "speciated_mass":
        return SpeciatedMassStrategy()
    raise ValueError(f"Unknown particle strategy: {particle_representation}")


# note there is no inheritance from ParticleStrategy, this is a design choice
# for injection dependency over inheritance
class Particle:
    """
    Represents a particle or a collection of particles, encapsulating the
    strategy for calculating mass, radius, and total mass based on a
    specified particle distribution, density, and concentration. This class
    allows for flexibility in representing particles by delegating computation
    to a strategy pattern.

    Attributes:
    - strategy (ParticleStrategy): The computation strategy for particle
    properties.
    - distribution (NDArray[np.float64]): The distribution data for the
    particles, which could represent sizes, masses, or another relevant metric.
    - density (np.float64): The density of the material from which the
    particles are made.
    - concentration (NDArray[np.float64]): The concentration of particles
    within the distribution.
    - charge (Optional[NDArray[np.float64]]): The charge distribution of the
    particles.
    - shape_factor (Optional[NDArray[np.float64]]): The shape factor
    distribution of the particles.
    """

    def __init__(self,
                 strategy: ParticleStrategy,
                 distribution: NDArray[np.float64],
                 density: NDArray[np.float64],
                 concentration: NDArray[np.float64]):
        """
        Initializes a Particle instance with a strategy, distribution,
        density, and concentration.

        Parameters:
        - strategy (ParticleStrategy): The strategy to use for particle
        property calculations.
        - distribution (NDArray[np.float64]): The distribution of particle
        sizes or masses.
        - density (np.float64): The material density of the particles.
        - concentration (NDArray[np.float64]): The concentration of each size
        or mass in the distribution.
        """
        self.strategy = strategy
        self.distribution = distribution
        self.density = density
        self.concentration = concentration
        # Initialize optional attributes
        self.charge = None
        self.shape_factor = None
        # self.viscosity = None

    def set_charge(self, charge: NDArray[np.float64]):
        """
        Sets the charge distribution for the particles.

        Parameters:
        - charge (NDArray[np.float64]): The charge distribution across the
        particles.
        """
        self.charge = charge

    def set_shape_factor(self, shape_factor: NDArray[np.float64]):
        """
        Sets the shape factor distribution for the particles.

        Parameters:
        - shape_factor (NDArray[np.float64]): The shape factor distribution
        across the particles.
        """
        self.shape_factor = shape_factor

    def get_mass(self) -> NDArray[np.float64]:
        """
        Returns the mass of the particles as calculated by the strategy.

        Returns:
        - NDArray[np.float64]: The mass of the particles.
        """
        return self.strategy.get_mass(self.distribution, self.density)

    def get_radius(self) -> NDArray[np.float64]:
        """
        Returns the radius of the particles as calculated by the strategy.

        Returns:
        - NDArray[np.float64]: The radius of the particles.
        """
        return self.strategy.get_radius(self.distribution, self.density)

    def get_total_mass(self) -> np.float64:
        """
        Returns the total mass of the particles as calculated by the strategy,
        taking into account the distribution and concentration.

        Returns:
            np.float64: The total mass of the particles.
        """
        return self.strategy.get_total_mass(
            self.distribution, self.concentration, self.density)
