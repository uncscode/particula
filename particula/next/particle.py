"""Particle distribution classes and factory."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

# From Particula
from particula.next.particle_activity import ParticleActivityStrategy
from particula.next.surface import SurfaceStrategy


class ParticleStrategy(ABC):
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


class MassBasedMovingBin(ParticleStrategy):
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


class RadiiBasedMovingBin(ParticleStrategy):
    """
    A strategy for particles represented by their radius (distribution),
    and particle concentration. Implementing the ParticleStrategy interface.
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


class SpeciatedMassMovingBin(ParticleStrategy):
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
        - dinsity (NDArray[np.float_]): An array of densities for each
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
            distribution*concentration+added_mass, 0)/concentration
        return (concentration, new_mass)


def particle_strategy_factory(
        particle_representation: str) -> ParticleStrategy:
    """
    Factory function for creating instances of particle strategies based on
    the specified representation type.

    Parameters:
    - particle_type (str): The type of particle representation, determining
    which strategy instance to create.
        - mass_based_moving_bin
        - radii_based_moving_bin
        - speciated_mass_moving_bin

    Returns:
    - An instance of ParticleStrategy corresponding to the specified
        particle type.

    Raises:
    - ValueError: If an unknown particle type is specified.
    """
    if particle_representation == "mass_based_moving_bin":
        return MassBasedMovingBin()
    if particle_representation == "radii_based_moving_bin":
        return RadiiBasedMovingBin()
    if particle_representation == "speciated_mass_moving_bin":
        return SpeciatedMassMovingBin()
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
    representations.
    - activity (ParticleActivityStrategy): The activity strategy for the
    partial pressure calculations.
    - surface (SurfaceStrategy): The surface strategy for surface tension and
    Kelvin effect.
    - distribution (NDArray[np.float_]): The distribution data for the
    particles, which could represent sizes, masses, or another relevant metric.
    - density (np.float_): The density of the material from which the
    particles are made.
    - concentration (NDArray[np.float_]): The concentration of particles
    within the distribution.
    """

    def __init__(
        self,
        strategy: ParticleStrategy,
        activity: ParticleActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_],
        concentration: NDArray[np.float_]
    ):  # pylint: disable=too-many-arguments
        """
        Initializes a Particle instance with a strategy, distribution,
        density, and concentration.

        Parameters:
        - strategy (ParticleStrategy): The strategy to use for particle
        property calculations.
        - distribution (NDArray[np.float_]): The distribution of particle
        sizes or masses.
        - density (np.float_): The material density of the particles.
        - concentration (NDArray[np.float_]): The concentration of each size
        or mass in the distribution.
        """
        self.strategy = strategy
        self.activity = activity
        self.surface = surface
        self.distribution = distribution
        self.density = density
        self.concentration = concentration

    def get_mass(self) -> NDArray[np.float_]:
        """
        Returns the mass of the particles as calculated by the strategy.

        Returns:
        - NDArray[np.float_]: The mass of the particles.
        """
        return self.strategy.get_mass(self.distribution, self.density)

    def get_radius(self) -> NDArray[np.float_]:
        """
        Returns the radius of the particles as calculated by the strategy.

        Returns:
        - NDArray[np.float_]: The radius of the particles.
        """
        return self.strategy.get_radius(self.distribution, self.density)

    def get_total_mass(self) -> np.float_:
        """
        Returns the total mass of the particles as calculated by the strategy,
        taking into account the distribution and concentration.

        Returns:
            np.float_: The total mass of the particles.
        """
        return self.strategy.get_total_mass(
            self.distribution, self.concentration, self.density)

    def add_mass(self, added_mass: NDArray[np.float_]) -> None:
        """
        Adds mass to the particle distribution, updating the concentration
        and distribution arrays.

        Parameters:
        - added_mass (NDArray[np.float_]): The mass to be added per
            distribution bin.
        """
        self.concentration, self.distribution = self.strategy.add_mass(
            self.distribution, self.concentration, self.density, added_mass)

    def add_concentration(
        self,
        added_concentration: NDArray[np.float_]
    ) -> None:
        """
        Adds concentration to the particle distribution, updating the
        concentration array.

        Parameters:
        - added_concentration (NDArray[np.float_]): The concentration to be
            added per distribution bin.
        """
        self.concentration += added_concentration
