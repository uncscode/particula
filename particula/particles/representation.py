"""Particle representation."""

from typing import Optional
from copy import deepcopy
import logging
import numpy as np
from numpy.typing import NDArray

# From Particula
from particula.particles.activity_strategies import ActivityStrategy
from particula.particles.surface_strategies import SurfaceStrategy
from particula.particles.distribution_strategies import (
    DistributionStrategy,
)

logger = logging.getLogger("particula")


# pylint: disable=too-many-instance-attributes
class ParticleRepresentation:
    """Everything needed to represent a particle or a collection of particles.

    Represents a particle or a collection of particles, encapsulating the
    strategy for calculating mass, radius, and total mass based on a
    specified particle distribution, density, and concentration. This class
    allows for flexibility in representing particles.

    Attributes:
        strategy: The computation strategy for particle representations.
        activity: The activity strategy for the partial pressure calculations.
        surface: The surface strategy for surface tension and Kelvin effect.
        distribution: The distribution data for the particles, which could
            represent sizes, masses, or another relevant metric.
        density: The density of the material from which the particles are made.
        concentration: The concentration of particles within the distribution.
        charge: The charge on each particle.
        volume: The air volume for simulation of particles in the air,
            default is 1 m^3. This is only used in ParticleResolved Strategies.
    """

    def __init__(
        self,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
        concentration: NDArray[np.float64],
        charge: NDArray[np.float64],
        volume: float = 1,
    ):  # pylint: disable=too-many-positional-arguments, too-many-arguments
        self.strategy = strategy
        self.activity = activity
        self.surface = surface
        self.distribution = distribution
        self.density = density
        self.concentration = concentration
        self.charge = charge
        self.volume = volume

    def __str__(self) -> str:
        """Returns a string representation of the particle representation.

        Returns:
            str: A string representation of the particle representation.
        """
        return (
            f"Particle Representation:\n"
            f"\tStrategy: {self.get_strategy_name()}\n"
            f"\tActivity: {self.get_activity_name()}\n"
            f"\tSurface: {self.get_surface_name()}\n"
            f"\tMass Concentration: "
            f"{self.get_mass_concentration():.3e} [kg/m^3]\n"
            f"\tNumber Concentration: "
            f"{self.get_total_concentration():.3e} [#/m^3]"
        )

    def get_strategy(self, clone: bool = False) -> DistributionStrategy:
        """Returns the strategy used for particle representation.

        Args:
            clone: If True, then return a deepcopy of the strategy.

        Returns:
            The strategy used for particle representation.
        """
        if clone:
            return deepcopy(self.strategy)
        return self.strategy

    def get_strategy_name(self) -> str:
        """Returns the name of the strategy used for particle representation.

        Returns:
            The name of the strategy used for particle representation.
        """
        return self.strategy.get_name()

    def get_activity(self, clone: bool = False) -> ActivityStrategy:
        """Returns the activity strategy used for partial pressure
        calculations.

        Args:
            clone: If True, then return a deepcopy of the activity strategy.

        Returns:
            The activity strategy used for partial pressure calculations.
        """
        if clone:
            return deepcopy(self.activity)
        return self.activity

    def get_activity_name(self) -> str:
        """Returns the name of the activity strategy used for partial pressure
        calculations.

        Returns:
            The name of the activity strategy used for partial pressure
            calculations.
        """
        return self.activity.get_name()

    def get_surface(self, clone: bool = False) -> SurfaceStrategy:
        """Returns the surface strategy used for surface tension and
            Kelvin effect.

        Args:
            clone: If True, then return a deepcopy of the surface strategy.

        Returns:
            The surface strategy used for surface tension and Kelvin effect.
        """
        if clone:
            return deepcopy(self.surface)
        return self.surface

    def get_surface_name(self) -> str:
        """Returns the name of the surface strategy used for surface tension
        and Kelvin effect.

        Returns:
            The name of the surface strategy used for surface tension and
            Kelvin effect.
        """
        return self.surface.get_name()

    def get_distribution(self, clone: bool = False) -> NDArray[np.float64]:
        """Returns the distribution of the particles.

        Args:
            clone: If True, then return a copy of the distribution array.

        Returns:
            The distribution of the particles.
        """
        if clone:
            return np.copy(self.distribution)
        return self.distribution

    def get_density(self, clone: bool = False) -> NDArray[np.float64]:
        """Returns the density of the particles.

        Args:
            clone: If True, then return a copy of the density array.

        Returns:
            The density of the particles.
        """
        if clone:
            return np.copy(self.density)
        return self.density

    def get_concentration(self, clone: bool = False) -> NDArray[np.float64]:
        """Returns the volume concentration of the particles.

        For ParticleResolved Strategies, the concentration is the number of
        particles per self.volume to get concentration/m^3. For other
        Strategies, the concentration is the already per 1/m^3.

        Args:
            clone: If True, then return a copy of the concentration array.

        Returns:
            The concentration of the particles.
        """
        if clone:
            return np.copy(self.concentration / self.volume)
        return self.concentration / self.volume

    def get_total_concentration(self, clone: bool = False) -> np.float64:
        """Returns the total concentration of the particles.

        Args:
            clone: If True, then return a copy of the concentration array.

        Returns:
            The concentration of the particles.
        """
        return np.sum(self.get_concentration(clone=clone))

    def get_charge(self, clone: bool = False) -> NDArray[np.float64]:
        """Returns the charge per particle.

        Args:
            clone: If True, then return a copy of the charge array.

        Returns:
            The charge of the particles.
        """
        if clone:
            return np.copy(self.charge)
        return self.charge

    def get_volume(self, clone: bool = False) -> float:
        """Returns the volume of the particles.

        Args:
            clone: If True, then return a copy of the volume array.

        Returns:
            The volume of the particles.
        """
        if clone:
            return deepcopy(self.volume)
        return self.volume

    def get_species_mass(self, clone: bool = False) -> NDArray[np.float64]:
        """Returns the masses per species in the particles.

        Args:
            clone: If True, then return a copy of the mass array.

        Returns:
            The mass of the particles per species.
        """
        if clone:
            return np.copy(
                self.strategy.get_species_mass(self.distribution, self.density)
            )
        return self.strategy.get_species_mass(self.distribution, self.density)

    def get_mass(self, clone: bool = False) -> NDArray[np.float64]:
        """Returns the mass of the particles as calculated by the strategy.

        Args:
            clone: If True, then return a copy of the mass array.

        Returns:
            The mass of the particles.
        """
        if clone:
            return np.copy(
                self.strategy.get_mass(self.distribution, self.density)
            )
        return self.strategy.get_mass(self.distribution, self.density)

    def get_mass_concentration(self, clone: bool = False) -> np.float64:
        """Returns the total mass / volume simulated.

        The mass concentration is as calculated by the strategy, taking into
        account the distribution and concentration.

        Args:
            clone: If True, then return a copy of the mass concentration.

        Returns:
            np.float64: The mass concentration of the particles, kg/m^3.
        """
        if clone:
            return deepcopy(
                self.strategy.get_total_mass(
                    self.get_distribution(),
                    self.get_concentration(),
                    self.get_density(),
                )
            )
        return self.strategy.get_total_mass(
            self.get_distribution(),
            self.get_concentration(),
            self.get_density(),
        )

    def get_radius(self, clone: bool = False) -> NDArray[np.float64]:
        """Returns the radius of the particles as calculated by the strategy.

        Args:
            clone: If True, then return a copy of the radius array
        Returns:
            The radius of the particles.
        """
        if clone:
            return np.copy(
                self.strategy.get_radius(self.distribution, self.density)
            )
        return self.strategy.get_radius(self.distribution, self.density)

    def add_mass(self, added_mass: NDArray[np.float64]) -> None:
        """Adds mass to the particle distribution, and updates parameters.

        Args:
            added_mass: The mass to be added per
                distribution bin.
        """
        # maybe remove return concentration
        (self.distribution, _) = self.strategy.add_mass(
            self.get_distribution(),
            self.get_concentration(),
            self.get_density(),
            added_mass,
        )

    def add_concentration(
        self,
        added_concentration: NDArray[np.float64],
        added_distribution: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Adds concentration to the particle distribution.

        Args:
            added_concentration: The concentration to be
                added per distribution bin.
        """
        # if added_distribution is None, then it will be calculated
        if added_distribution is None:
            message = "Added distribution is value None."
            logger.warning(message)
            added_distribution = self.get_distribution()
        # self.concentration += added_concentration
        (self.distribution, self.concentration) = (
            self.strategy.add_concentration(
                distribution=self.get_distribution(),
                concentration=self.get_concentration(),
                added_distribution=added_distribution,
                added_concentration=added_concentration,
            )
        )

    def collide_pairs(self, indices: NDArray[np.int64]) -> None:
        """Collide pairs of indices, used for ParticleResolved Strategies.

        Args:
            indices: The indices to collide.
        """
        (self.distribution, self.concentration) = self.strategy.collide_pairs(
            self.distribution, self.concentration, self.density, indices
        )
