"""Particle representation."""

import numpy as np
from numpy.typing import NDArray

# From Particula
from particula.next.particles.activity_strategies import ActivityStrategy
from particula.next.particles.surface_strategies import SurfaceStrategy
from particula.next.particles.distribution_strategies import (
    DistributionStrategy,
)


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
    ):  # pylint: disable=too-many-arguments
        self.strategy = strategy
        self.activity = activity
        self.surface = surface
        self.distribution = distribution
        self.density = density
        self.concentration = concentration
        self.charge = charge
        self.volume = volume

    def get_mass(self) -> NDArray[np.float64]:
        """Returns the mass of the particles as calculated by the strategy.

        Returns:
            The mass of the particles.
        """
        return self.strategy.get_mass(self.distribution, self.density)

    def get_radius(self) -> NDArray[np.float64]:
        """Returns the radius of the particles as calculated by the strategy.

        Returns:
            The radius of the particles.
        """
        return self.strategy.get_radius(self.distribution, self.density)

    def get_charge(self) -> NDArray[np.float64]:
        """Returns the charge per particle.

        Returns:
            The charge of the particles.
        """
        return self.charge

    def get_total_mass(self) -> np.float64:
        """Returns the total mass of the particles.

        The total mass is as calculated by the strategy, taking into account
        the distribution and concentration.

        Returns:
            np.float64: The total mass of the particles.
        """
        return self.strategy.get_total_mass(
            self.distribution, self.concentration, self.density
        )

    def add_mass(self, added_mass: NDArray[np.float64]) -> None:
        """Adds mass to the particle distribution, and updates parameters.

        Args:
            added_mass: The mass to be added per
                distribution bin.
        """
        (self.distribution, self.concentration) = self.strategy.add_mass(
            self.distribution, self.concentration, self.density, added_mass
        )

    def add_concentration(
        self, added_concentration: NDArray[np.float64]
    ) -> None:
        """Adds concentration to the particle distribution.

        Args:
            added_concentration: The concentration to be
                added per distribution bin.
        """
        self.concentration += added_concentration

    def collide_pairs(
        self, indices: NDArray[np.int64]
    ) -> None:
        """Collide pairs of indices, used for ParticleResolved Strategies.

        Args:
            indices: The indices to collide.
        """
        (self.distribution, self.concentration) = self.strategy.collide_pairs(
            self.distribution, self.concentration, self.density, indices
        )
