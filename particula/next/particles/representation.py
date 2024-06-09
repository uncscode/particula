"""Particle representation."""

import numpy as np
from numpy.typing import NDArray

# From Particula
from particula.next.particles.activity_strategies import ActivityStrategy
from particula.next.particles.surface_strategies import SurfaceStrategy
from particula.next.particles.distribution_strategies import (
    DistributionStrategy,
)


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
    """

    def __init__(
        self,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_],
        concentration: NDArray[np.float_],
        charge: NDArray[np.float_],
    ):  # pylint: disable=too-many-arguments
        self.strategy = strategy
        self.activity = activity
        self.surface = surface
        self.distribution = distribution
        self.density = density
        self.concentration = concentration
        self.charge = charge

    def get_mass(self) -> NDArray[np.float_]:
        """Returns the mass of the particles as calculated by the strategy.

        Returns:
            The mass of the particles.
        """
        return self.strategy.get_mass(self.distribution, self.density)

    def get_radius(self) -> NDArray[np.float_]:
        """Returns the radius of the particles as calculated by the strategy.

        Returns:
            The radius of the particles.
        """
        return self.strategy.get_radius(self.distribution, self.density)

    def get_charge(self) -> NDArray[np.float_]:
        """Returns the charge per particle.

        Returns:
            The charge of the particles.
        """
        return self.charge

    def get_total_mass(self) -> np.float_:
        """Returns the total mass of the particles.

        The total mass is as calculated by the strategy, taking into account
        the distribution and concentration.

        Returns:
            np.float_: The total mass of the particles.
        """
        return self.strategy.get_total_mass(
            self.distribution, self.concentration, self.density
        )

    def add_mass(self, added_mass: NDArray[np.float_]) -> None:
        """Adds mass to the particle distribution, and updates parameters.

        Args:
            added_mass: The mass to be added per
            distribution bin.
        """
        (self.distribution, self.concentration) = self.strategy.add_mass(
            self.distribution, self.concentration, self.density, added_mass
        )

    def add_concentration(
        self, added_concentration: NDArray[np.float_]
    ) -> None:
        """Adds concentration to the particle distribution.

        Args:
            added_concentration: The concentration to be
            added per distribution bin.
        """
        self.concentration += added_concentration
