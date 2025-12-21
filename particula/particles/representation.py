"""Particle representation for a collection of particles."""

import logging
from copy import deepcopy
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# From Particula
from particula.particles.activity_strategies import ActivityStrategy
from particula.particles.distribution_strategies import (
    DistributionStrategy,
)
from particula.particles.properties.sort_bins import (
    get_sorted_bins_by_radius,
)
from particula.particles.surface_strategies import SurfaceStrategy

logger = logging.getLogger("particula")


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class ParticleRepresentation:
    """Everything needed to represent a particle or a collection of particles.

    Represents a particle or a collection of particles, encapsulating the
    strategy for calculating mass, radius, and total mass based on a
    specified particle distribution, density, and concentration. This class
    allows for flexibility in representing particles.

    Attributes:
        - strategy : The computation strategy for particle representations.
        - activity : The activity strategy for the partial pressure
            calculations.
        - surface : The surface strategy for surface tension and Kelvin effect.
        - distribution : The distribution data for the particles, which could
            represent sizes, masses, or another relevant metric.
        - density : The density of the material from which the particles are
            made.
        - concentration : The concentration of particles within the
            distribution.
        - charge : The charge on each particle.
        - volume : The air volume for simulation of particles in the air,
            default is 1 m^3. This is only used in ParticleResolved Strategies.

    Methods:
    - get_strategy : Return the distribution strategy (optionally cloned).
    - get_strategy_name : Return the name of the distribution strategy.
    - get_activity : Return the activity strategy (optionally cloned).
    - get_activity_name : Return the name of the activity strategy.
    - get_surface : Return the surface strategy (optionally cloned).
    - get_surface_name : Return the name of the surface strategy.
    - get_distribution : Return the distribution array (optionally cloned).
    - get_density : Return the density array (optionally cloned).
    - get_concentration : Return the concentration array (optionally cloned).
    - get_total_concentration : Return the total concentration (1/m^3).
    - get_charge : Return the per-particle charge (optionally cloned).
    - get_volume : Return the representation volume in m^3 (optionally cloned).
    - get_species_mass : Return the mass per species, in kg.
    - get_mass : Return the array of total particle masses, in kg.
    - get_mass_concentration : Return the total mass concentration in kg/m^3.
    - get_radius : Return the array of particle radii in meters.
    - add_mass : Add mass to the distribution in each bin.
    - add_concentration : Add concentration to the distribution in each bin.
    - collide_pairs : Collide pairs of indices (ParticleResolved strategies).
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
        """Initialize the ParticleRepresentation.

        Sets up the particle representation with required strategies and
        properties including distribution, density, concentration, charge,
        and volume for particle calculations.
        """
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
            - A string representation of the particle representation.

        Example:
            ``` py title="Get String Representation"
            str_rep = str(particle_representation)
            print(str_rep)
            ```
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
        """Return the strategy used for particle representation.

        Arguments:
            - clone : If True, then return a deepcopy of the strategy.

        Returns:
            - The strategy used for particle
                representation.

        Example:
            ``` py title="Get Strategy"
            strategy = particle_representation.get_strategy()
            ```
        """
        if clone:
            return deepcopy(self.strategy)
        return self.strategy

    def get_strategy_name(self) -> str:
        """Return the name of the strategy used for particle representation.

        Returns:
            - The name of the strategy used for particle representation.

        Example:
            ``` py title="Get Strategy Name"
            strategy_name = particle_representation.get_strategy_name()
            print(strategy_name)
            ```
        """
        return self.strategy.get_name()

    def get_activity(self, clone: bool = False) -> ActivityStrategy:
        """Return the activity strategy used for partial pressure calculations.

        Arguments:
            - clone : If True, then return a deepcopy of the activity strategy.

        Returns:
            - The activity strategy used for partial
              pressure calculations.

        Example:
            ``` py title="Get Activity Strategy"
            activity = particle_representation.get_activity()
            ```
        """
        if clone:
            return deepcopy(self.activity)
        return self.activity

    def get_activity_name(self) -> str:
        """Return the name of the activity strategy used for partial pressure
        calculations.

        Returns:
            - The name of the activity strategy used for partial
              pressure calculations.

        Example:
            ``` py title="Get Activity Strategy Name"
            activity_name = particle_representation.get_activity_name()
            print(activity_name)
            ```
        """
        return self.activity.get_name()

    def get_surface(self, clone: bool = False) -> SurfaceStrategy:
        """Return surface strategy for surface tension and Kelvin effect.

        Arguments:
            - clone : If True, then return a deepcopy of the surface strategy.

        Returns:
            - The surface strategy used for surface tension
              and Kelvin effect.

        Example:
            ``` py title="Get Surface Strategy"
            surface = particle_representation.get_surface()
            ```
        """
        if clone:
            return deepcopy(self.surface)
        return self.surface

    def get_surface_name(self) -> str:
        """Return the name of the surface strategy used for surface tension and
        Kelvin effect.

        Returns:
            - The name of the surface strategy used for surface tension
              and Kelvin effect.

        Example:
            ``` py title="Get Surface Strategy Name"
            surface_name = particle_representation.get_surface_name()
            print(surface_name)
            ```
        """
        return self.surface.get_name()

    def get_distribution(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the distribution of the particles.

        Arguments:
            - clone : If True, then return a copy of the distribution array.

        Returns:
            - The distribution of the particles.

        Example:
            ``` py title="Get Distribution Array"
            distribution = particle_representation.get_distribution()
            ```
        """
        if clone:
            return np.copy(self.distribution)
        return self.distribution

    def get_density(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the density of the particles.

        Arguments:
            - clone : If True, then return a copy of the density array.

        Returns:
            - The density of the particles.

        Example:
            ``` py title="Get Density Array"
            density = particle_representation.get_density()
            ```
        """
        if clone:
            return np.copy(self.density)
        return self.density

    def get_effective_density(self) -> NDArray[np.float64]:
        """Return the effective density of the particles, weighted by the
        mass of the species.

        Arguments:
            - None

        Returns:
            - The effective density of the particles.

        Example:
            ``` py title="Get Effective Density Array"
            effective_density = particle_representation.get_effective_density()
            ```
        """
        densities = self.get_density()
        # if only one species is used, return the density of that species
        if isinstance(densities, float) or np.size(densities) == 1:
            return np.ones_like(self.get_species_mass()) * densities
        # calculate weighted particle density
        mass_total = self.get_mass()
        weighted_mass = np.sum(self.get_species_mass() * densities, axis=1)
        return np.divide(
            weighted_mass,
            mass_total,
            where=mass_total != 0,
            out=np.zeros_like(weighted_mass),
        )

    def get_mean_effective_density(self) -> float:
        """Return the mean effective density of the particles.

        Arguments:
            - None

        Returns:
            - The mean effective density of the particles.

        Example:
            ``` py title="Get Mean Effective Density Array"
            mean_effective_density = (
                particle_representation.get_mean_effective_density()
            )
            ```
        """
        # filter out zero densities for no mass in bin/particle
        effective_density = self.get_effective_density()
        effective_density = effective_density[effective_density != 0]
        if effective_density.size == 0:
            return 0.0
        return np.mean(effective_density)

    def get_concentration(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the volume concentration of the particles.

        For ParticleResolved Strategies, this is the number of
        particles per self.volume. Otherwise, it's per 1/m^3.

        Arguments:
            - clone : If True, then return a copy of the concentration array.

        Returns:
            - The concentration of the particles in 1/m^3.

        Example:
            ``` py title="Get Concentration Array"
            concentration = particle_representation.get_concentration()
            ```
        """
        if clone:
            return np.copy(self.concentration / self.volume)
        return self.concentration / self.volume

    def get_total_concentration(self, clone: bool = False) -> np.float64:
        """Return the total concentration of the particles.

        Arguments:
            - clone : If True, then return a copy of the concentration array.

        Returns:
            - The total number concentration of the particles in 1/m^3.

        Example:
            ``` py title="Get Total Concentration"
            total_concentration = (
                particle_representation.get_total_concentration()
            )
            print(total_concentration)
            ```
        """
        return np.sum(self.get_concentration(clone=clone))

    def get_charge(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the charge per particle.

        Arguments:
            - clone : If True, then return a copy of the charge array.

        Returns:
            - The charge of the particles (dimensionless).

        Example:
            ``` py title="Get Charge Array"
            charge = particle_representation.get_charge()
            ```
        """
        if clone:
            return np.copy(self.charge)
        return self.charge

    def get_volume(self, clone: bool = False) -> float:
        """Return the volume used for the particle representation.

        Arguments:
            - clone : If True, then return a copy of the volume value.

        Returns:
            - The volume of the particles in m^3.

        Example:
            ``` py title="Get Volume"
            volume = particle_representation.get_volume()
            ```
        """
        if clone:
            return deepcopy(self.volume)
        return self.volume

    def get_species_mass(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the masses per species in the particles.

        Arguments:
            - clone : If True, then return a copy of the computed mass array.

        Returns:
            - The mass of the particles per species in kg.

        Example:
            ``` py title="Get Species Mass"
            species_mass = particle_representation.get_species_mass()
            ```
        """
        if clone:
            return np.copy(
                self.strategy.get_species_mass(self.distribution, self.density)
            )
        return self.strategy.get_species_mass(self.distribution, self.density)

    def get_mass(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the mass of the particles as calculated by the strategy.

        Arguments:
            - clone : If True, then return a copy of the mass array.

        Returns:
            - The mass of the particles in kg.

        Example:
            ``` py title="Get Mass"
            mass = particle_representation.get_mass()
            ```
        """
        if clone:
            return np.copy(
                self.strategy.get_mass(self.distribution, self.density)
            )
        return self.strategy.get_mass(self.distribution, self.density)

    def get_mass_concentration(self, clone: bool = False) -> np.float64:
        """Return the total mass per volume of the simulated particles.

        The mass concentration is calculated from the distribution
        and concentration arrays.

        Arguments:
            - clone : If True, then return a copy of the mass concentration
              value.

        Returns:
            - The mass concentration in kg/m^3.

        Example:
            ``` py title="Get Mass Concentration"
            mass_concentration = (
                particle_representation.get_mass_concentration()
            )
            print(mass_concentration)
            ```
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
        """Return the radius of the particles as calculated by the strategy.

        Arguments:
            - clone : If True, then return a copy of the radius array.

        Returns:
            - The radius of the particles in meters.

        Example:
            ``` py title="Get Radius"
            radius = particle_representation.get_radius()
            ```
        """
        if clone:
            return np.copy(
                self.strategy.get_radius(self.distribution, self.density)
            )
        return self.strategy.get_radius(self.distribution, self.density)

    def add_mass(self, added_mass: NDArray[np.float64]) -> None:
        """Add mass to the particle distribution and update parameters.

        Arguments:
            - added_mass : The mass to be added per distribution bin, in kg.

        Example:
            ``` py title="Add Mass"
            particle_representation.add_mass(added_mass)
            ```
        """
        (self.distribution, _) = self.strategy.add_mass(
            self.get_distribution(),
            self.get_concentration(),
            self.get_density(),
            added_mass,
        )
        self._enforce_increasing_bins()

    def add_concentration(
        self,
        added_concentration: NDArray[np.float64],
        added_distribution: Optional[NDArray[np.float64]] = None,
        *,
        added_charge: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Add concentration to the particle distribution.

        Arguments:
            - added_concentration : The concentration to be added per bin
              (1/m^3).
            - added_distribution : Optional distribution array to merge into
              the existing distribution. If None, the current distribution
              is reused.
            - added_charge : Optional charge array for newly added particles.
              Defaults to zeros when charge is tracked but no values are
              provided. Ignored when charge is not tracked.

        Example:
            ``` py title="Add Concentration"
            particle_representation.add_concentration(added_concentration)
            ```
        """
        # if added_distribution is None, then it will be calculated
        if added_distribution is None:
            message = "Added distribution is value None."
            logger.warning(message)
            added_distribution = self.get_distribution()
        (
            self.distribution,
            self.concentration,
            self.charge,
        ) = self.strategy.add_concentration(
            distribution=self.get_distribution(),
            concentration=self.get_concentration(),
            added_distribution=added_distribution,
            added_concentration=added_concentration,
            charge=self.charge,
            added_charge=added_charge,
        )
        self._enforce_increasing_bins()

    def collide_pairs(self, indices: NDArray[np.int64]) -> None:
        """Collide pairs of particles, used for ParticleResolved Strategies.

        Performs coagulation between particle pairs by delegating to the
        distribution strategy's collide_pairs method. The smaller particle
        (first index in each pair) is merged into the larger particle (second
        index). Mass, concentration, and charge are all updated accordingly.

        Charge conservation is handled automatically: if the particles have
        non-zero charges, they are summed during collisions. This enables
        physically accurate charge conservation in particle-resolved
        coagulation simulations.

        Arguments:
            - indices : Array of particle pair indices to collide, shape
                (K, 2) where each row is [small_index, large_index].

        Example:
            ``` py title="Collide Pairs"
            particle_representation.collide_pairs(indices)
            ```
        """
        (self.distribution, self.concentration, self.charge) = (  # type: ignore[assignment]
            self.strategy.collide_pairs(  # type: ignore[assignment]
                self.distribution,
                self.concentration,
                self.density,
                indices,
                self.charge,
            )
        )  # type: ignore[assignment]

    def _enforce_increasing_bins(self) -> None:
        """Ensure distribution bins are sorted by increasing radius."""
        (
            self.distribution,
            self.concentration,
            self.charge,
        ) = get_sorted_bins_by_radius(  # type: ignore[assignment]
            radius=self.get_radius(),
            distribution=self.distribution,
            concentration=self.concentration,
            charge=self.charge,
        )
