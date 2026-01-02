"""Provide condensation strategies for aerosol mass transfer.

This module defines the abstract :class:`CondensationStrategy` base class
alongside concrete implementations that compute mass transfer rates for
isothermal and staggered condensation updates. Helpers cover Knudsen-number
calculations, vapor transition corrections, and batch-aware mass adjustments
that keep updates stable when radii approach continuum limits.

References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
      Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.
    - Topping, D. & Bane, M. (2022). Introduction to Aerosol Modelling. Wiley.
      https://doi.org/10.1002/9781119625728
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k,
    get_mass_transfer,
    get_mass_transfer_rate,
)
from particula.gas import get_molecule_mean_free_path
from particula.gas.species import GasSpecies
from particula.particles import (
    get_knudsen_number,
    get_partial_pressure_delta,
    get_vapor_transition_correction,
)

# particula imports
from particula.particles.representation import ParticleRepresentation
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")

# Minimum particle radius (0.1 nm) below which continuum mechanics breaks down
# and condensation equations are no longer physically valid
MIN_PARTICLE_RADIUS_M = 1e-10  # meters


# mass transfer abstract class
class CondensationStrategy(ABC):
    """Abstract base class for condensation strategies.

    This class defines the interface for various condensation models
    used in atmospheric physics. Subclasses should implement specific
    condensation algorithms based on different physical models and equations.

    Attributes:
        - molar_mass : The molar mass of the species [kg/mol].
        - diffusion_coefficient : The diffusion coefficient [m^2/s].
        - accommodation_coefficient : The mass accommodation coefficient
          (unitless).
        - update_gases : Whether to update gas concentrations after
          condensation.
        - skip_partitioning_indices : Optional list of indices for species
          that should not partition during condensation (default is None).

    Methods:
    - mean_free_path : Calculate the mean free path of the gas molecules.
    - knudsen_number : Compute the Knudsen number for a given particle radius.
    - first_order_mass_transport : Calculate first-order mass transport
        coefficient.
    - calculate_pressure_delta : Compute the partial pressure difference.
    - mass_transfer_rate : Abstract method for the mass transfer rate [kg/s].
    - rate : Abstract method for condensation rate per particle/bin.
    - step : Abstract method to perform one timestep of condensation.

    Examples:
        ```py title="Example Usage of CondensationStrategy"
        import particula as par
        strategy = par.dynamics.ConcreteCondensationStrategy(...)
        # Use strategy.mass_transfer_rate(...) to get the transfer rate
        ```

    References:
    - Seinfeld, J. H. & Pandis, S. N. (2016). Atmospheric Chemistry and
      Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.
    """

    # pylint: disable=R0913, R0917
    @validate_inputs(
        {
            "molar_mass": "positive",
            "diffusion_coefficient": "positive",
            "accommodation_coefficient": "nonnegative",
            "skip_partitioning_indices": "nonnegative",
        }
    )
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
        skip_partitioning_indices: Optional[Sequence[int]] = None,
    ):
        """Initialize the CondensationStrategy instance.

        Args:
            molar_mass: Molar mass of the species [kg/mol].
            diffusion_coefficient: Diffusion coefficient [m^2/s].
            accommodation_coefficient: Mass accommodation coefficient
                (unitless).
            update_gases: Flag indicating whether gas concentrations should
                be updated on condensation.
            skip_partitioning_indices: Optional list of indices for species
                that should not partition during condensation (default is None).
        """
        self.molar_mass = molar_mass
        self.diffusion_coefficient = diffusion_coefficient
        self.accommodation_coefficient = accommodation_coefficient
        self.update_gases = update_gases
        self.skip_partitioning_indices = skip_partitioning_indices

    def mean_free_path(
        self,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the mean free path of the gas molecules based on the
        temperature, pressure, and dynamic viscosity of the gas.

        Arguments:
            - temperature : The temperature of the gas [K].
            - pressure : The pressure of the gas [Pa].
            - dynamic_viscosity : The dynamic viscosity of the gas [Pa*s]. If
                not provided, it will be calculated based on the temperature

        Returns:
            The mean free path of the gas molecules in meters (m).

        References:
        - Mean Free Path
            [Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)

        Examples:
            ```py title="Example – Mean-free-path"
            cond = CondensationIsothermal(molar_mass=0.018)  # water vapour
            lam = cond.mean_free_path(temperature=298.15, pressure=101325)
            print(f"{lam:.2e} m")
            ```
        """
        return get_molecule_mean_free_path(
            molar_mass=self.molar_mass,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )

    def knudsen_number(
        self,
        radius: Union[float, NDArray[np.float64]],
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """The Knudsen number for a particle.

        Calculate the Knudsen number based on the mean free path of the gas
        molecules and the radius of the particle.

        Arguments:
            - radius : The radius of the particle [m].
            - temperature : The temperature of the gas [K].
            - pressure : The pressure of the gas [Pa].
            - dynamic_viscosity : The dynamic viscosity of the gas [Pa*s]. If
                not provided, it will be calculated based on the temperature

        Returns:
            The Knudsen number, which is the ratio of the mean free path to
                the particle radius.

        References:
            - [Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)

        Examples:
            ```py title="Example – Knudsen number"
            cond = CondensationIsothermal(molar_mass=0.018)
            kn = cond.knudsen_number(
                radius=1e-7, temperature=298.15, pressure=101325
            )
            ```
        """
        return get_knudsen_number(
            mean_free_path=self.mean_free_path(
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity,
            ),
            particle_radius=radius,
        )

    def first_order_mass_transport(
        self,
        particle_radius: Union[float, NDArray[np.float64]],
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """First-order mass transport coefficient per particle.

        Calculate the first-order mass transport coefficient, K, for a given
        particle based on the diffusion coefficient, radius, and vapor
        transition correction factor.

        Arguments:
            - radius : The radius of the particle [m].
            - temperature : The temperature at which the first-order mass
                transport coefficient is to be calculated.
            - pressure : The pressure of the gas phase.
            - dynamic_viscosity : The dynamic viscosity of the gas [Pa*s]. If
                not provided, it will be calculated based on the temperature

        Returns:
            The first-order mass transport coefficient per particle (m^3/s).

        References:
        - Chapter 2, Equation 2.49 (excluding particle number)
        - Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
            (D. Topping & M. Bane, Eds.). Wiley.
            [DOI](https://doi.org/10.1002/9781119625728)

        Examples:
            ```py title="Example – First-order mass-transport"
            cond = CondensationIsothermal(molar_mass=0.018)
            k = cond.first_order_mass_transport(
                particle_radius=1e-7,
                temperature=298.15,
                pressure=101325,
            )
            ```
        """
        vapor_transition = get_vapor_transition_correction(
            knudsen_number=self.knudsen_number(
                radius=particle_radius,
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity,
            ),
            mass_accommodation=self.accommodation_coefficient,
        )
        return get_first_order_mass_transport_k(
            particle_radius=particle_radius,
            vapor_transition=vapor_transition,
            diffusion_coefficient=self.diffusion_coefficient,
        )

    def _fill_zero_radius(
        self, radius: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Fill zero radius values with the maximum radius. The concentration
        value of zero will ensure that the rate of condensation is zero. The
        fill is necessary to avoid division by zero in the array operations.

        Arguments:
            - radius : The radius of the particles.

        Returns:
            - radius : The radius of the particles with zero values filled.

        Raises:
            - Warning : If all radius values are zero.

        Examples:
            ```py title="Example – Fill zero radii"
            r = np.array([0.0, 5e-8, 1e-7])
            filled = self._fill_zero_radius(r)
            ```
        """
        if np.max(radius) == 0.0:
            message = (
                "All radius values are zero, radius set to 1 m for "
                "condensation calculations. This should be ignored as the "
                "particle concentration would also be zero."
            )
            logger.warning(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            radius = np.where(radius == 0, 1, radius)
        return np.where(radius == 0, np.max(radius), radius)

    def calculate_pressure_delta(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        radius: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the difference in partial pressure between the gas and
        particle phases.

        Arguments:
            - particle : The particle for which the partial pressure difference
                is to be calculated.
            - gas_species : The gas species with which the particle is in
                contact.
            - temperature : The temperature at which the partial pressure
                difference is to be calculated.
            - radius : The radius of the particles.

        Returns:
            - partial_pressure_delta : The difference in partial pressure
                between the gas and particle phases.

        Examples:
            ```py title="Example – Δp calculation"
            delta_p = cond.calculate_pressure_delta(
                particle=particle,
                gas_species=gas_species,
                temperature=298.15,
                radius=particle.get_radius(),
            )
            ```
        """
        mass_concentration_in_particle = particle.get_species_mass()
        pure_vapor_pressure = gas_species.get_pure_vapor_pressure(
            temperature=temperature
        )
        partial_pressure_particle = np.asarray(
            particle.activity.partial_pressure(
                pure_vapor_pressure=pure_vapor_pressure,
                mass_concentration=mass_concentration_in_particle,
            )
        )
        if (
            partial_pressure_particle.ndim == 2
            and partial_pressure_particle.shape[1] == 1
        ):
            partial_pressure_particle = partial_pressure_particle[:, 0]

        partial_pressure_gas = gas_species.get_partial_pressure(temperature)
        kelvin_term = particle.surface.kelvin_term(
            radius=radius,
            molar_mass=self.molar_mass,
            mass_concentration=mass_concentration_in_particle,
            temperature=temperature,
        )

        pressure_delta = get_partial_pressure_delta(
            partial_pressure_gas=partial_pressure_gas,
            partial_pressure_particle=partial_pressure_particle,
            kelvin_term=kelvin_term,
        )

        # Ensure return is always an array
        if isinstance(pressure_delta, (int, float)):
            return np.array([pressure_delta])
        return pressure_delta

    def _apply_skip_partitioning(
        self, array: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Zero-out entries for species indices that are configured NOT to
        partition/condense.

        Arguments:
            - array : 1-D or 2-D array whose elements correspond to gas-species
              order used by this strategy.

        Returns:
            - Same array object with the chosen columns / elements set to zero.

        Examples:
            ```py title="Example – Skip selected species"
            arr = np.ones((3, 5))
            cond.skip_partitioning_indices = [1, 3]
            masked = cond._apply_skip_partitioning(arr.copy())
            ```
        """
        if self.skip_partitioning_indices is None:
            return array
        if array.ndim == 2:
            array[:, self.skip_partitioning_indices] = 0.0
        else:
            array[self.skip_partitioning_indices] = 0.0
        return array

    @abstractmethod
    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        # pylint: disable=too-many-positional-arguments, too-many-arguments
        """Compute the isothermal mass transfer rate for a particle.

        Implements dm/dt = 4π × r × Dᵢ × Mᵢ × f(Kn, α) × Δpᵢ / (R × T),
        where:
        - r is the particle radius,
        - Dᵢ is diffusion coefficient,
        - Mᵢ is molar mass,
        - f(Kn, α) is the transition correction factor,
        - Δpᵢ is the difference in partial pressure,
        - R is the gas constant,
        - T is temperature in Kelvin.

        Arguments:
            - particle : The particle representation, providing radius,
              concentration, etc.
            - gas_species : The gas species condensing onto the particles.
            - temperature : System temperature [K].
            - pressure : System pressure [Pa].
            - dynamic_viscosity : Optional dynamic viscosity [Pa*s].

        Returns:
            - Mass transfer rate [kg/s] for each particle.

        Examples:
            ```py title="Example Usage of mass_transfer_rate"
            m_rate = iso_cond.mass_transfer_rate(
                particle, gas_species, 298.15, 101325
            )
            ```
        """

    @abstractmethod
    def rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute the net condensation rate per particle, scaled by
        concentration.

        Calculates the mass transfer rate and multiplies it by particle
        concentration, yielding the total mass condensation rate per particle.

        Arguments:
            - particle : ParticleRepresentation object with distribution and
              concentration.
            - gas_species : GasSpecies object for the condensing gas.
            - temperature : The absolute temperature in Kelvin.
            - pressure : The pressure in Pascals.

        Returns:
            - Condensation rate per particle or bin, in kg/s.

        Examples:
            ```py title="Example Usage of rate"
            rates = iso_cond.rate(particle, gas_species, 298.15, 101325)
            # returns array([...]) with condensation rates
            ```
        """

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    @abstractmethod
    def step(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> Tuple[ParticleRepresentation, GasSpecies]:
        """Perform one timestep of isothermal condensation on the particle.

        Calculates the mass transfer for the specified time_step and updates
        both the particle mass and the gas concentration
        (if update_gases=True).

        Arguments:
            - particle : The particle representation to update.
            - gas_species : The gas species whose concentration is reduced.
            - temperature : System temperature [K].
            - pressure : System pressure [Pa].
            - time_step : The time interval for condensation [s].

        Returns:
            - Updated ParticleRepresentation.
            - Updated GasSpecies.

        Examples:
            ```py
            updated_particle, updated_gas = iso_cond.step(
                particle, gas_species, 298.15, 101325, 1.0
            )
            ```
        """


# Define a condensation strategy with no latent heat of vaporization effect
class CondensationIsothermal(CondensationStrategy):
    """Condensation strategy under isothermal conditions.

    This class implements the isothermal condensation model, wherein
    temperature remains constant during mass transfer. It calculates
    condensation rates based on partial pressure differences, using
    no latent heat terms.

    Attributes:
        - Inherits attributes from the base CondensationStrategy:
          molar_mass, diffusion_coefficient, etc.

    Methods:
        - mass_transfer_rate : Calculate the mass transfer rate under
          isothermal conditions.
        - rate : Get the per-particle condensation rate, accounting for
          concentration.
        - step : Advance the condensation state over a given time step.

    Examples:
        ```py title="Example Usage"
        iso_cond = CondensationIsothermal(molar_mass=0.018)
        rate_array = iso_cond.rate(particle, gas_species, 298.15, 101325)
        # rate_array now contains the condensation rate per particle
        ```

    References:
        - Aerosol Modeling, Chapter 2, Equation 2.40
        - Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
            (D. Topping & M. Bane, Eds.). Wiley.
            [DOI](https://doi.org/10.1002/9781119625728)
        - Seinfeld & Pandis, "Atmospheric Chemistry and Physics," 3rd Ed.,
          Wiley, 2016.
    """

    # pylint: disable=R0913, R0917
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
        skip_partitioning_indices: Optional[Sequence[int]] = None,
    ):
        """Initialize the CondensationIsothermal strategy.

        Args:
            molar_mass: Molar mass of the species [kg/mol].
            diffusion_coefficient: Diffusion coefficient [m^2/s].
            accommodation_coefficient: Mass accommodation coefficient.
            update_gases: Whether to update gas concentrations on update.
            skip_partitioning_indices: Species indices that should skip
                partitioning.
        """
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient,
            update_gases=update_gases,
            skip_partitioning_indices=skip_partitioning_indices,
        )

    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        # pylint: disable=too-many-positional-arguments, too-many-arguments
        """Compute the isothermal mass transfer rate per particle.

        Particle radii are filled for zeros, clipped to the minimum valid
        radius, and the resulting pressure delta is converted to a mass transfer
        rate while discarding non-finite values.

        Args:
            particle: Particle representation providing radius and activity
                information.
            gas_species: Gas species supplying vapor properties and
                concentrations.
            temperature: System temperature in Kelvin.
            pressure: System pressure in Pascals.
            dynamic_viscosity: Optional dynamic viscosity passed to the first-
                order transport calculation.

        Returns:
            Mass transfer rate per particle and per species in kg/s.

        Examples:
            ```py title="Example – Mass-transfer rate"
            m_rate = iso_cond.mass_transfer_rate(
                particle, gas_species, 298.15, 101325
            )
            ```
        """
        radius_with_fill = self._fill_zero_radius(particle.get_radius())

        # Clip radii to minimum physical size
        # Below MIN_PARTICLE_RADIUS_M, condensation equations are not valid
        radius_with_fill = np.maximum(radius_with_fill, MIN_PARTICLE_RADIUS_M)

        first_order_mass_transport = self.first_order_mass_transport(
            particle_radius=radius_with_fill,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        pressure_delta = self.calculate_pressure_delta(
            particle, gas_species, temperature, radius_with_fill
        )

        # Replace all non-finite values (±inf, NaN) with 0.0
        # Infinite pressure_delta indicates numerical instability for
        # very small particles where Kelvin effect dominates.
        # Setting to 0 effectively treats condensation as negligible
        # for these extreme cases, which is physically reasonable since
        # continuum mechanics breaks down below 0.1 nm anyway.
        pressure_delta = np.nan_to_num(
            pressure_delta, posinf=0.0, neginf=0.0, nan=0.0
        )

        return get_mass_transfer_rate(
            pressure_delta=pressure_delta,
            first_order_mass_transport=first_order_mass_transport,
            temperature=temperature,
            molar_mass=self.molar_mass,
        )

    def rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute the condensation rate per particle or bin.

        Mass transfer rates are multiplied by particle concentration, optional
        skip-partitioning is applied, and the result is returned as an array
        matching the particle inventory shape.

        Args:
            particle: Particle representation supplying concentration data.
            gas_species: Gas species providing vapor properties.
            temperature: System temperature in Kelvin.
            pressure: System pressure in Pascals.

        Returns:
            Condensation rate in kg/s per particle or bin.

        Examples:
            ```py title="Example – Condensation rate array"
            rates = iso_cond.rate(particle, gas_species, 298.15, 101325)
            ```
        """
        # Step 1: Calculate the mass transfer rate due to condensation
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )

        # Step 2: Reshape the particle concentration if necessary
        # Type guard: ensure mass_rate is an array before checking ndim
        if isinstance(mass_rate, np.ndarray) and mass_rate.ndim == 2:
            concentration = particle.concentration[:, np.newaxis]
        else:
            concentration = particle.concentration

        # Step 3: Calculate the overall condensation rate by scaling
        # mass rate by particle concentration
        rates_raw = mass_rate * concentration

        # Ensure rates is an array (scalar * array or array * array -> array)
        if not isinstance(rates_raw, np.ndarray):
            rates = np.asarray(rates_raw)
        else:
            rates = rates_raw

        # Apply optional skipping of selected species
        rates = self._apply_skip_partitioning(rates)
        return rates

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def step(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> Tuple[ParticleRepresentation, GasSpecies]:
        """Advance the simulation one timestep using isothermal condensation.

        The mass transfer rate is computed, optional skip-partitioning applied,
        and both the particle and gas states are updated while respecting
        inventory limits.

        Args:
            particle: Particle representation to advance.
            gas_species: Gas species whose concentration may decrease.
            temperature: System temperature in Kelvin.
            pressure: System pressure in Pascals.
            time_step: Duration of the time step in seconds.

        Returns:
            Tuple of the updated particle and gas species objects.

        Examples:
            ```py
            updated_particle, updated_gas = iso_cond.step(
                particle, gas_species, 298.15, 101325, 1.0
            )
            ```
        """
        # Calculate the mass transfer rate
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )

        # Type guard: ensure mass_rate is an array
        if isinstance(mass_rate, (int, float)):
            mass_rate_array = np.array([mass_rate])
        else:
            mass_rate_array = mass_rate

        # Apply optional skipping of selected species
        mass_rate_array = self._apply_skip_partitioning(mass_rate_array)

        # calculate the mass gain or loss per bin
        gas_mass_array: NDArray[np.float64] = np.atleast_1d(
            np.asarray(gas_species.get_concentration(), dtype=np.float64)
        )
        mass_transfer = get_mass_transfer(
            mass_rate=mass_rate_array,
            time_step=time_step,
            gas_mass=gas_mass_array,
            particle_mass=particle.get_species_mass(),
            particle_concentration=particle.get_concentration(),
        )
        species_mass = particle.get_species_mass()
        # Handle both 1D (single species) and 2D (multi-species) arrays
        if species_mass.ndim == 1:
            species_count = 1
        else:
            species_count = species_mass.shape[1]
        if mass_transfer.ndim == 1:
            mass_transfer = mass_transfer.reshape(-1, species_count)
        elif mass_transfer.shape[1] > species_count:
            mass_transfer = mass_transfer[:, :species_count]
        elif mass_transfer.shape[1] < species_count:
            mass_transfer = np.broadcast_to(
                mass_transfer, (mass_transfer.shape[0], species_count)
            )
        # apply the mass change
        particle.add_mass(added_mass=mass_transfer)
        if self.update_gases:
            # remove mass from gas phase concentration
            gas_species.add_concentration(
                added_concentration=-mass_transfer.sum(axis=0)
            )
        return particle, gas_species


class CondensationIsothermalStaggered(CondensationStrategy):
    """Staggered condensation strategy with two-pass batching.

    Implements Gauss-Seidel style staggering that splits timesteps across two
    passes. Supports theta modes ("half", "random", "batch"), validates batch
    counts to avoid empty batches, and reuses the stored random state for
    deterministic shuffling when requested. Mass transfer computations clamp
    radii to ``MIN_PARTICLE_RADIUS_M`` and clip non-finite deltas to keep
    updates stable near continuum limits.

    Attributes:
        theta_mode: Stepping mode, one of "half", "random", or "batch".
        num_batches: Number of batches for staggered updates (>= 1).
        shuffle_each_step: Whether to reshuffle particle order each step.
        random_state: Optional seed or generator for reproducibility.
    """

    VALID_THETA_MODES = ("half", "random", "batch")

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        theta_mode: str = "half",
        num_batches: int = 1,
        shuffle_each_step: bool = True,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None,
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
        skip_partitioning_indices: Optional[Sequence[int]] = None,
    ):
        """Initialize the staggered condensation strategy.

        Args:
            molar_mass: Molar mass of the condensing species [kg/mol].
            theta_mode: Staggered stepping mode; must be one of
                ``("half", "random", "batch")``.
            num_batches: Number of batches for Gauss-Seidel style updates;
                must be at least 1. Values larger than the particle count are
                clipped with an informational log to avoid empty batches.
            shuffle_each_step: Whether to shuffle particle order every step.
            random_state: Optional seed or generator controlling randomness
                for staggered permutations.
            diffusion_coefficient: Diffusion coefficient [m^2/s].
            accommodation_coefficient: Mass accommodation coefficient.
            update_gases: Whether to update gas concentrations.
            skip_partitioning_indices: Species indices to skip partitioning.

        Raises:
            ValueError: If ``theta_mode`` is unsupported or ``num_batches`` is
                less than 1.
        """
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient,
            update_gases=update_gases,
            skip_partitioning_indices=skip_partitioning_indices,
        )

        if theta_mode not in self.VALID_THETA_MODES:
            raise ValueError(
                f"theta_mode must be one of {self.VALID_THETA_MODES}, got "
                f"'{theta_mode}'"
            )
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1.")

        self.theta_mode = theta_mode
        self.num_batches = num_batches
        self.shuffle_each_step = shuffle_each_step
        self.random_state = random_state

    def _get_theta_values(self, n_particles: int) -> NDArray[np.float64]:
        """Generate theta values for staggered condensation steps.

        Theta values split the timestep across two passes. The first pass uses
        ``theta`` and the second uses ``1 - theta``. Values depend on the
        configured ``theta_mode`` and reuse the stored random state for
        reproducibility in random mode.

        Args:
            n_particles: Number of particles requiring theta values.

        Returns:
            Array of shape ``(n_particles,)`` with fractional step values:
            - ``half`` mode returns all ``0.5`` values.
            - ``random`` mode draws uniform values in ``[0, 1]`` using the
              stored ``random_state`` (seed, ``Generator``, or
              ``RandomState``).
            - ``batch`` mode returns all ``1.0``; batching is handled
              elsewhere.

        Raises:
            ValueError: If ``theta_mode`` is unsupported.
        """
        if self.theta_mode == "half":
            return np.full(n_particles, 0.5, dtype=np.float64)

        if self.theta_mode == "random":
            if isinstance(self.random_state, np.random.Generator):
                rng = self.random_state
                return rng.uniform(0.0, 1.0, n_particles)
            if isinstance(self.random_state, np.random.RandomState):
                return self.random_state.random(n_particles).astype(
                    np.float64, copy=False
                )

            rng = np.random.default_rng(self.random_state)
            return rng.uniform(0.0, 1.0, n_particles)

        if self.theta_mode == "batch":
            return np.ones(n_particles, dtype=np.float64)

        raise ValueError(f"Invalid theta_mode: {self.theta_mode}")

    def _validate_num_batches(self, num_batches: int, n_particles: int) -> int:
        """Validate and clip ``num_batches`` for batching.

        Args:
            num_batches: Requested number of batches.
            n_particles: Number of particles available for batching.

        Returns:
            A valid batch count respecting ``n_particles``. Returns the
            clipped value when ``num_batches`` exceeds ``n_particles``.

        Raises:
            ValueError: If ``num_batches`` is less than 1.

        Notes:
            Logs an ``INFO`` message when clipping occurs. When ``n_particles``
            is zero the caller short-circuits before invoking this helper, so
            no logging occurs for the empty case.
        """
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1.")

        if num_batches > n_particles and n_particles > 0:
            logger.info(
                "Clipping num_batches from %s to %s to avoid empty batches",
                num_batches,
                n_particles,
            )
            # Clipping avoids empty batches while preserving all particles.
            return n_particles

        return num_batches

    def _make_batches(self, n_particles: int) -> list[NDArray[np.intp]]:
        """Divide particle indices into batches for Gauss-Seidel updates.

        Creates batches of particle indices for sequential processing.
        Optionally shuffles the order before batching for randomized updates.

        Args:
            n_particles: Number of particles in the simulation.

        Returns:
            List of arrays, each containing particle indices for one batch.

        Notes:
            - Returns an empty list when ``n_particles`` is zero.
            - Clips and logs when ``num_batches`` exceeds ``n_particles`` to
              avoid empty batches.
            - When ``shuffle_each_step`` is True, indices are shuffled using
              the stored ``random_state`` (``Generator``, ``RandomState``, or
              seed), otherwise the original ordering is preserved.
        """
        if n_particles == 0:
            return []

        effective_batches = self._validate_num_batches(
            self.num_batches, n_particles
        )

        indices = np.arange(n_particles, dtype=np.intp)

        if self.shuffle_each_step:
            if isinstance(self.random_state, np.random.Generator):
                self.random_state.shuffle(indices)
            elif isinstance(self.random_state, np.random.RandomState):
                self.random_state.shuffle(indices)
            else:
                rng = np.random.default_rng(self.random_state)
                rng.shuffle(indices)

        return list(np.array_split(indices, effective_batches))

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """Compute mass transfer rate for staggered condensation.

        Mirrors the isothermal flow while leaving skip-partitioning to callers.
        Radii are filled and clipped to ``MIN_PARTICLE_RADIUS_M`` before
        transport is computed, pressure deltas are converted to rates, and any
        non-finite deltas are zeroed to avoid propagating NaNs or infinities.

        Args:
            particle: Particle representation providing radii and masses.
            gas_species: Gas species with vapor properties and concentrations.
            temperature: System temperature in kelvin.
            pressure: System pressure in pascals.
            dynamic_viscosity: Optional gas viscosity forwarded to
                :meth:`first_order_mass_transport`.

        Returns:
            Mass transfer rate per particle per species (kg/s), shaped like
            ``particle.get_species_mass()``.
        """
        radius_with_fill = np.maximum(
            self._fill_zero_radius(particle.get_radius()), MIN_PARTICLE_RADIUS_M
        )
        first_order_mass_transport = self.first_order_mass_transport(
            particle_radius=radius_with_fill,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        pressure_delta = self.calculate_pressure_delta(
            particle, gas_species, temperature, radius_with_fill
        )
        pressure_delta = np.nan_to_num(
            pressure_delta, posinf=0.0, neginf=0.0, nan=0.0
        )
        return get_mass_transfer_rate(
            pressure_delta=pressure_delta,
            first_order_mass_transport=first_order_mass_transport,
            temperature=temperature,
            molar_mass=self.molar_mass,
        )

    def rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute staggered condensation rate per particle.

        Args:
            particle: Particle representation supplying concentrations.
            gas_species: Gas species with vapor properties and concentrations.
            temperature: System temperature in kelvin.
            pressure: System pressure in pascals.

        Returns:
            Condensation/evaporation rate (kg/s) scaled by particle
            concentration with skip-partitioning applied.
        """
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )

        if isinstance(mass_rate, np.ndarray) and mass_rate.ndim == 2:
            concentration = particle.concentration[:, np.newaxis]
        else:
            concentration = particle.concentration

        rates_raw = mass_rate * concentration
        rates = (
            np.asarray(rates_raw)
            if not isinstance(rates_raw, np.ndarray)
            else rates_raw
        )
        return self._apply_skip_partitioning(rates)

    def _calculate_single_particle_transfer(
        self,
        particle: ParticleRepresentation,
        particle_index: int,
        gas_species: GasSpecies,
        gas_concentration: NDArray[np.float64],
        temperature: float,
        pressure: float,
        dt_local: float,
        radii: Optional[NDArray[np.float64]] = None,
        first_order_mass_transport: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Calculate mass change for one particle without mutating inputs.

        Uses a working gas concentration array to compute pressure deltas and
        mass-transfer rates, applies nan-safe clipping to pressure deltas, and
        enforces the minimum particle radius before evaluating transport.
        Inventory limits are applied via :func:`get_mass_transfer`. Optional
        precomputed radii and first-order transport coefficients can be reused
        across passes to avoid duplicate work.

        Args:
            particle: Particle representation with distribution and activity.
            particle_index: Index of the particle being updated.
            gas_species: Gas species object providing vapor strategies.
            gas_concentration: Working gas concentration array (kg/m^3).
            temperature: System temperature in kelvin.
            pressure: System pressure in pascals.
            dt_local: Local timestep for this particle in seconds.
            radii: Optional precomputed radii array to reuse.
            first_order_mass_transport: Optional precomputed transport
                coefficients.

        Returns:
            Per-species mass change for the particle (kg), shaped
            ``(n_species,)``.
        """
        particle_mass = particle.get_species_mass()[particle_index]
        particle_concentration = np.asarray(
            particle.concentration[particle_index]
        )
        gas_mass = np.asarray(gas_concentration, dtype=np.float64)

        radius = (
            radii[particle_index]
            if radii is not None
            else particle.get_radius()[particle_index]
        )
        radius = np.maximum(radius, MIN_PARTICLE_RADIUS_M)

        if first_order_mass_transport is None:
            mass_transport = self.first_order_mass_transport(
                particle_radius=radius,
                temperature=temperature,
                pressure=pressure,
            )
        elif np.ndim(first_order_mass_transport) == 0:
            mass_transport = first_order_mass_transport
        else:
            mass_transport = first_order_mass_transport[particle_index]

        pure_vapor_pressure = gas_species.get_pure_vapor_pressure(temperature)
        partial_pressure_particle = np.asarray(
            particle.activity.partial_pressure(
                pure_vapor_pressure=pure_vapor_pressure,
                mass_concentration=particle_mass,
            )
        )
        if (
            partial_pressure_particle.ndim == 2
            and partial_pressure_particle.shape[1] == 1
        ):
            partial_pressure_particle = partial_pressure_particle[:, 0]

        vapor_strategy = gas_species.pure_vapor_pressure_strategy
        molar_mass = gas_species.molar_mass
        if isinstance(vapor_strategy, list):
            partial_pressure_gas = np.array(
                [
                    strategy.partial_pressure(
                        concentration=gas_mass[idx],
                        molar_mass=molar_mass[idx],  # type: ignore[index]
                        temperature=temperature,
                    )
                    for idx, strategy in enumerate(vapor_strategy)
                ],
                dtype=np.float64,
            )
        else:
            partial_pressure_gas = np.asarray(
                vapor_strategy.partial_pressure(
                    concentration=gas_mass,
                    molar_mass=molar_mass,
                    temperature=temperature,
                ),
                dtype=np.float64,
            )

        kelvin_term = particle.surface.kelvin_term(
            radius=radius,
            molar_mass=self.molar_mass,
            mass_concentration=particle_mass,
            temperature=temperature,
        )
        pressure_delta = get_partial_pressure_delta(
            partial_pressure_gas=partial_pressure_gas,
            partial_pressure_particle=partial_pressure_particle,
            kelvin_term=kelvin_term,
        )
        pressure_delta = np.nan_to_num(
            pressure_delta, posinf=0.0, neginf=0.0, nan=0.0
        )

        mass_rate = get_mass_transfer_rate(
            pressure_delta=pressure_delta,
            first_order_mass_transport=mass_transport,
            temperature=temperature,
            molar_mass=self.molar_mass,
        )
        # Reshape single-particle data to (1, n_species) for get_mass_transfer
        mass_rate_2d = np.atleast_2d(np.asarray(mass_rate))
        particle_mass_2d = np.atleast_2d(particle_mass)
        particle_conc_1d = np.atleast_1d(particle_concentration)
        gas_mass_1d = np.atleast_1d(gas_mass)

        mass_to_change = get_mass_transfer(
            mass_rate=mass_rate_2d,
            time_step=dt_local,
            gas_mass=gas_mass_1d,
            particle_mass=particle_mass_2d,
            particle_concentration=particle_conc_1d,
        )
        # Squeeze back to 1D (n_species,) for single particle
        return mass_to_change.squeeze()

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def step(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> Tuple[ParticleRepresentation, GasSpecies]:
        """Perform two-pass staggered condensation update.

        The timestep is split into two passes using per-particle ``theta``
        values. Each pass iterates over batches of particles, accumulating mass
        changes, updating a working gas concentration after each batch (Gauss-
        Seidel style), and deferring mutation of the particle and gas objects
        until the end. Batch counts are validated and clipped to avoid empty
        batches before iteration. Gas is updated after every batch in both
        passes; when ``num_batches`` is 1 this reduces to the original single-
        batch behavior for backward compatibility. This mirrors staggered
        condensation approaches discussed by Jacobson (1997) and Riemer et al.
        (2009) for improved stability.

        Args:
            particle: Particle representation to update.
            gas_species: Gas species object providing vapor properties.
            temperature: System temperature in kelvin.
            pressure: System pressure in pascals.
            time_step: Full timestep to split across the two passes (seconds).

        Returns:
            Updated ``(particle, gas_species)`` tuple after both passes.
        """
        n_particles = particle.concentration.shape[0]
        if time_step == 0.0 or n_particles == 0:
            return particle, gas_species

        radii = np.maximum(
            self._fill_zero_radius(particle.get_radius()), MIN_PARTICLE_RADIUS_M
        )
        first_order_mass_transport = np.asarray(
            self.first_order_mass_transport(
                particle_radius=radii,
                temperature=temperature,
                pressure=pressure,
            )
        )

        theta = self._get_theta_values(n_particles)
        theta_dt_first = theta * time_step
        theta_dt_second = (1.0 - theta) * time_step
        batches = self._make_batches(n_particles)

        working_gas_concentration = np.asarray(
            gas_species.get_concentration(), dtype=np.float64
        ).copy()
        mass_changes = np.zeros_like(particle.get_species_mass())
        batch_dm_total = np.zeros_like(working_gas_concentration)

        for batch in batches:
            batch_dm_total.fill(0.0)
            # Gauss-Seidel: update gas after each batch in this pass.
            for particle_index in batch:
                dt_local = theta_dt_first[particle_index]
                if dt_local <= 0.0:
                    continue
                mass_change = self._calculate_single_particle_transfer(
                    particle=particle,
                    particle_index=int(particle_index),
                    gas_species=gas_species,
                    gas_concentration=working_gas_concentration,
                    temperature=temperature,
                    pressure=pressure,
                    dt_local=float(dt_local),
                    radii=radii,
                    first_order_mass_transport=first_order_mass_transport,
                )
                mass_changes[particle_index] += mass_change
                batch_dm_total += mass_change
            working_gas_concentration = np.maximum(
                working_gas_concentration - batch_dm_total, 0.0
            )

        for batch in batches:
            batch_dm_total.fill(0.0)
            # Second pass updates gas after each batch for parity with
            # num_batches=1 behavior.
            for particle_index in batch:
                dt_local = theta_dt_second[particle_index]
                if dt_local <= 0.0:
                    continue
                mass_change = self._calculate_single_particle_transfer(
                    particle=particle,
                    particle_index=int(particle_index),
                    gas_species=gas_species,
                    gas_concentration=working_gas_concentration,
                    temperature=temperature,
                    pressure=pressure,
                    dt_local=float(dt_local),
                    radii=radii,
                    first_order_mass_transport=first_order_mass_transport,
                )
                mass_changes[particle_index] += mass_change
                batch_dm_total += mass_change
            working_gas_concentration = np.maximum(
                working_gas_concentration - batch_dm_total, 0.0
            )

        mass_changes = self._apply_skip_partitioning(mass_changes)
        particle.add_mass(added_mass=mass_changes)
        if self.update_gases:
            gas_species.add_concentration(
                added_concentration=-mass_changes.sum(axis=0)
            )
        return particle, gas_species
