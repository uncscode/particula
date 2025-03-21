"""
Particle Vapor Equilibrium, condensation and evaporation
based on partial pressures to get dm/dt or other forms of
particle growth and decay.

From Seinfeld and Pandas: The Condensation (chapter 13) Equation 13.3
This function calculates the rate of change of the mass of an aerosol
particle with diameter Dp.

The rate of change of mass, dm, is given by the formula:
    dm/dt = 4π * radius * Di * Mi * f(Kn, alpha) * delta_pi / RT
where:
    radius is the radius of the particle,
    Di is the diffusion coefficient of species i,
    Mi is the molar mass of species i,
    f(Kn, alpha) is a transition function of the Knudsen number (Kn) and the
    mass accommodation coefficient (alpha),
    delta_pi is the partial pressure of species i in the gas phase vs the
    particle phase.
    R is the gas constant,
    T is the temperature.

    An additional denominator term is added to acount for temperature changes,
    which is need for cloud droplets, but can be used in general too.

This is also in Aerosol Modeling Chapter 2 Equation 2.40

Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and Physics:
From Air Pollution to Climate Change. In Wiley (3rd ed.).
John Wiley & Sons, Inc.

Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
(D. Topping & M. Bane, Eds.). Wiley. https://doi.org/10.1002/9781119625728

Units are all Base SI units.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
import logging
from numpy.typing import NDArray
import numpy as np

# particula imports
from particula.particles.representation import ParticleRepresentation
from particula.gas.species import GasSpecies
from particula.particles import (
    get_knudsen_number,
    get_vapor_transition_correction,
    get_partial_pressure_delta,
)
from particula.gas import get_molecule_mean_free_path
from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k,
    get_mass_transfer_rate,
    get_mass_transfer,
)

logger = logging.getLogger("particula")


# mass transfer abstract class
class CondensationStrategy(ABC):
    """
    Abstract base class for condensation strategies.

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

    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ):
        """
        Initialize the CondensationStrategy instance.

        Arguments:
            - molar_mass : Molar mass of the species [kg/mol].
            - diffusion_coefficient : Diffusion coefficient [m^2/s].
            - accommodation_coefficient : Mass accommodation coefficient
              (unitless).
            - update_gases : Flag indicating whether gas concentrations should
              be updated on condensation.
        """
        self.molar_mass = molar_mass
        self.diffusion_coefficient = diffusion_coefficient
        self.accommodation_coefficient = accommodation_coefficient
        self.update_gases = update_gases

    def mean_free_path(
        self,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the mean free path of the gas molecules based on the
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
        """
        if np.max(radius) == 0:
            message = (
                "All radius values are zero, radius set to 1 m for "
                "condensation calculations. This should be ignored as the "
                "particle concentration would also be zero."
            )
            logger.warning(message)
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
        """
        mass_concentration_in_particle = particle.get_species_mass()
        pure_vapor_pressure = gas_species.get_pure_vapor_pressure(
            temperature=temperature
        )
        partial_pressure_particle = particle.activity.partial_pressure(
            pure_vapor_pressure=pure_vapor_pressure,
            mass_concentration=mass_concentration_in_particle,
        )

        partial_pressure_gas = gas_species.get_partial_pressure(temperature)
        kelvin_term = particle.surface.kelvin_term(
            radius=radius,
            molar_mass=self.molar_mass,
            mass_concentration=mass_concentration_in_particle,
            temperature=temperature,
        )

        return get_partial_pressure_delta(
            partial_pressure_gas=partial_pressure_gas,
            partial_pressure_particle=partial_pressure_particle,
            kelvin_term=kelvin_term,
        )

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
        """
        Compute the isothermal mass transfer rate for a particle.

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
        """
        Compute the net condensation rate per particle, scaled by
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
        """
        Perform one timestep of isothermal condensation on the particle.

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
    """
    Condensation strategy under isothermal conditions.

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

    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ):
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient,
            update_gases=update_gases,
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

        radius_with_fill = self._fill_zero_radius(particle.get_radius())
        first_order_mass_transport = self.first_order_mass_transport(
            particle_radius=radius_with_fill,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        pressure_delta = self.calculate_pressure_delta(
            particle, gas_species, temperature, radius_with_fill
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

        # Step 1: Calculate the mass transfer rate due to condensation
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )

        # Step 2: Reshape the particle concentration if necessary
        if mass_rate.ndim == 2:  # Multiple gas species  # type: ignore
            concentration = particle.concentration[:, np.newaxis]
        else:
            concentration = particle.concentration

        # Step 3: Calculate the overall condensation rate by scaling
        # mass rate by particle concentration
        rates = mass_rate * concentration

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

        # Calculate the mass transfer rate
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        # calculate the mass gain or loss per bin
        mass_transfer = get_mass_transfer(
            mass_rate=mass_rate,  # type: ignore
            time_step=time_step,
            gas_mass=gas_species.get_concentration(),  # type: ignore
            particle_mass=particle.get_species_mass(),
            particle_concentration=particle.get_concentration(),
        )
        # apply the mass change
        particle.add_mass(added_mass=mass_transfer)
        if self.update_gases:
            # remove mass from gas phase concentration
            gas_species.add_concentration(
                added_concentration=-mass_transfer.sum(axis=0)
            )
        return particle, gas_species
