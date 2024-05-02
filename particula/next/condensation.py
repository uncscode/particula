"""Particle Vapor Equilibrium, condensation and evaporation
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
from typing import Union, Optional
from numpy.typing import NDArray
import numpy as np

# particula imports
from particula.next.particle import Particle
from particula.next.gas import GasSpecies
from particula.constants import GAS_CONSTANT
from particula.util.knudsen_number import calculate_knudsen_number
from particula.util.mean_free_path import molecule_mean_free_path


# define Suchs and Futugin transition function
def vapor_transition_correction(
    knudsen_number: Union[float, NDArray[np.float_]],
    mass_accommodation: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the transition correction factor, f(Kn, alpha), for a given
    Knudsen number and mass accommodation coefficient. This function is used to
    account for the intermediate regime between continuum and free molecular
    flow. This is the Suchs and Futugin transition function.

    Args:
    -----
    - knudsen_number (Union[float, NDArray[np.float_]]): The Knudsen number,
    which quantifies the relative importance of the mean free path of gas
    molecules to the size of the particle.
    - mass_accommodation (Union[float, NDArray[np.float_]]): The mass
    accommodation coefficient, representing the probability of a gas molecule
    sticking to the particle upon collision.

    Returns:
    --------
    - Union[float, NDArray[np.float_]]: The transition correction value
    calculated based on the specified inputs.

    References:
    ----------
    - Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Chapter 12,
    equation 12.43.
    Note: There are various formulations for this correction, so further
    extensions of this function might be necessary depending on specific
    requirements.

    Original reference:
    - FUCHS, N. A., & SUTUGIN, A. G. (1971). HIGH-DISPERSED AEROSOLS.
    In Topics in Current Aerosol Research (p. 1). Elsevier.
    https://doi.org/10.1016/B978-0-08-016674-2.50006-6
    """
    return (
        (0.75 * mass_accommodation * (1 + knudsen_number))
        /
        ((knudsen_number**2 + knudsen_number)
         + 0.283 * mass_accommodation * knudsen_number
         + 0.75 * mass_accommodation)
    )


def partial_pressure_delta(
    partial_pressure_gas: Union[float, NDArray[np.float_]],
    partial_pressure_particle: Union[float, NDArray[np.float_]],
    kelvin_term: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the difference in partial pressure of a species between the gas
    phase and the particle phase, which is used in the calculation of the rate
    of change of mass of an aerosol particle.

    Args:
    -----
    - partial_pressure_gas (Union[float, NDArray[np.float_]]): The partial
    pressure of the species in the gas phase.
    - partial_pressure_particle (Union[float, NDArray[np.float_]]): The partial
    pressure of the species in the particle phase.
    - kelvin_term (Union[float, NDArray[np.float_]]): Kelvin effect to account
    for the curvature of the particle.

    Returns:
    --------
    - Union[float, NDArray[np.float_]]: The difference in partial pressure
    between the gas phase and the particle phase.
    """
    return partial_pressure_gas - partial_pressure_particle * kelvin_term


def thermal_conductivity(
    temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the thermal conductivity of air as a function of temperature.

    Args:
    -----
    - temperature (Union[float, NDArray[np.float_]]): The temperature at which
    the thermal conductivity of air is to be calculated.

    Returns:
    --------
    - Union[float, NDArray[np.float_]]: The thermal conductivity of air at the
    specified temperature. Units of J/(m s K).

    References:
    ----------
    - Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 17.54.
    """
    return 1e-3 * (4.39 + 0.071 * temperature)


def first_order_mass_transport_k(
        radius: Union[float, NDArray[np.float_]],
        vapor_transition: Union[float, NDArray[np.float_]],
        diffusion_coefficient: Union[float, NDArray[np.float_]] = 2*1e-9
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the first-order mass transport coefficient, K, for a given radius
    diffusion coefficient, and vapor transition correction factor. For a
    single particle.

    Args:
    -----
    - radius (Union[float, NDArray[np.float_]]): The radius of the particle
    [m].
    - diffusion_coefficient (Union[float, NDArray[np.float_]]): The diffusion
    coefficient of the vapor [m^2/s], default to air.
    - vapor_transition (Union[float, NDArray[np.float_]]): The vapor transition
    correction factor. [unitless]

    Returns:
    --------
    - Union[float, NDArray[np.float_]]: The first-order mass transport
    coefficient per particle (m^3/s).

    References:
    ----------
    - Aerosol Modeling, Chapter 2, Equation 2.49 (excluding particle number)
    - https://en.wikipedia.org/wiki/Mass_diffusivity
    """
    return 4 * np.pi * radius * diffusion_coefficient * vapor_transition


def mass_transfer_rate(
        pressure_delta: Union[float, NDArray[np.float_]],
        first_order_mass_transport: Union[float, NDArray[np.float_]],
        temperature: Union[float, NDArray[np.float_]],
        molar_mass: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the mass transfer rate based on the difference in partial
    pressure and the first-order mass transport coefficient.

    Args:
    -----
    - pressure_delta (Union[float, NDArray[np.float_]]): The difference in
    partial pressure between the gas phase and the particle phase.
    - first_order_mass_transport (Union[float, NDArray[np.float_]]): The
    first-order mass transport coefficient per particle.
    - temperature (Union[float, NDArray[np.float_]]): The temperature at which
    the mass transfer rate is to be calculated.

    Returns:
    --------
    - Union[float, NDArray[np.float_]]: The mass transfer rate for the particle
    [kg/s].

    References:
    ----------
    - Aerosol Modeling, Chapter 2, Equation 2.41 (excluding particle number)
    - Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 13.3
    """
    return np.array(
        first_order_mass_transport * pressure_delta
        / (GAS_CONSTANT.m/molar_mass * temperature),
        dtype=np.float_
    )


# mass transfer abstract class
class CondensationStrategy(ABC):
    """
    Abstract class for mass transfer strategies, for condensation or
    evaporation of particles. This class should be subclassed to implement
    specific mass transfer strategies.

    Parameters:
    -----------
    - molar_mass (Union[float, NDArray[np.float_]]): The molar mass of the
    species [kg/mol]. If a single value is provided, it will be used for all
    species.
    - diffusion_coefficient (Union[float, NDArray[np.float_]]): The diffusion
    coefficient of the species [m^2/s]. If a single value is provided, it will
    be used for all species. Default is 2*1e-9 m^2/s for air.
    - accommodation_coefficient (Union[float, NDArray[np.float_]]): The mass
    accommodation coefficient of the species. If a single value is provided,
    it will be used for all species. Default is 1.0.
    """

    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        diffusion_coefficient: Union[float, NDArray[np.float_]] = 2*1e-9,
        accommodation_coefficient: Union[float, NDArray[np.float_]] = 1.0
    ):
        self.molar_mass = molar_mass
        self.diffusion_coefficient = diffusion_coefficient
        self.accommodation_coefficient = accommodation_coefficient
        super().__init__()

    def mean_free_path(
        self,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the mean free path of the gas molecules based on the
        temperature, pressure, and dynamic viscosity of the gas.

        Args:
        -----
        - temperature (float): The temperature of the
        gas [K].
        - pressure (float): The pressure of the gas
        [Pa].
        - dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
        [Pa*s]. If not provided, it will be calculated based on the temperature

        Returns:
        --------
        - Union[float, NDArray[np.float_]]: The mean free path of the gas
        molecules in meters (m).

        References:
        ----------
        - https://en.wikipedia.org/wiki/Mean_free_path
        """
        return molecule_mean_free_path(
            molar_mass=self.molar_mass,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity
        )

    def knudsen_number(
        self,
        radius: Union[float, NDArray[np.float_]],
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the Knudsen number based on the mean free path of the gas
        molecules and the radius of the particle.

        Args:
        -----
        - radius (Union[float, NDArray[np.float_]]): The radius of the particle
        [m].
        - temperature (float): The temperature of the gas [K].
        - pressure (float): The pressure of the gas [Pa].
        - dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
        [Pa*s]. If not provided, it will be calculated based on the temperature

        Returns:
        --------
        - Union[float, NDArray[np.float_]]: The Knudsen number, which is the
        ratio of the mean free path to the particle radius.

        References:
        ----------
        - https://en.wikipedia.org/wiki/Knudsen_number
        """
        return calculate_knudsen_number(
            mean_free_path=self.mean_free_path(
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity
            ),
            particle_radius=radius
        )

    def first_order_mass_transport(
        self,
        radius: Union[float, NDArray[np.float_]],
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the first-order mass transport coefficient, K, for a given
        particle based on the diffusion coefficient, radius, and vapor
        transition correction factor.

        Args:
        -----
        - radius (Union[float, NDArray[np.float_]]): The radius of the particle
        [m].
        - temperature (float): The temperature at which the first-order mass
        transport coefficient is to be calculated.
        - pressure (float): The pressure of the gas phase.
        - dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
        [Pa*s]. If not provided, it will be calculated based on the temperature

        Returns:
        --------
        - Union[float, NDArray[np.float_]]: The first-order mass transport
        coefficient per particle (m^3/s).

        References:
        ----------
        - Aerosol Modeling, Chapter 2, Equation 2.49 (excluding particle
        number)
        """
        vapor_transition = vapor_transition_correction(
            knudsen_number=self.knudsen_number(
                radius=radius,
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity
            ),
            mass_accommodation=self.accommodation_coefficient
        )
        return first_order_mass_transport_k(
            radius=radius,
            vapor_transition=vapor_transition,
            diffusion_coefficient=self.diffusion_coefficient,
        )

    @abstractmethod
    def mass_transfer_rate(
        self,
        particle: Particle,
        partial_pressure_gas: Union[float, NDArray[np.float_]],
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the mass transfer rate based on the difference in partial
        pressure and the first-order mass transport coefficient.

        Args:
        -----
        - particle (Particle class): The particle for which the mass transfer
        rate is to be calculated.
        - partial_pressure_gas (Union[float, NDArray[np.float_]]): The partial
        pressure of the species in the gas phase.
        - temperature (float): The temperature at which the mass transfer rate
        is to be calculated.
        - pressure (float): The pressure of the gas phase.
        - dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
        [Pa*s]. If not provided, it will be calculated based on the temperature

        Returns:
        --------
        - Union[float, NDArray[np.float_]]: The mass transfer rate for the
        particle [kg/s].
        """


# Define a condensation strategy with no latent heat of vaporization effect
class CondesnationIsothermal(CondensationStrategy):
    """
    Condensation strategy for isothermal conditions, where the temperature
    remains constant. This class implements the mass transfer rate calculation
    for condensation of particles based on partial pressures. No Latent heat
    of vaporization effect is considered.
    """
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        diffusion_coefficient: Union[float, NDArray[np.float_]] = 2*1e-9,
        accommodation_coefficient: Union[float, NDArray[np.float_]] = 1.0
    ):
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient
        )

    def mass_transfer_rate(
        self,
        particle: Particle,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the mass transfer rate based on the difference in partial
        pressure and the first-order mass transport coefficient.

        Args:
        -----
        - particle (Particle class): The particle for which the mass transfer
        rate is to be calculated.
        - partial_pressure_gas (Union[float, NDArray[np.float_]]): The partial
        pressure of the species in the gas phase.
        - temperature (float): The temperature at which the mass transfer rate
        is to be calculated.
        - pressure (float): The pressure of the gas phase.
        - dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
        [Pa*s]. If not provided, it will be calculated based on the temperature

        Returns:
        --------
        - Union[float, NDArray[np.float_]]: The mass transfer rate for the
        particle [kg/s].
        """

        # Calculate the first-order mass transport coefficient
        first_order_mass_transport = self.first_order_mass_transport(
            radius=particle.get_radius(),
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity
        )

        # Calculate the difference in partial pressure
        partial_pressure_particle = particle.activity.partial_pressure(
            pure_vapor_pressure=gas_species.get_vapor_pressure(temperature),
            mass_concentration=particle.get_mass()
        )
        partial_pressure_gas = gas_species.get_partial_pressure(temperature)
        # calculate the kelvin term
        kelvin_term = particle.surface.kelvin_term(
            radius=particle.get_radius(),
            molar_mass=self.molar_mass,
            mass_concentration=particle.get_mass(),
            temperature=temperature
        )
        # calculate the pressure delta
        pressure_delta = partial_pressure_delta(
            partial_pressure_gas=partial_pressure_gas,
            partial_pressure_particle=partial_pressure_particle,
            kelvin_term=kelvin_term
        )

        # Calculate the mass transfer rate per particle
        return mass_transfer_rate(
            pressure_delta=pressure_delta,
            first_order_mass_transport=first_order_mass_transport,
            temperature=temperature,
            molar_mass=self.molar_mass
        )
