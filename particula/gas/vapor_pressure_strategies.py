"""Vapor Pressure Module.

This module calculates the vapor pressure of substances based on different
strategies. These strategies are interchangeable and can be used to calculate
the vapor pressure of a substance at a given temperature.

All units are in base SI units (kg, m, s).
"""

from abc import ABC, abstractmethod
from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.gas.properties.concentration_function import (
    calculate_concentration,
)
from particula.gas.properties.pressure_function import (
    calculate_partial_pressure,
)
from particula.gas.properties.vapor_pressure_module import (
    antoine_vapor_pressure,
    buck_vapor_pressure,
    clausius_clapeyron_vapor_pressure,
)


class VaporPressureStrategy(ABC):
    """Abstract class for vapor pressure calculations. The methods
    defined here must be implemented by subclasses below."""

    def partial_pressure(
        self,
        concentration: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        temperature: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the partial pressure of the gas from its concentration, molar
        mass, and temperature.

        Args:
            concentration (float or NDArray[np.float64]): Concentration of the
                gas in kg/m^3.
            molar_mass (float or NDArray[np.float64]): Molar mass of the gas in
                kg/mol.
            temperature (float or NDArray[np.float64]): Temperature in Kelvin.

        Returns:
            partial_pressure: Partial pressure of the gas in Pascals.
        """
        return calculate_partial_pressure(
            concentration, molar_mass, temperature
        )

    def concentration(
        self,
        partial_pressure: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        temperature: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the concentration of the gas at a given pressure and
        temperature.

        Args:
            partial_pressure: Pressure in Pascals.
            molar_mass: Molar mass of the gas in kg/mol.
            temperature: Temperature in Kelvin.

        Returns:
            concentration: The concentration of the gas in kg/m^3.
        """
        return calculate_concentration(
            partial_pressure, molar_mass, temperature
        )

    def saturation_ratio(
        self,
        concentration: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        temperature: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the saturation ratio of the gas at a given pressure and
        temperature.

        Args:
            pressure: Pressure in Pascals.
            temperature: Temperature in Kelvin.

        Returns:
            saturation_ratio: The saturation ratio of the gas.
        """
        return self.partial_pressure(
            concentration, molar_mass, temperature
        ) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        temperature: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the saturation concentration of the gas at a given
        temperature.

        Args:
        ----
        - molar_mass (float or NDArray[np.float64]): Molar mass of the gas in
        kg/mol.
        - temperature (float or NDArray[np.float64]): Temperature in Kelvin.

        Returns:
        -------
        - saturation_concentration (float or NDArray[np.float64]):
        The saturation concentration of the gas in kg/m^3.
        """

        return self.concentration(
            self.pure_vapor_pressure(temperature), molar_mass, temperature
        )

    @abstractmethod
    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the pure (saturation) vapor pressure at a given
        temperature. Units are in Pascals Pa=kg/(m·s²).

        Args:
            temperature (float or NDArray[np.float64]): Temperature in Kelvin.
        """


class ConstantVaporPressureStrategy(VaporPressureStrategy):
    """Concrete implementation of the VaporPressureStrategy using a constant
    vapor pressure value."""

    def __init__(self, vapor_pressure: Union[float, NDArray[np.float64]]):
        self.vapor_pressure = vapor_pressure

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Return the constant vapor pressure value.

        Args:
        ----
        - temperature (float or NDArray[np.float64]): Not used.

        Returns:
        -------
        - vapor_pressure (float or NDArray[np.float64]): The constant vapor
        pressure value in Pascals.
        """
        # repeat the constant value for each element temperature
        return np.full_like(temperature, self.vapor_pressure)


class AntoineVaporPressureStrategy(VaporPressureStrategy):
    """Concrete implementation of the VaporPressureStrategy using the
    Antoine equation for vapor pressure calculations."""

    MMHG_TO_PA = 133.322  # Conversion factor from mmHg to Pascal

    def __init__(
        self,
        a: Union[float, NDArray[np.float64]] = 0.0,
        b: Union[float, NDArray[np.float64]] = 0.0,
        c: Union[float, NDArray[np.float64]] = 0.0,
    ):
        self.a = a
        self.b = b
        self.c = c

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate vapor pressure using the Antoine equation.

        Args:
            a, b, c: Antoine equation parameters.
            temperature: Temperature in Kelvin.

        Returns:
            Vapor pressure in Pascals.

        References:
            - Equation: log10(P) = a - b / (T - c)
            - https://en.wikipedia.org/wiki/Antoine_equation (but in Kelvin)
            - Kelvin form:
                https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1
        """
        return antoine_vapor_pressure(
            a=self.a, b=self.b, c=self.c, temperature=temperature
        )


class ClausiusClapeyronStrategy(VaporPressureStrategy):
    """Concrete implementation of the VaporPressureStrategy using the
    Clausius-Clapeyron equation for vapor pressure calculations.
    """

    def __init__(
        self,
        latent_heat: Union[float, NDArray[np.float64]],
        temperature_initial: Union[float, NDArray[np.float64]],
        pressure_initial: Union[float, NDArray[np.float64]],
    ):
        """
        Initializes the Clausius-Clapeyron strategy with the specific latent
        heat
        of vaporization and the specific gas constant of the substance.

        Args:
        ----
        - latent_heat (float or NDArray[np.float64]): specific Latent heat of
        in J/mol.
        - temperature_initial (float or NDArray[np.float64]): Initial
        temperature in Kelvin.
        - pressure_initial (float or NDArray[np.float64]): Initial vapor
        pressure in Pascals.
        """
        self.latent_heat = latent_heat
        self.temperature_initial = temperature_initial
        self.pressure_initial = pressure_initial

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate vapor pressure using Clausius-Clapeyron equation.

        Args:
            latent_heat: Latent heat of vaporization in J/mol.
            temperature_initial: Initial temperature in Kelvin.
            pressure_initial: Initial vapor pressure in Pascals.
            temperature: Final temperature in Kelvin.
            gas_constant: gas constant (default is 8.314 J/(mol·K)).

        Returns:
            Pure vapor pressure in Pascals.

        References:
            - https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
        """
        return clausius_clapeyron_vapor_pressure(
            latent_heat=self.latent_heat,
            temperature_initial=self.temperature_initial,
            pressure_initial=self.pressure_initial,
            temperature=temperature,
        )


class WaterBuckStrategy(VaporPressureStrategy):
    """Concrete implementation of the VaporPressureStrategy using the
    Buck equation for water vapor pressure calculations."""

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate vapor pressure using the Buck equation for water vapor.

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Vapor pressure in Pascals.

        References:
            - Buck, A. L., 1981: New Equations for Computing Vapor Pressure and
                Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527-1532,
                https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.
            - https://en.wikipedia.org/wiki/Arden_Buck_equation
        """
        return buck_vapor_pressure(temperature)
