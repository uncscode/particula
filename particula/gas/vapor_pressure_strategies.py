"""Vapor Pressure Module.

This module calculates the vapor pressure of substances based on different
strategies. These strategies are interchangeable and can be used to calculate
the vapor pressure of a substance at a given temperature.

All units are in base SI units (kg, m, s).
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.gas.properties.concentration_function import (
    get_concentration_from_pressure,
)
from particula.gas.properties.pressure_function import (
    get_partial_pressure,
)
from particula.gas.properties.vapor_pressure_module import (
    get_antoine_vapor_pressure,
    get_buck_vapor_pressure,
    get_clausius_clapeyron_vapor_pressure,
)


class VaporPressureStrategy(ABC):
    """Base class for vapor pressure calculations.

    This abstract class defines standard methods for partial pressure,
    concentration, saturation ratio, saturation concentration, and pure vapor
    pressure. Subclasses must implement the pure_vapor_pressure method
    with specific formulae or empirical correlations for vapor pressure.

    Attributes:
        - None

    Methods:
    - partial_pressure: Compute partial pressure from concentration.
    - concentration: Compute concentration from partial pressure.
    - saturation_ratio: Compute ratio of partial pressure to saturation
      pressure.
    - saturation_concentration: Compute concentration at saturation pressure.
    - pure_vapor_pressure: Abstract method to compute pure (saturation) vapor
      pressure.

    Examples:
        ```py title="General Usage"
        # Cannot instantiate directly:
        #    strategy = VaporPressureStrategy()  # Error (abstract)
        # Use a derived strategy class instead.
        ```

    References:
        - "Vapor Pressure,"
        [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).
    """

    def partial_pressure(
        self,
        concentration: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        temperature: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate partial pressure from concentration, molar mass, T.

        Arguments:
            - concentration : Concentration of the gas in kg/m^3.
            - molar_mass : Molar mass of the gas in kg/mol.
            - temperature : Temperature in Kelvin.

        Returns:
            Partial pressure of the gas in Pascals.

        Examples:
            ``` py title="Partial Pressure Calculation"
            partial_pressure = strategy.partial_pressure(
                concentration=5.0,
                molar_mass=18.01528,
                temperature=298.15
            )
            ```
        """
        return get_partial_pressure(concentration, molar_mass, temperature)

    def concentration(
        self,
        partial_pressure: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        temperature: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the concentration of the gas at a given pressure and
        temperature.

        Arguments:
            - partial_pressure: Pressure in Pascals.
            - molar_mass: Molar mass of the gas in kg/mol.
            - temperature: Temperature in Kelvin.

        Returns:
            The concentration of the gas in kg/m^3.

        Examples:
            ``` py title="Concentration Calculation"
            concentration = strategy.concentration(
                partial_pressure=101325,
                molar_mass=18.01528,
                temperature=298.15
            )
            ```
        """
        return get_concentration_from_pressure(
            partial_pressure, molar_mass, temperature
        )

    def saturation_ratio(
        self,
        concentration: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        temperature: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the saturation ratio of the gas from its concentration and
        molar mass at a given temperature.

        Arguments:
            - concentration : Concentration of the gas in kg/m^3.
            - molar_mass : Molar mass of the gas in kg/mol.
            - temperature : Temperature in Kelvin.

        Returns:
            The saturation ratio of the gas.

        Examples:
            ``` py title="Saturation Ratio Calculation"
            saturation_ratio = strategy.saturation_ratio(
                concentration=5.0,
                molar_mass=18.01528,
                temperature=298.15
            )
            ```
        """
        return self.partial_pressure(
            concentration, molar_mass, temperature
        ) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        temperature: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the saturation concentration of the gas at a given
        temperature.

        Arguments:
            - molar_mass : Molar mass of the gas in kg/mol.
            - temperature : Temperature in Kelvin.

        Returns:
            The saturation concentration of the gas in kg/m^3.

        Examples:
            ``` py title="Saturation Concentration Calculation"
            saturation_concentration = strategy.saturation_concentration(
                molar_mass=18.01528,
                temperature=298.15
            )
            ```
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

        Arguments:
            - temperature : Temperature in Kelvin.
        """


class ConstantVaporPressureStrategy(VaporPressureStrategy):
    """Vapor pressure strategy with a constant value.

    This class returns a single, unchanging vapor pressure value regardless of
    the temperature. It is useful for scenarios that require a simplified
    model.

    Attributes:
        - vapor_pressure : The constant vapor pressure in Pascals.

    Methods:
    - partial_pressure: Compute partial pressure from concentration.
    - concentration: Compute concentration from partial pressure.
    - saturation_ratio: Compute ratio of partial pressure to saturation
      pressure.
    - saturation_concentration: Compute concentration at saturation pressure.
    - pure_vapor_pressure: Returns the constant vapor pressure.

    Examples:
        ```py title="Constant Vapor Pressure Example"
        import particula as par
        strategy = par.gas.ConstantVaporPressureStrategy(101325.0)
        vp = strategy.pure_vapor_pressure(temperature=300)
        # vp is 101325.0
        ```

    References:
        - None
    """

    def __init__(self, vapor_pressure: Union[float, NDArray[np.float64]]):
        """Initialize the Constant Vapor Pressure strategy.

        Arguments:
            vapor_pressure: The constant vapor pressure value in Pascals.
        """
        self.vapor_pressure = vapor_pressure

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Return the constant vapor pressure value.

        Arguments:
            - temperature : Not used.

        Returns:
            The constant vapor pressure value in Pascals.

        Examples:
            ``` py title="Constant Vapor Pressure Calculation"
            vapor_pressure = strategy.pure_vapor_pressure(
                temperature=300
            )
            ```
        """
        # repeat the constant value for each element temperature
        return np.full_like(temperature, self.vapor_pressure)


class AntoineVaporPressureStrategy(VaporPressureStrategy):
    """Vapor pressure strategy using the Antoine equation.

    This class calculates vapor pressure by applying the Antoine equation,
    which relates temperature in Kelvin to the logarithm of vapor pressure.

    Attributes:
        - a : Coefficient "a" in the Antoine equation.
        - b : Coefficient "b" in the Antoine equation.
        - c : Coefficient "c" in the Antoine equation.

    Methods:
    - partial_pressure: Compute partial pressure from concentration.
    - concentration: Compute concentration from partial pressure.
    - saturation_ratio: Compute ratio of partial pressure to saturation
      pressure.
    - saturation_concentration: Compute concentration at saturation pressure.
    - pure_vapor_pressure: Computes vapor pressure from the Antoine equation.

    Examples:
        ```py title="Antoine Vapor Pressure Example"
        import particula as par
        strategy = par.gas.AntoineVaporPressureStrategy(
            a=8.07131, b=1730.63, c=233.426
        )
        vp = strategy.pure_vapor_pressure(temperature=373.15)
        # Returns the vapor pressure in Pascals
        ```

    References:
        - "Antoine Equation,"
          [Wikipedia](https://en.wikipedia.org/wiki/Antoine_equation).
        - Kelvin-based adaptation:
          https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1
    """

    MMHG_TO_PA = 133.322  # Conversion factor from mmHg to Pascal

    def __init__(
        self,
        a: Union[float, NDArray[np.float64]] = 0.0,
        b: Union[float, NDArray[np.float64]] = 0.0,
        c: Union[float, NDArray[np.float64]] = 0.0,
    ):
        """Initialize the Antoine Vapor Pressure strategy.

        Arguments:
            a: Coefficient 'a' in the Antoine equation.
            b: Coefficient 'b' in the Antoine equation (K).
            c: Coefficient 'c' in the Antoine equation (K).
        """
        self.a = a
        self.b = b
        self.c = c

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate vapor pressure using the Antoine equation.

        Arguments:
            - temperature : Temperature in Kelvin.

        Returns:
            Vapor pressure in Pascals.

        Examples:
            ``` py title="Antoine Vapor Pressure Calculation"
            vapor_pressure = strategy.pure_vapor_pressure(
                temperature=300
            )
            ```

        References:
            - Equation: log10(P) = a - b / (T - c)
            - https://en.wikipedia.org/wiki/Antoine_equation (but in Kelvin)
            - Kelvin form:
                https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1
        """
        return get_antoine_vapor_pressure(
            a=self.a, b=self.b, c=self.c, temperature=temperature
        )


class ClausiusClapeyronStrategy(VaporPressureStrategy):
    """Vapor pressure strategy using the Clausius-Clapeyron equation.

    This class calculates vapor pressure by applying the Clausius-Clapeyron
    relation, which relates how the vapor pressure of a substance changes
    with temperature, given latent heat data and a reference point.

    Attributes:
        - latent_heat : Latent heat of vaporization (J/mol).
        - temperature_initial : Reference temperature (K).
        - pressure_initial : Reference pressure (Pa).

    Methods:
    - partial_pressure: Compute partial pressure from concentration.
    - concentration: Compute concentration from partial pressure.
    - saturation_ratio: Compute ratio of partial pressure to saturation
      pressure.
    - saturation_concentration: Compute concentration at saturation pressure.
    - pure_vapor_pressure: Computes vapor pressure via Clausius-Clapeyron
      relation.

    Examples:
        ```py title="Clausius-Clapeyron Example"
        strategy = ClausiusClapeyronStrategy(
            latent_heat=4.07e4,
            temperature_initial=298.15,
            pressure_initial=3167.0
        )
        vp = strategy.pure_vapor_pressure(temperature=310)
        ```

    References:
        - "Clausius–Clapeyron relation,"
          [Wikipedia](https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation).
    """

    def __init__(
        self,
        latent_heat: Union[float, NDArray[np.float64]],
        temperature_initial: Union[float, NDArray[np.float64]],
        pressure_initial: Union[float, NDArray[np.float64]],
    ):
        """Initializes the Clausius-Clapeyron strategy with the specific latent
        heat of vaporization and the specific gas constant of the substance.

        Arguments:
            - latent_heat : Latent heat of vaporization in J/mol.
            - temperature_initial : Initial temperature in Kelvin.
            - pressure_initial : Initial vapor pressure in Pascals.
        """
        self.latent_heat = latent_heat
        self.temperature_initial = temperature_initial
        self.pressure_initial = pressure_initial

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate vapor pressure using Clausius-Clapeyron equation.

        Arguments:
            - temperature : Final temperature in Kelvin.

        Returns:
            Pure vapor pressure in Pascals.

        Examples:
            ``` py title="Clausius-Clapeyron Vapor Pressure Calculation"
            vapor_pressure = strategy.pure_vapor_pressure(
                temperature=300
            )
            ```

        References:
            - https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
        """
        return get_clausius_clapeyron_vapor_pressure(
            latent_heat=self.latent_heat,
            temperature_initial=self.temperature_initial,
            pressure_initial=self.pressure_initial,
            temperature=temperature,
        )


class WaterBuckStrategy(VaporPressureStrategy):
    """Vapor pressure strategy using the Buck equation for water.

    This class computes water vapor pressure using the Buck equation, an
    empirically derived correlation often applied in meteorology to determine
    the saturation vapor pressure of water.

    Methods:
    - partial_pressure: Compute partial pressure from concentration.
    - concentration: Compute concentration from partial pressure.
    - saturation_ratio: Compute ratio of partial pressure to saturation
      pressure.
    - saturation_concentration: Compute concentration at saturation pressure.
    - pure_vapor_pressure: Computes water vapor pressure from the Buck
      equation.

    Examples:
        ```py title="Water Buck Vapor Pressure Example"
        strategy = WaterBuckStrategy()
        vp = strategy.pure_vapor_pressure(temperature=298.15)
        # Returns water vapor pressure in Pascals
        ```

    References:
        - A. L. Buck, "New Equations for Computing Vapor Pressure...",
          J. Appl. Meteor. Climatol. 20(12), 1527–1532 (1981).
        - https://en.wikipedia.org/wiki/Arden_Buck_equation
    """

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate vapor pressure using the Buck equation for water vapor.

        Arguments:
            - temperature: Temperature in Kelvin.

        Returns:
            Vapor pressure in Pascals.

        Examples:
            ``` py title="Water Buck Vapor Pressure Calculation"
            vapor_pressure = strategy.pure_vapor_pressure(
                temperature=300
            )
            ```

        References:
            - Buck, A. L., 1981: New Equations for Computing Vapor Pressure and
              Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527-1532,
              https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.
            - https://en.wikipedia.org/wiki/Arden_Buck_equation
        """
        return get_buck_vapor_pressure(temperature)


class TableVaporPressureStrategy(VaporPressureStrategy):
    """Vapor pressure strategy using interpolation of a lookup table.

    This strategy accepts a set of temperatures and corresponding vapor
    pressures and linearly interpolates between the points to determine
    the vapor pressure at arbitrary temperatures.

    Attributes:
        - vapor_pressures: NDArray[np.float64]
          Vapor pressure values in Pascals.
        - temperatures: NDArray[np.float64]
          Temperatures in Kelvin corresponding to the vapor pressures.

    Methods:
    - partial_pressure: Compute partial pressure from concentration.
    - concentration: Compute concentration from partial pressure.
    - saturation_ratio: Compute ratio of partial pressure to saturation
      pressure.
    - saturation_concentration: Compute concentration at saturation pressure.
    - pure_vapor_pressure: Interpolate the vapor pressure for the
      given temperature.

    Examples:
        ```py title="Table Vapor Pressure Example"
        import numpy as np
        from particula.gas import TableVaporPressureStrategy

        table_vp = [100.0, 200.0, 300.0, 400.0]  # Pa
        table_t = [270.0, 280.0, 290.0, 300.0]   # K
        strategy = TableVaporPressureStrategy(
            vapor_pressures=table_vp,
            temperatures=table_t
        )
        vp = strategy.pure_vapor_pressure(295.0)
        ```
    """

    def __init__(
        self,
        vapor_pressures: NDArray[np.float64],
        temperatures: NDArray[np.float64],
    ) -> None:
        """Initialize the Table Vapor Pressure strategy.

        Arguments:
            vapor_pressures: Array of vapor pressure values in Pascals.
            temperatures: Array of corresponding temperatures in Kelvin.

        Raises:
            ValueError: If temperature and pressure tables differ in length.
        """
        table_pressures = np.asarray(vapor_pressures, dtype=float)
        table_temps = np.asarray(temperatures, dtype=float)
        if table_pressures.size != table_temps.size:
            raise ValueError(
                "Temperature and pressure tables must be the same length"
            )
        idx = np.argsort(table_temps)
        self.vapor_pressures = table_pressures[idx]
        self.temperatures = table_temps[idx]

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Interpolate the vapor pressure for the given temperature."""
        return np.interp(
            np.asarray(temperature, dtype=float),
            self.temperatures,
            self.vapor_pressures,
            left=self.vapor_pressures[0],
            right=self.vapor_pressures[-1],
        )


class ArblasterLiquidVaporPressureStrategy(VaporPressureStrategy):
    r"""Vapor pressure strategy using a 5-term logarithmic polynomial.

    This strategy applies the Honig–Kramer/Arblaster correlation for
    liquid metals. The relationship is expressed as:

    - ln(P[bar]) = A + B ln(T) + C ∕ T + D T + E T²
        - P is the vapor pressure in bar,
        - T is the temperature in kelvin,
        - A, B, C, D and E are empirical coefficients.

    Methods:
    - pure_vapor_pressure: Compute the vapor pressure using the coefficients.

    Examples:
        ```py title="Instantiate the Arblaster strategy"
        coeffs = (7.5, 1.0, -2000.0, 0.002, -1e-6)
        strategy = ArblasterLiquidVaporPressureStrategy(coeffs)
        vp = strategy.pure_vapor_pressure(3000.0)
        ```
    """

    def __init__(self, coefficients: tuple[float, float, float, float, float]):
        """Initialize with the 5-term polynomial coefficients.

        Arguments:
            - coefficients : Tuple ``(A, B, C, D, E)`` for the correlation.
        """
        self.coefficients = coefficients

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate vapor pressure from the 5-term polynomial.

        Arguments:
            - temperature : Temperature in Kelvin.

        Returns:
            Vapor pressure in Pascals.

        Examples:
            ``` py title="Vapor pressure from polynomial"
            vp = strategy.pure_vapor_pressure(3000.0)
            ```
        """
        a, b, c, d, e = self.coefficients
        ln_p_bar = (
            a
            + b * np.log(temperature)
            + c / temperature
            + d * temperature
            + e * temperature**2
        )
        return np.exp(ln_p_bar) * 1e5


class LiquidClausiusHybridStrategy(VaporPressureStrategy):
    """Hybrid strategy blending liquid and Clausius–Clapeyron methods.

    This strategy uses the 5-term liquid polynomial at low temperature and
    gradually transitions to the Clausius–Clapeyron relation above the
    boiling point. A logistic weight provides a smooth changeover.

    Methods:
    - pure_vapor_pressure: Compute the blended vapor pressure.
    """

    def __init__(
        self,
        coefficients: tuple[float, float, float, float, float],
        latent_heat: float,
        temperature_initial: float,
        pressure_initial: float,
        boiling_point: float,
        transition_width: float = 10.0,
    ) -> None:
        """Initialize the hybrid vapor pressure strategy.

        Arguments:
            - coefficients : Polynomial coefficients ``(A, B, C, D, E)``.
            - latent_heat : Latent heat of vaporization in J/mol.
            - temperature_initial : Reference temperature in Kelvin.
            - pressure_initial : Reference pressure in Pascals.
            - boiling_point : Temperature where weighting is 0.5.
            - transition_width : Width of the logistic transition in Kelvin.
        """
        self.liquid_strategy = ArblasterLiquidVaporPressureStrategy(
            coefficients
        )
        self.clausius_strategy = ClausiusClapeyronStrategy(
            latent_heat, temperature_initial, pressure_initial
        )
        self.boiling_point = boiling_point
        self.transition_width = transition_width

    def pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Smoothly combine liquid and Clausius–Clapeyron approaches.

        Arguments:
            - temperature : Temperature in Kelvin.

        Returns:
            Vapor pressure in Pascals.

        Examples:
            ``` py title="Hybrid vapor pressure"
            vp = strategy.pure_vapor_pressure(boiling_point + 50.0)
            ```
        """
        temp = np.array(temperature, dtype=float)
        p_liq = self.liquid_strategy.pure_vapor_pressure(temp)
        p_claus = self.clausius_strategy.pure_vapor_pressure(temp)
        weight = 1.0 / (
            1.0 + np.exp(-(temp - self.boiling_point) / self.transition_width)
        )
        return (1.0 - weight) * p_liq + weight * p_claus
