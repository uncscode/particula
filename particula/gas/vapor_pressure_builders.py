"""
Builders to create vapor pressure models for gas species.

This module provides builder classes for Antoine, Clausius-Clapeyron,
constant, and WaterBuck vapor pressure strategies. Each builder follows
the same workflow:

1. Configure coefficients or parameters using dedicated methods.
2. Validate required parameters.
3. Return the corresponding vapor pressure strategy object.

References:
    - "Antoine Equation,"
      [Wikipedia](https://en.wikipedia.org/wiki/Antoine_equation)
    - "Clausius–Clapeyron Relation,"
      [Wikipedia](https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation)
    - "Vapor Pressure,"
      [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure)
"""

import logging
from typing import Optional

from particula.abc_builder import BuilderABC
from particula.builder_mixin import (
    BuilderMolarMassMixin, BuilderTemperatureMixin
)
from particula.gas.vapor_pressure_strategies import (
    AntoineVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    WaterBuckStrategy,
    ConstantVaporPressureStrategy,
)
from particula.gas.properties.pressure_function import get_partial_pressure
from particula.util.validate_inputs import validate_inputs
from particula.util import get_unit_conversion

logger = logging.getLogger("particula")


class AntoineBuilder(BuilderABC):
    """
    Builder class for AntoineVaporPressureStrategy. It allows setting
    the coefficients 'a', 'b', and 'c' separately and then building the
    strategy object. Follows the general form of the Antoine equation in
    Unicode:

        log₁₀(P) = a − b / (T − c)

    Attributes:
        - a : Coefficient "a" of the Antoine equation (dimensionless).
        - b : Coefficient "b" (in Kelvin).
        - c : Coefficient "c" (in Kelvin).

    Methods:
    - set_a : Set the coefficient "a" of the Antoine equation.
    - set_b : Set the coefficient "b".
    - set_c : Set the coefficient "c".
    - build : Validate parameters and return an AntoineVaporPressureStrategy.

    Example:
        ``` py title="AntoineBuilder"
        strategy = (
            AntoineBuilder()
            .set_a(8.07131)
            .set_b(1730.63)
            .set_c(233.426)
            .build()
        )
        ```

        ``` py title="AntoineBuilder with units"
        strategy = (
            AntoineBuilder()
            .set_a(8.07131)
            .set_b(1730.63, "K")
            .set_c(233.426, "K")
            .build()
        )
        ```

    References:
        - Equation: log10(P_mmHG) = a - b / (Temperature_K - c)
          (Reference: https://en.wikipedia.org/wiki/Antoine_equation)
        - "Vapor Pressure,"
          [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure)
        - "Atmospheric Pressure Unit Conversions,"
          [Wikipedia](https://en.wikipedia.org/wiki/Pascal_(unit))
    """

    def __init__(self):
        required_parameters = ["a", "b", "c"]
        # Call the base class's __init__ method
        super().__init__(required_parameters)
        self.a = None
        self.b = None
        self.c = None

    def set_a(
        self, a: float, a_units: Optional[str] = None
    ) -> "AntoineBuilder":
        """Set the coefficient 'a' of the Antoine equation."""
        if a < 0:
            logger.error("Coefficient 'a' must be a positive value.")
            raise ValueError("Coefficient 'a' must be a positive value.")
        if a_units is not None:
            logger.warning("Ignoring units for coefficient 'a'.")
        self.a = a
        return self

    @validate_inputs({"b": "positive"})
    def set_b(self, b: float, b_units: str = "K") -> "AntoineBuilder":
        """Set the coefficient 'b' of the Antoine equation."""
        if b_units == "K":
            self.b = b
            return self
        raise ValueError("Only K units are supported for coefficient 'b'.")

    @validate_inputs({"c": "positive"})
    def set_c(self, c: float, c_units: str = "K") -> "AntoineBuilder":
        """Set the coefficient 'c' of the Antoine equation."""
        if c_units == "K":
            self.c = c
            return self
        raise ValueError("Only K units are supported for coefficient 'c'.")

    def build(self) -> AntoineVaporPressureStrategy:
        """
        Validate and return an AntoineVaporPressureStrategy using the set
        coefficients.

        Returns:
            - Configured with coefficients a, b, and c.
        """
        self.pre_build_check()
        return AntoineVaporPressureStrategy(self.a, self.b, self.c)


class ClausiusClapeyronBuilder(BuilderABC):
    """Builder class for ClausiusClapeyronStrategy. This class facilitates
    setting the latent heat of vaporization, initial temperature, and initial
    pressure with unit handling and then builds the strategy object.

    The Clausius–Clapeyron relation can be approximated as:

    - dP / dT = (L / (R × T²))


    Methods:
    - set_latent_heat : Set latent heat in J/mol (or convertible units).
    - set_temperature_initial : Set initial temperature in K
        (or convertible units).
    - set_pressure_initial : Set initial pressure in Pa
        (or convertible units).
    - build : Validate parameters and return a ClausiusClapeyronStrategy.


    Example:
        ``` py title="ClausiusClapeyronBuilder"
        strategy = (
            ClausiusClapeyronBuilder()
            .set_latent_heat(2260)
            .set_temperature_initial(373.15)
            .set_pressure_initial(101325)
            .build()
        )
        ```

        ``` py title="ClausiusClapeyronBuilder with units"
        strategy = (
            ClausiusClapeyronBuilder()
            .set_latent_heat(2260, "J/mol")
            .set_temperature_initial(373.15, "K")
            .set_pressure_initial(101325, "Pa")
            .build()
        )
        ```

    References:
        - Equation: dP/dT = L / (R * T^2)
          https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
    """

    def __init__(self):
        required_keys = [
            "latent_heat",
            "temperature_initial",
            "pressure_initial",
        ]
        super().__init__(required_keys)
        self.latent_heat = None
        self.temperature_initial = None
        self.pressure_initial = None

    @validate_inputs({"latent_heat": "positive"})
    def set_latent_heat(
        self, latent_heat: float, latent_heat_units: str
    ) -> "ClausiusClapeyronBuilder":
        """Set the latent heat of vaporization: Default units J/mol."""
        if latent_heat_units == "J/mol":
            self.latent_heat = latent_heat
            return self
        self.latent_heat = latent_heat * get_unit_conversion(
            latent_heat_units, "J/mol"
        )
        return self

    @validate_inputs({"temperature_initial": "positive"})
    def set_temperature_initial(
        self, temperature_initial: float, temperature_initial_units: str
    ) -> "ClausiusClapeyronBuilder":
        """Set the initial temperature. Default units: K."""
        if temperature_initial_units == "K":
            self.temperature_initial = temperature_initial
            return self
        self.temperature_initial = get_unit_conversion(
            temperature_initial_units, "K", temperature_initial
        )
        return self

    @validate_inputs({"pressure_initial": "positive"})
    def set_pressure_initial(
        self, pressure_initial: float, pressure_initial_units: str
    ) -> "ClausiusClapeyronBuilder":
        """Set the initial pressure. Default units: Pa."""
        if pressure_initial_units == "Pa":
            self.pressure_initial = pressure_initial
            return self
        self.pressure_initial = pressure_initial * get_unit_conversion(
            pressure_initial_units, "Pa"
        )
        return self

    def build(self) -> ClausiusClapeyronStrategy:
        """
        Validate parameters and return a ClausiusClapeyronStrategy object.

        Returns:
            - Configured with latent heat, initial Temperature, and Pressure.
        """
        self.pre_build_check()
        return ClausiusClapeyronStrategy(
            self.latent_heat,  # type: ignore
            self.temperature_initial,  # type: ignore
            self.pressure_initial,  # type: ignore
        )


class ConstantBuilder(BuilderABC):
    """Builder class for ConstantVaporPressureStrategy. This class facilitates
    setting the constant vapor pressure and then building the strategy object.

    Attributes:
        - vapor_pressure: The vapor pressure in Pa (scalar/float).

    Methods:
    - set_vapor_pressure: Set the constant vapor pressure in Pa
      (or convertible units).
    - build : Validate parameters and return a ConstantVaporPressureStrategy.

    Example:
        ``` py title="ConstantBuilder"
        strategy = (
            ConstantBuilder()
            .set_vapor_pressure(101325)
            .build()
        )
        ```

        ``` py title="ConstantBuilder with units"
        strategy = (
            ConstantBuilder()
            .set_vapor_pressure(1, "atm")
            .build()
        )
        ```

    References:
        - Equation: P = vapor_pressure
          https://en.wikipedia.org/wiki/Vapor_pressure
    """

    def __init__(self):
        required_keys = ["vapor_pressure"]
        super().__init__(required_keys)
        self.vapor_pressure = None

    @validate_inputs({"vapor_pressure": "positive"})
    def set_vapor_pressure(
        self, vapor_pressure: float, vapor_pressure_units: str
    ) -> "ConstantBuilder":
        """Set the constant vapor pressure."""
        if vapor_pressure_units == "Pa":
            self.vapor_pressure = vapor_pressure
            return self
        self.vapor_pressure = vapor_pressure * get_unit_conversion(
            vapor_pressure_units, "Pa"
        )
        return self

    def build(self) -> ConstantVaporPressureStrategy:
        """
        Validate parameters and return a ConstantVaporPressureStrategy object.

        Returns:
            - Configured with vapor_pressure in Pa.
        """
        self.pre_build_check()
        return ConstantVaporPressureStrategy(self.vapor_pressure)


class SaturationConcentrationVaporPressureBuilder(
    BuilderABC,
    BuilderMolarMassMixin,
    BuilderTemperatureMixin,
    ):
    """Builder class for SaturationConcentrationVaporPressureStrategy.

    This class facilitates the building of the
    SaturationConcentrationVaporPressureStrategy object. It allows
    for setting the vapor pressure using the saturation concentration,
    molar mass, and temperature. The saturation concentration is commonly
    called C^sat or C* when in a mixture.

    Example:
        ```py title="SaturationConcentrationVaporPressureBuilder"
        import particula as par
        strategy = par.gas.SaturationConcentrationVaporPressureBuilder().build()
        ```
    """

    def __init__(self):
        required_parameters = [
            "saturation_concentration",
            "molar_mass",
            "temperature"
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)
        BuilderTemperatureMixin.__init__(self)
        self.saturation_concentration = None
        self.molar_mass = None
        self.temperature = None

    def set_saturation_concentration(
        self, saturation_concentration: float, units: str = "kg/m^3"
    )
        """Set the saturation concentration in kg/m^3.
        
        
        
        """
        


    def build(self) -> ConstantVaporPressureStrategy:
        """

        """
        self.pre_build_check()

        return ConstantVaporPressureStrategy(
            vapor_pressure=get_partial_pressure(
                self.saturation_concentration,
                self.molar_mass,
                self.temperature,
            )
        )


class WaterBuckBuilder(BuilderABC):  # pylint: disable=too-few-public-methods
    """Builder class for WaterBuckStrategy.

    This class facilitates the building of the WaterBuckStrategy object.
    Which as of now has no additional parameters to set but could be
    extended in the future (e.g., ice-only calculations).

    Example:
        ```py title="WaterBuckBuilder"
        import particula as par
        strategy = par.gas.WaterBuckBuilder().build()
        ```
    """

    def __init__(self):
        super().__init__()

    def build(self) -> WaterBuckStrategy:
        """
        Build and return a WaterBuckStrategy object.

        Returns:
            - Configured for water-specific Buck vapor pressure.
        """
        return WaterBuckStrategy()
