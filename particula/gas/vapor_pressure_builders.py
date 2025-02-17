"""Builders to create vapor pressure models for gas species."""

import logging
from typing import Optional

from particula.abc_builder import BuilderABC
from particula.gas.vapor_pressure_strategies import (
    AntoineVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    WaterBuckStrategy,
    ConstantVaporPressureStrategy,
)
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")


class AntoineBuilder(BuilderABC):
    """Builder class for AntoineVaporPressureStrategy. It allows setting the
    coefficients 'a', 'b', and 'c' separately and then building the strategy
    object.

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
    """

    def __init__(self):
        required_parameters = ["a", "b", "c"]
        # Call the base class's __init__ method
        super().__init__(required_parameters)
        self.a = None
        self.b = None
        self.c = None

    def set_a(self, a: float, a_units: Optional[str] = None) -> "AntoineBuilder":
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
        """Build the AntoineVaporPressureStrategy object with the set
        coefficients."""
        self.pre_build_check()
        return AntoineVaporPressureStrategy(self.a, self.b, self.c)  # type: ignore


class ClausiusClapeyronBuilder(BuilderABC):
    """Builder class for ClausiusClapeyronStrategy. This class facilitates
    setting the latent heat of vaporization, initial temperature, and initial
    pressure with unit handling and then builds the strategy object.

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
            .set_latent_heat(2260, "J/kg")
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
        self, latent_heat: float, latent_heat_units: str = "J/kg"
    ) -> "ClausiusClapeyronBuilder":
        """Set the latent heat of vaporization: Default units J/kg."""
        if latent_heat_units == "J/kg":
            self.latent_heat = latent_heat
            return self
        raise ValueError("Only J/kg units are supported for latent heat.")

    @validate_inputs({"temperature_initial": "positive"})
    def set_temperature_initial(
        self, temperature_initial: float, temperature_initial_units: str = "K"
    ) -> "ClausiusClapeyronBuilder":
        """Set the initial temperature. Default units: K."""
        if temperature_initial_units == "K":
            self.temperature_initial = temperature_initial
            return self
        raise ValueError("Only K units are supported for initial temperature.")

    @validate_inputs({"pressure_initial": "positive"})
    def set_pressure_initial(
        self, pressure_initial: float, pressure_initial_units: str = "Pa"
    ) -> "ClausiusClapeyronBuilder":
        """Set the initial pressure. Default units: Pa."""
        if pressure_initial_units == "Pa":
            self.pressure_initial = pressure_initial
            return self
        raise ValueError("Only Pa units are supported for initial pressure.")

    def build(self) -> ClausiusClapeyronStrategy:
        """Build and return a ClausiusClapeyronStrategy object with the set
        parameters."""
        self.pre_build_check()
        return ClausiusClapeyronStrategy(
            self.latent_heat,  # type: ignore
            self.temperature_initial,  # type: ignore
            self.pressure_initial,  # type: ignore
        )


class ConstantBuilder(BuilderABC):
    """Builder class for ConstantVaporPressureStrategy. This class facilitates
    setting the constant vapor pressure and then building the strategy object.

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
        self, vapor_pressure: float, vapor_pressure_units: str = "Pa"
    ) -> "ConstantBuilder":
        """Set the constant vapor pressure."""
        if vapor_pressure_units == "Pa":
            self.vapor_pressure = vapor_pressure
            return self
        raise ValueError("Only Pa units are supported for vapor pressure.")

    def build(self) -> ConstantVaporPressureStrategy:
        """Build and return a ConstantVaporPressureStrategy object with the set
        parameters."""
        self.pre_build_check()
        return ConstantVaporPressureStrategy(self.vapor_pressure)


class WaterBuckBuilder(BuilderABC):  # pylint: disable=too-few-public-methods
    """Builder class for WaterBuckStrategy. This class facilitates
    the building of the WaterBuckStrategy object. Which as of now has no
    additional parameters to set. But could be extended in the future for
    ice only calculations.

    Example:
        ``` py title="WaterBuckBuilder"
        WaterBuckBuilder().build()
        ```
    """

    def __init__(self):
        super().__init__()

    def build(self) -> WaterBuckStrategy:
        """Build and return a WaterBuckStrategy object."""
        return WaterBuckStrategy()
