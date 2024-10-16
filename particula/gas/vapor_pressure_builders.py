"""Builders to create vapor pressure models for gas species."""

from typing import Optional
import logging
from particula.abc_builder import BuilderABC
from particula.gas.vapor_pressure_strategies import (
    AntoineVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    WaterBuckStrategy,
    ConstantVaporPressureStrategy,
)
from particula.util.input_handling import convert_units  # type: ignore

logger = logging.getLogger("particula")


class AntoineBuilder(BuilderABC):
    """Builder class for AntoineVaporPressureStrategy. It allows setting the
    coefficients 'a', 'b', and 'c' separately and then building the strategy
    object.

    - Equation: log10(P_mmHG) = a - b / (Temperature_K - c)
    - Units: 'a_units' = None, 'b_units' = 'K', 'c_units' = 'K'

    Methods:
    --------
    - set_a(a, a_units): Set the coefficient 'a' of the Antoine equation.
    - set_b(b, b_units): Set the coefficient 'b' of the Antoine equation.
    - set_c(c, c_units): Set the coefficient 'c' of the Antoine equation.
    - set_parameters(params): Set coefficients from a dictionary including
        optional units.
    - build(): Build the AntoineVaporPressureStrategy object with the set
        coefficients.
    """

    def __init__(self):
        required_parameters = ["a", "b", "c"]
        # Call the base class's __init__ method
        super().__init__(required_parameters)
        self.a = None
        self.b = None
        self.c = None

    def set_a(self, a: float, a_units: Optional[str] = None):
        """Set the coefficient 'a' of the Antoine equation."""
        if a < 0:
            logger.error("Coefficient 'a' must be a positive value.")
            raise ValueError("Coefficient 'a' must be a positive value.")
        if a_units is not None:
            logger.warning("Ignoring units for coefficient 'a'.")
        self.a = a
        return self

    def set_b(self, b: float, b_units: str = "K"):
        """Set the coefficient 'b' of the Antoine equation."""
        if b < 0:
            logger.error("Coefficient 'b' must be a positive value.")
            raise ValueError("Coefficient 'b' must be a positive value.")
        self.b = convert_units(b_units, "K", b)
        return self

    def set_c(self, c: float, c_units: str = "K"):
        """Set the coefficient 'c' of the Antoine equation."""
        if c < 0:
            logger.error("Coefficient 'c' must be a positive value.")
            raise ValueError("Coefficient 'c' must be a positive value.")
        self.c = convert_units(c_units, "K", c)
        return self

    def build(self):
        """Build the AntoineVaporPressureStrategy object with the set
        coefficients."""
        self.pre_build_check()
        return AntoineVaporPressureStrategy(
            self.a, self.b, self.c
        )  # type: ignore


class ClausiusClapeyronBuilder(BuilderABC):
    """Builder class for ClausiusClapeyronStrategy. This class facilitates
    setting the latent heat of vaporization, initial temperature, and initial
    pressure with unit handling and then builds the strategy object.

    - Equation: dP/dT = L / (R * T^2)
    - Units: 'latent_heat_units' = 'J/kg', 'temperature_initial_units' = 'K',
        'pressure_initial_units' = 'Pa'

    Methods:
    --------
    - set_latent_heat(latent_heat, latent_heat_units): Set the latent heat of
        vaporization.
    - set_temperature_initial(temperature_initial, temperature_initial_units):
        Set the initial temperature.
    - set_pressure_initial(pressure_initial, pressure_initial_units): Set the
        initial pressure.
    - set_parameters(parameters): Set parameters from a dictionary including
        optional units.
    - build(): Build the ClausiusClapeyronStrategy object with the set
        parameters.
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

    def set_latent_heat(
        self, latent_heat: float, latent_heat_units: str = "J/kg"
    ):
        """Set the latent heat of vaporization: Default units J/kg."""
        if latent_heat < 0:
            raise ValueError("Latent heat must be a positive numeric value.")
        self.latent_heat = latent_heat * convert_units(
            latent_heat_units, "J/kg"
        )
        return self

    def set_temperature_initial(
        self, temperature_initial: float, temperature_initial_units: str = "K"
    ):
        """Set the initial temperature. Default units: K."""
        if temperature_initial < 0:
            raise ValueError(
                "Initial temperature must be a positive numeric value."
            )
        self.temperature_initial = convert_units(
            temperature_initial_units, "K", temperature_initial
        )
        return self

    def set_pressure_initial(
        self, pressure_initial: float, pressure_initial_units: str = "Pa"
    ):
        """Set the initial pressure. Default units: Pa."""
        if pressure_initial < 0:
            raise ValueError(
                "Initial pressure must be a positive numeric value."
            )
        self.pressure_initial = pressure_initial * convert_units(
            pressure_initial_units, "Pa"
        )
        return self

    def build(self):
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

    - Equation: P = vapor_pressure
    - Units: 'vapor_pressure_units' = 'Pa'

    Methods:
    --------
    - set_vapor_pressure(constant, constant_units): Set the constant vapor
    pressure.
    - set_parameters(parameters): Set parameters from a dictionary including
        optional units.
    - build(): Build the ConstantVaporPressureStrategy object with the set
        parameters.
    """

    def __init__(self):
        required_keys = ["vapor_pressure"]
        super().__init__(required_keys)
        self.vapor_pressure = None

    def set_vapor_pressure(
        self, vapor_pressure: float, vapor_pressure_units: str = "Pa"
    ):
        """Set the constant vapor pressure."""
        if vapor_pressure < 0:
            raise ValueError("Vapor pressure must be a positive value.")
        self.vapor_pressure = vapor_pressure * convert_units(
            vapor_pressure_units, "Pa"
        )
        return self

    def build(self):
        """Build and return a ConstantVaporPressureStrategy object with the set
        parameters."""
        self.pre_build_check()
        return ConstantVaporPressureStrategy(self.vapor_pressure)


class WaterBuckBuilder(BuilderABC):  # pylint: disable=too-few-public-methods
    """Builder class for WaterBuckStrategy. This class facilitates
    the building of the WaterBuckStrategy object. Which as of now has no
    additional parameters to set. But could be extended in the future for
    ice only calculations. We keep the builder for consistency.

    Methods:
    --------
    - build(): Build the WaterBuckStrategy object.
    """

    def __init__(self):
        super().__init__()

    def build(self):
        """Build and return a WaterBuckStrategy object."""
        return WaterBuckStrategy()
