"""Builders to create vapor pressure models for gas species."""

from abc import ABC, abstractmethod
from typing import Optional, Any
import logging
from particula.next.gas.vapor_pressure_strategies import (
    AntoineVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    WaterBuckStrategy,
    ConstantVaporPressureStrategy
)
from particula.util.input_handling import convert_units  # type: ignore

logger = logging.getLogger("particula")


class BuilderBase(ABC):
    """Abstract base class for builders with common methods to check keys and
    set parameters from dict."""

    def __init__(
        self,
        required_parameters: Optional[list[str]] = None
    ):
        self.required_parameters = required_parameters or []

    def check_keys(
        self,
        parameters: dict[str, Any],
    ):
        """Check if the keys you want to set are present in the
        self.required_parameters dictionary.

        Args:
        ----
        - parameters (dict): The parameters dictionary to check.
        - required_keys (list): List of required keys to be checked in the
        parameters.

        Returns:
        -------
        - None

        Raises:
        ------
        - ValueError: If you are trying to set an invalid parameter.
        """
        # check if all required keys are present
        missing = [p for p in self.required_parameters if p not in parameters]
        if missing:
            logger.error(
                "Missing required parameter(s): %s", ', '.join(missing))
            raise ValueError(
                f"Missing required parameter(s): {', '.join(missing)}")
        # check if all keys in parameters are valid, account for _units
        valid_keys = set(
            self.required_parameters
            + [f"{key}_units" for key in self.required_parameters]
        )
        key_to_set = [key for key in parameters
                      if key not in valid_keys]
        if key_to_set:
            logger.error(
                "Trying to set an invalid parameter(s) '%s'. "
                "The valid parameter(s) '%s'.",
                key_to_set, valid_keys
            )
            raise ValueError(
                f"Trying to set an invalid parameter(s) '{key_to_set}'."
                f" The valid parameter(s) '{valid_keys}'."
            )

    def set_parameters(self, parameters: dict[str, Any]):
        """Set coefficients from a dictionary including optional units.

        Args:
        ----
        - parameters (dict): The parameters dictionary to set.

        Returns:
        -------
        - self: The builder object with the set parameters.

        Raises:
        ------
        - ValueError: If any required key is missing.
        - Warning: If using default units for any parameter.
        """
        self.check_keys(parameters)  # check if all required keys are present
        for key in self.required_parameters:  # set the parameters
            unit_key = f'{key}_units'
            if unit_key in parameters:
                # build the set call set with units, from keys
                # e.g. self.set_a(params['a'], params['a_units'])
                getattr(self, f'set_{key}')(
                    parameters[key], parameters[unit_key]
                )
            else:
                logger.warning(
                    "Using default units for coefficient '%s'.", key)
                # build set call, e.g. self.set_a(params['a'])
                getattr(self, f'set_{key}')(parameters[key])
        return self

    def pre_build_check(self):
        """Check if all required attribute parameters are set before building.

        Returns:
        -------
        - None

        Raises:
        ------
        - ValueError: If any required parameter is missing.
        """
        missing = [p for p in self.required_parameters
                   if getattr(self, p) is None]
        if missing:
            logger.error(
                "Required parameter(s) not set: %s", ', '.join(missing))
            raise ValueError(
                f"Required parameter(s) not set: {', '.join(missing)}")

    @abstractmethod
    def build(self) -> Any:
        """Build and return the strategy object with the set parameters.

        Returns:
        -------
        - strategy: The built strategy object.
        """


class AntoineBuilder(BuilderBase):
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
        required_parameters = ['a', 'b', 'c']
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

    def set_b(self, b: float, b_units: str = 'K'):
        """Set the coefficient 'b' of the Antoine equation."""
        if b < 0:
            logger.error("Coefficient 'b' must be a positive value.")
            raise ValueError("Coefficient 'b' must be a positive value.")
        self.b = b * convert_units(b_units, 'K')
        return self

    def set_c(self, c: float, c_units: str = 'K'):
        """Set the coefficient 'c' of the Antoine equation."""
        if c < 0:
            logger.error("Coefficient 'c' must be a positive value.")
            raise ValueError("Coefficient 'c' must be a positive value.")
        self.c = c * convert_units(c_units, 'K')
        return self

    def build(self):
        """Build the AntoineVaporPressureStrategy object with the set
        coefficients."""
        self.pre_build_check()
        return AntoineVaporPressureStrategy(
            self.a, self.b, self.c)  # type: ignore


class ClausiusClapeyronBuilder(BuilderBase):
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
            'latent_heat',
            'temperature_initial',
            'pressure_initial']
        super().__init__(required_keys)
        self.latent_heat = None
        self.temperature_initial = None
        self.pressure_initial = None

    def set_latent_heat(
        self,
        latent_heat: float,
        latent_heat_units: str = 'J/kg'
    ):
        """Set the latent heat of vaporization: Default units J/kg."""
        if latent_heat < 0:
            raise ValueError("Latent heat must be a positive numeric value.")
        self.latent_heat = latent_heat * convert_units(
            latent_heat_units, 'J/kg')
        return self

    def set_temperature_initial(
        self,
        temperature_initial: float,
        temperature_initial_units: str = 'K'
    ):
        """Set the initial temperature. Default units: K."""
        if temperature_initial < 0:
            raise ValueError(
                "Initial temperature must be a positive numeric value.")
        self.temperature_initial = temperature_initial * convert_units(
            temperature_initial_units, 'K')
        return self

    def set_pressure_initial(
        self,
        pressure_initial: float,
        pressure_initial_units: str = 'Pa'
    ):
        """Set the initial pressure. Default units: Pa."""
        if pressure_initial < 0:
            raise ValueError(
                "Initial pressure must be a positive numeric value.")
        self.pressure_initial = pressure_initial * convert_units(
            pressure_initial_units, 'Pa')
        return self

    # def set_parameters(self, parameters: dict):  # type: ignore
    #     """Set parameters from a dictionary including optional units."""
    #     required_keys = [
    #         'latent_heat',
    #         'temperature_initial',
    #         'pressure_initial']
    #     for key in required_keys:
    #         if key not in parameters:
    #             raise ValueError(f"Missing coefficient '{key}'.")
    #         unit_key = f'{key}_units'
    #         if unit_key in parameters:
    #             # units provided
    #             getattr(self, f'set_{key}')(
    #                 parameters[key], parameters[unit_key]
    #             )
    #         else:
    #             # no units provided
    #             logger.warning(
    #                 "Using default units for coefficient '%s'.", key)
    #             getattr(self, f'set_{key}')(parameters[key])
    #     return self

    def build(self):
        """Build and return a ClausiusClapeyronStrategy object with the set
        parameters."""
        # if None in [self.latent_heat, self.temperature_initial,
        #             self.pressure_initial]:
        #     missing = [
        #         p for p in [
        #             'latent_heat',
        #             'temperature_initial',
        #             'pressure_initial'] if getattr(
        #             self,
        #             p) is None]
        #     raise ValueError(f"Missing parameters: {', '.join(missing)}")
        self.pre_build_check()
        return ClausiusClapeyronStrategy(
            self.latent_heat,  # type: ignore
            self.temperature_initial,  # type: ignore
            self.pressure_initial  # type: ignore
        )


class ConstantBuilder(BuilderBase):
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
        required_keys = ['vapor_pressure']
        super().__init__(required_keys)
        self.vapor_pressure = None

    def set_vapor_pressure(
        self,
        vapor_pressure: float,
        vapor_pressure_units: str = 'Pa'
    ):
        """Set the constant vapor pressure."""
        if vapor_pressure < 0:
            raise ValueError("Vapor pressure must be a positive value.")
        self.vapor_pressure = vapor_pressure * convert_units(
            vapor_pressure_units, 'Pa')
        return self

    # def set_parameters(self, parameters: dict):  # type: ignore
    #     """Set parameters from a dictionary including optional units."""
    #     required_keys = ['vapor_pressure']
    #     for key in required_keys:
    #         if key not in parameters:
    #             raise ValueError(f"Missing coefficient '{key}'.")
    #         unit_key = f'{key}_units'
    #         if unit_key in parameters:
    #             # units provided
    #             getattr(self, f'set_{key}')(
    #                 parameters[key], parameters[unit_key]
    #             )
    #         else:
    #             # no units provided
    #             logger.warning(
    #                 "Using default units for coefficient '%s'.", key)
    #             getattr(self, f'set_{key}')(parameters[key])
    #     return self

    def build(self):
        """Build and return a ConstantVaporPressureStrategy object with the set
        parameters."""
        # if self.vapor_pressure is None:
        #     raise ValueError("Missing parameter: vapor_pressure")
        self.pre_build_check()
        return ConstantVaporPressureStrategy(self.vapor_pressure)


class WaterBuckBuilder(BuilderBase):  # pylint: disable=too-few-public-methods
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
