"""Abstract Base Class for Builder classes.

This module also defines mixin classes for the Builder classes to set
some optional method to be used in the Builder classes.
https://en.wikipedia.org/wiki/Mixin
"""
# pylint: disable=too-few-public-methods

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import logging
from numpy.typing import NDArray
import numpy as np

from particula.util.input_handling import convert_units  # type: ignore

logger = logging.getLogger("particula")


class BuilderABC(ABC):
    """Abstract base class for builders with common methods to check keys and
    set parameters from a dictionary.

    Attributes:
    ----------
    - required_parameters (list): List of required parameters for the builder.

    Methods:
    -------
    - check_keys(parameters): Check if the keys you want to set are
    present in the parameters dictionary.
    - set_parameters(parameters): Set parameters from a dictionary including
        optional suffix for units as '_units'.
    - pre_build_check(): Check if all required attribute parameters are set
        before building.

    Abstract Methods:
    -----------------
    - build(): Build and return the strategy object with the set parameters.

    Raises:
    ------
    - ValueError: If any required key is missing during check_keys or
        pre_build_check, or if trying to set an invalid parameter.
    - Warning: If using default units for any parameter.
    """

    def __init__(self, required_parameters: Optional[list[str]] = None):
        self.required_parameters = required_parameters or []

    def check_keys(self, parameters: dict[str, Any]):
        """Check if the keys you want to set are present in the
        parameters dictionary and if all keys are valid.

        Args:
        ----
        - parameters (dict): The parameters dictionary to check.

        Returns:
        -------
        - None

        Raises:
        ------
        - ValueError: If any required key is missing or if trying to set an
        invalid parameter.
        """

        # Check if all required keys are present
        if missing := [p for p in self.required_parameters
                       if p not in parameters]:
            error_message = (
                f"Missing required parameter(s): {', '.join(missing)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Generate a set of all valid keys
        valid_keys = set(
            self.required_parameters +
            [f"{key}_units" for key in self.required_parameters]
        )
        # Check for any invalid keys and handle them within the if condition
        if invalid_keys := [key for key in parameters
                            if key not in valid_keys]:
            error_message = (
                f"Trying to set an invalid parameter(s) '{invalid_keys}'. "
                f"The valid parameter(s) '{valid_keys}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)

    def set_parameters(self, parameters: dict[str, Any]):
        """Set parameters from a dictionary including optional suffix for
        units as '_units'.

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
        self.check_keys(parameters)
        for key in self.required_parameters:
            unit_key = f'{key}_units'
            if unit_key in parameters:
                # Call set method with units
                getattr(
                    self,
                    f'set_{key}')(
                    parameters[key],
                    parameters[unit_key])
            else:
                logger.warning("Using default units for parameter: '%s'.", key)
                # Call set method without units
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
        if missing := [p for p in self.required_parameters
                       if getattr(self, p) is None]:
            error_message = (
                f"Required parameter(s) not set: {', '.join(missing)}")
            logger.error(error_message)
            raise ValueError(error_message)

    @abstractmethod
    def build(self) -> Any:
        """Build and return the strategy object with the set parameters.

        Returns:
        -------
        - strategy: The built strategy object.
        """


class BuilderDensityMixin():
    """Mixin class for Builder classes to set density and density_units.

    Methods:
    -------
    - set_density(density: float, density_units: str): Set the density
        attribute and units.
    """

    def __init__(self):
        self.density = None

    def set_density(
        self,
        density: Union[float, NDArray[np.float_]],
        density_units: Optional[str] = 'kg/m^3'
    ):
        """Set the density of the particle in kg/m^3.

        Args:
        -----
        - density (float or NDArray[float]): Density of the particle [kg/m^3].
        - density_units (str, optional): Units of the density. Default is
            'kg/m^3'.
        """
        if np.any(density < 0):
            error_message = "Density must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.density = density * convert_units(density_units, 'kg/m^3')


class BuilderSurfaceTensionMixin():
    """Mixin class for Builder classes to set surface_tension and
    surface_tension_units.

    Methods:
    -------
    - set_surface_tension(surface_tension: float, surface_tension_units: str):
        Set the surface_tension attribute and units.
    """

    def __init__(self):
        self.surface_tension = None

    def set_surface_tension(
        self,
        surface_tension: Union[float, NDArray[np.float_]],
        surface_tension_units: Optional[str] = 'N/m'
    ):
        """Set the surface tension of the particle in N/m.

        Args:
        -----
        - surface_tension (float or NDArray[float]): Surface tension of the
            particle [N/m].
        - surface_tension_units (str, optional): Units of the surface tension.
            Default is 'N/m'.
        """
        if np.any(surface_tension < 0):
            error_message = "Surface tension must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.surface_tension = surface_tension * convert_units(
            surface_tension_units, 'N/m')


class BuilderMolarMassMixin():
    """Mixin class for Builder classes to set molar_mass and molar_mass_units.

    Methods:
    -------
    - set_molar_mass(molar_mass: float, molar_mass_units: str): Set the
        molar_mass attribute and units.
    """

    def __init__(self):
        self.molar_mass = None

    def set_molar_mass(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        molar_mass_units: Optional[str] = 'kg/mol'
    ):
        """Set the molar mass of the particle in kg/mol.

        Args:
        -----
        - molar_mass (float or NDArray[float]): Molar mass of the particle
            [kg/mol].
        - molar_mass_units (str, optional): Units of the molar mass. Default is
            'kg/mol'.
        """
        if np.any(molar_mass < 0):
            error_message = "Molar mass must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.molar_mass = molar_mass \
            * convert_units(molar_mass_units, 'kg/mol')


class BuilderConcentrationMixin():
    """Mixin class for Builder classes to set concentration and
    concentration_units.

    Methods:
    -------
    - set_concentration(concentration: float, concentration_units: str):
    Set the concentration attribute and units.
    """

    def __init__(self):
        self.concentration = None

    def set_concentration(
        self,
        concentration: Union[float, NDArray[np.float_]],
        concentration_units: Optional[str] = 'kg/m^3'
    ):
        """Set the concentration of the particle in kg/m^3.

        Args:
        -----
        - concentration (float or NDArray[float]): Concentration of the
        species or particle in the mixture.
        - concentration_units (str, optional): Units of the concentration.
            Default is 'kg/m^3'.
        """
        if np.any(concentration < 0):
            error_message = "Concentration must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.concentration = concentration \
            * convert_units(concentration_units, 'kg/m^3')
