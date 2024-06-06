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
from particula.next.particles.surface_strategies import SurfaceStrategy
from particula.next.particles.activity_strategies import ActivityStrategy
from particula.next.particles.distribution_strategies import (
    DistributionStrategy,
)

logger = logging.getLogger("particula")


class BuilderABC(ABC):
    """Abstract base class for builders with common methods to check keys and
    set parameters from a dictionary.

    Attributes:
        required_parameters: List of required parameters for the builder.

    Methods:
        check_keys (parameters): Check if the keys you want to set are
        present in the parameters dictionary.
        set_parameters (parameters): Set parameters from a dictionary including
        optional suffix for units as '_units'.
        pre_build_check(): Check if all required attribute parameters are set
        before building.
        build (abstract): Build and return the strategy object.

    Raises:
        ValueError: If any required key is missing during check_keys or
        pre_build_check, or if trying to set an invalid parameter.
        Warning: If using default units for any parameter.

    References:
        This module also defines mixin classes for the Builder classes to set
        some optional method to be used in the Builder classes.
        [Mixin Wikipedia](https://en.wikipedia.org/wiki/Mixin)
    """

    def __init__(self, required_parameters: Optional[list[str]] = None):
        self.required_parameters = required_parameters or []

    def check_keys(self, parameters: dict[str, Any]):
        """Check if the keys are present and valid.

        Args:
            parameters: The parameters dictionary to check.

        Raises:
            ValueError: If any required key is missing or if trying to set an
            invalid parameter.
        """

        # Check if all required keys are present
        if missing := [
            p for p in self.required_parameters if p not in parameters
        ]:
            error_message = (
                f"Missing required parameter(s): {', '.join(missing)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Generate a set of all valid keys
        valid_keys = set(
            self.required_parameters
            + [f"{key}_units" for key in self.required_parameters]
        )
        # Check for any invalid keys and handle them within the if condition
        if invalid_keys := [
            key for key in parameters if key not in valid_keys
        ]:
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
            parameters: The parameters dictionary to set.

        Returns:
            self: The builder object with the set parameters.

        Raises:
            ValueError: If any required key is missing.
            Warning: If using default units for any parameter.
        """
        self.check_keys(parameters)
        for key in self.required_parameters:
            unit_key = f"{key}_units"
            if unit_key in parameters:
                # Call set method with units
                getattr(self, f"set_{key}")(
                    parameters[key], parameters[unit_key]
                )
            else:
                logger.warning("Using default units for parameter: '%s'.", key)
                # Call set method without units
                getattr(self, f"set_{key}")(parameters[key])
        return self

    def pre_build_check(self):
        """Check if all required attribute parameters are set before building.

        Raises:
            ValueError: If any required parameter is missing.
        """
        if missing := [
            p for p in self.required_parameters if getattr(self, p) is None
        ]:
            error_message = (
                f"Required parameter(s) not set: {', '.join(missing)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

    @abstractmethod
    def build(self) -> Any:
        """Build and return the strategy object with the set parameters.

        Returns:
            strategy: The built strategy object.
        """


class BuilderDensityMixin:
    """Mixin class for Builder classes to set density and density_units.

    Methods:
        set_density: Set the density attribute and units.
    """

    def __init__(self):
        self.density = None

    def set_density(
        self,
        density: Union[float, NDArray[np.float_]],
        density_units: Optional[str] = "kg/m^3",
    ):
        """Set the density of the particle in kg/m^3.

        Args:
            density: Density of the particle.
            density_units: Units of the density. Default is *kg/m^3*
        """
        if np.any(density < 0):
            error_message = "Density must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.density = density * convert_units(density_units, "kg/m^3")
        return self


class BuilderSurfaceTensionMixin:
    """Mixin class for Builder classes to set surface_tension.

    Methods:
    -------
        set_surface_tension: Set the surface_tension attribute and units.
    """

    def __init__(self):
        self.surface_tension = None

    def set_surface_tension(
        self,
        surface_tension: Union[float, NDArray[np.float_]],
        surface_tension_units: Optional[str] = "N/m",
    ):
        """Set the surface tension of the particle in N/m.

        Args:
            surface_tension: Surface tension of the particle.
            surface_tension_units: Surface tension units. Default is *N/m*.
        """
        if np.any(surface_tension < 0):
            error_message = "Surface tension must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.surface_tension = surface_tension * convert_units(
            surface_tension_units, "N/m"
        )
        return self


class BuilderMolarMassMixin:
    """Mixin class for Builder classes to set molar_mass and molar_mass_units.

    Methods:
        set_molar_mass: Set the molar_mass attribute and units.
    """

    def __init__(self):
        self.molar_mass = None

    def set_molar_mass(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        molar_mass_units: Optional[str] = "kg/mol",
    ):
        """Set the molar mass of the particle in kg/mol.

        Args:
        -----
        - molar_mass: Molar mass of the particle.
        - molar_mass_units: Units of the molar mass. Default is *kg/mol*.
        """
        if np.any(molar_mass < 0):
            error_message = "Molar mass must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.molar_mass = molar_mass * convert_units(
            molar_mass_units, "kg/mol"
        )
        return self


class BuilderConcentrationMixin:
    """Mixin class for Builder classes to set concentration and
    concentration_units.

    Args:
        default_units: Default units of concentration. Default is *kg/m^3*.

    Methods:
        set_concentration: Set the concentration attribute and units.
    """

    def __init__(self, default_units: Optional[str] = "kg/m^3"):
        self.concentration = None
        self.default_units = default_units if default_units else "kg/m^3"

    def set_concentration(
        self,
        concentration: Union[float, NDArray[np.float_]],
        concentration_units: Optional[str] = None,
    ):
        """Set the concentration.

        Args:
            concentration: Concentration in the mixture.
            concentration_units: Units of the concentration.
            Default is *kg/m^3*.
        """
        if concentration_units is None:
            concentration_units = self.default_units
        if np.any(concentration < 0):
            error_message = "Concentration must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.concentration = concentration * convert_units(
            concentration_units, self.default_units
        )
        return self


class BuilderChargeMixin:
    """Mixin class for Builder classes to set charge and charge_units.

    Methods:
        set_charge: Set the charge attribute and units.
    """

    def __init__(self):
        self.charge = None

    def set_charge(
        self,
        charge: Union[float, NDArray[np.float_]],
        charge_units: Optional[str] = None,
    ):
        """Set the number of elemental charges on the particle.

        Args:
            charge: Charge of the particle [C].
            charge_units: Not used. (for interface consistency)
        """
        if charge_units is not None:
            logger.warning("Ignoring units for charge parameter.")
        self.charge = charge
        return self


class BuilderMassMixin:
    """Mixin class for Builder classes to set mass and mass_units.

    Methods:
        set_mass: Set the mass attribute and units.
    """

    def __init__(self):
        self.mass = None

    def set_mass(
        self,
        mass: Union[float, NDArray[np.float_]],
        mass_units: Optional[str] = "kg",
    ):
        """Set the mass of the particle in kg.

        Args:
            mass: Mass of the particle.
            mass_units: Units of the mass. Default is *kg*.

        Raises:
            ValueError: If mass is negative
        """
        if np.any(mass < 0):
            error_message = "Mass must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.mass = mass * convert_units(mass_units, "kg")
        return self


class BuilderRadiusMixin:
    """Mixin class for Builder classes to set radius and radius_units.

    Methods:
        set_radius: Set the radius attribute and units.
    """

    def __init__(self):
        self.radius = None

    def set_radius(
        self,
        radius: Union[float, NDArray[np.float_]],
        radius_units: Optional[str] = "m",
    ):
        """Set the radius of the particle in meters.

        Args:
            radius: Radius of the particle.
            radius_units: Units of the radius. Default is *m*.

        Raises:
            ValueError: If radius is negative
        """
        if np.any(radius < 0):
            error_message = "Radius must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.radius = radius * convert_units(radius_units, "m")
        return self


# mixins for strategy builders


class BuilderSurfaceStrategyMixin:
    """Mixin class for Builder classes to set surface_strategy.

    Methods:
        set_surface_strategy: Set the surface_strategy attribute.
    """

    def __init__(self):
        self.surface_strategy = None

    def set_surface_strategy(
        self,
        surface_strategy: SurfaceStrategy,
        surface_strategy_units: Optional[str] = None,
    ):
        """Set the surface strategy of the particle.

        Args:
            surface_strategy: Surface strategy of the particle.
            surface_strategy_units: Not used. (for interface consistency)
        """
        if surface_strategy_units is not None:
            logger.warning("Ignoring units for surface strategy parameter.")
        self.surface_strategy = surface_strategy
        return self


class BuilderActivityStrategyMixin:
    """Mixin class for Builder classes to set activity_strategy.

    Methods:
        set_activity_strategy: Set the activity_strategy attribute.
    """

    def __init__(self):
        self.activity_strategy = None

    def set_activity_strategy(
        self,
        activity_strategy: ActivityStrategy,
        activity_strategy_units: Optional[str] = None,
    ):
        """Set the activity strategy of the particle.

        Args:
            activity_strategy: Activity strategy of the particle.
            activity_strategy_units: Not used. (for interface consistency)
        """
        if activity_strategy_units is not None:
            logger.warning("Ignoring units for activity strategy parameter.")
        self.activity_strategy = activity_strategy
        return self


class BuilderDistributionStrategyMixin:
    """Mixin class for Builder classes to set distribution_strategy.

    Methods:
        set_distribution_strategy: Set the distribution_strategy attribute.
    """

    def __init__(self):
        self.distribution_strategy = None

    def set_distribution_strategy(
        self,
        distribution_strategy: DistributionStrategy,
        distribution_strategy_units: Optional[str] = None,
    ):
        """Set the distribution strategy of the particle.

        Args:
            distribution_strategy: Distribution strategy of the particle.
            distribution_strategy_units: Not used. (for interface consistency)
        """
        if distribution_strategy_units is not None:
            logger.warning(
                "Ignoring units for distribution strategy parameter."
            )
        self.distribution_strategy = distribution_strategy
        return self
