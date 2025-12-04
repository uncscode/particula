"""Coagulation Builder Mixin Classes.

Provides reusable mixin classes for building coagulation strategies
with validated inputs (e.g., distribution type, turbulent dissipation,
and fluid density). These mixins can be combined to form full-featured
coagulation builders, ensuring correct parameter values are passed to
the final coagulation strategy.
"""

# pylint: disable=too-few-public-methods

import logging
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from particula.util.convert_units import get_unit_conversion
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")


class BuilderDistributionTypeMixin:
    """Mixin class for distribution type in coagulation strategies.

    Provides an interface to set the distribution type for coagulation
    strategies. Ensures the chosen `distribution_type` is valid.

    Attributes:
        - distribution_type : Stores the selected distribution type
          (e.g., "discrete", "continuous_pdf", "particle_resolved").

    Methods:
    - set_distribution_type : Set and validate the distribution type.

    Examples:
        ```py title="Example of using BuilderDistributionTypeMixin"
        builder = SomeCoagulationBuilder()
        builder.set_distribution_type("discrete")
        # builder.distribution_type -> "discrete"
        ```
    """

    def __init__(self):
        """Initialize the distribution type mixin.

        Sets the distribution_type attribute to None, to be configured later.
        """
        self.distribution_type = None

    def set_distribution_type(
        self,
        distribution_type: str,
        distribution_type_units: Optional[str] = None,
    ):
        """Set the distribution type.

        Args:
            distribution_type : The distribution type to be set.
                Options are "discrete", "continuous_pdf", "particle_resolved".
            distribution_type_units : Not used.

        Returns:
            - The instance of the class with the updated
                distribution_type attribute.

        Raises:
            ValueError: If the distribution type is not valid.

        Examples:
            ```py title="Example of using set_distribution_type"
            builder = SomeCoagulationBuilder()
            builder.set_distribution_type("discrete")
            # builder.distribution_type -> "discrete"
            ```
        """
        valid_distribution_types = [
            "discrete",
            "continuous_pdf",
            "particle_resolved",
        ]
        if distribution_type not in valid_distribution_types:
            message = (
                f"Invalid distribution type: {distribution_type}. "
                f"Valid types are: {valid_distribution_types}."
            )
            logger.error(message)
            raise ValueError(message)
        if distribution_type_units is not None:
            message = (
                f"Units for distribution type are not used. "
                f"Received: {distribution_type_units}."
            )
            logger.warning(message)
        self.distribution_type = distribution_type
        return self


class BuilderTurbulentDissipationMixin:
    """Mixin class for turbulent shear parameters.

    Adds methods and attributes for setting and validating
    turbulent dissipation parameters in coagulation strategies.

    Attributes:
        - turbulent_dissipation : Numeric value of the energy dissipation
          rate in m^2/s^3 (default units).

    Methods:
    - set_turbulent_dissipation : Set and validate the turbulent
        dissipation rate.

    Examples:
        ```py title="Example of using BuilderTurbulentDissipationMixin"
        builder.set_turbulent_dissipation(1e-3, "m^2/s^3")
        ```
    """

    def __init__(self):
        """Initialize the turbulent dissipation mixin.

        Sets the turbulent_dissipation attribute to None.
        """
        self.turbulent_dissipation = None

    @validate_inputs({"turbulent_dissipation": "nonnegative"})
    def set_turbulent_dissipation(
        self,
        turbulent_dissipation: float,
        turbulent_dissipation_units: str,
    ):
        """Set the turbulent dissipation rate.

        Arguments:
            turbulent_dissipation : Turbulent dissipation rate.
            turbulent_dissipation_units : Units of the turbulent dissipation
                rate. Default is *m^2/s^3*.

        Returns:
            - The instance of the class with the updated
                turbulent_dissipation attribute.

        Raises:
            ValueError: Must be non-negative value.

        Examples:
            ```py title="Example of using set_turbulent_dissipation"
            builder = SomeCoagulationBuilder()
            builder.set_turbulent_dissipation(1e-3, "m^2/s^3")
            # builder.turbulent_dissipation -> 1e-3
            ```
        """
        if turbulent_dissipation_units == "m^2/s^3":
            self.turbulent_dissipation = turbulent_dissipation
            return self
        self.turbulent_dissipation = (
            turbulent_dissipation
            * get_unit_conversion(turbulent_dissipation_units, "m^2/s^3")
        )
        return self


class BuilderFluidDensityMixin:
    """Mixin class for fluid density parameters.

    Adds methods and attributes for setting and validating fluid
    density in coagulation strategies.

    Attributes:
        - fluid_density : Numeric value representing fluid density
          in kg/m^3 (default units).

    Methods:
    - set_fluid_density : Set and validate the fluid density.

    Examples:
        ```py title="Example of using BuilderFluidDensityMixin"
        builder.set_fluid_density(1.225, "kg/m^3")
        ```
    """

    def __init__(self):
        """Initialize the fluid density mixin.

        Sets the fluid_density attribute to None, to be configured later.
        """
        self.fluid_density = None

    @validate_inputs({"fluid_density": "positive"})
    def set_fluid_density(
        self,
        fluid_density: Union[float, NDArray[np.float64]],
        fluid_density_units: str,
    ):
        """Set the density of the particle in kg/m^3.

        Arguments:
            density : Density of the particle.
            density_units : Units of the density. Default is *kg/m^3*

        Returns:
            - The instance of the class with the updated
                fluid_density attribute.

        Raises:
            ValueError: Must be positive value.

        Examples:
            ```py title="Example of using set_fluid_density"
            builder = SomeCoagulationBuilder()
            builder.set_fluid_density(1.225, "kg/m^3")
            # builder.fluid_density -> 1.225
            ```
        """
        if fluid_density_units == "kg/m^3":
            self.fluid_density = fluid_density
            return self
        self.fluid_density = fluid_density * get_unit_conversion(
            fluid_density_units, "kg/m^3"
        )
        return self
