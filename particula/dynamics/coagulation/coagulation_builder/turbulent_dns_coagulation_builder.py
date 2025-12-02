"""Turbulent DNS Coagulation Builder Module.

Provides a builder for creating `TurbulentDNSCoagulationStrategy`
instances, which handle coagulation processes under Direct Numerical
Simulation (DNS) of turbulent flows. Ensures that required parameters
such as distribution type, turbulent dissipation, fluid density,
reynolds_lambda (Taylor-scale Reynolds number), and relative_velocity
are supplied.
"""

import logging
from typing import Optional

from particula.abc_builder import BuilderABC
from particula.dynamics.coagulation.coagulation_builder import (
    coagulation_builder_mixin,
)
from particula.util.convert_units import get_unit_conversion
from particula.util.validate_inputs import validate_inputs

from ..coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from ..coagulation_strategy.turbulent_dns_coagulation_strategy import (
    TurbulentDNSCoagulationStrategy,
)

logger = logging.getLogger("particula")


# pylint: disable=duplicate-code
class TurbulentDNSCoagulationBuilder(
    BuilderABC,
    coagulation_builder_mixin.BuilderDistributionTypeMixin,
    coagulation_builder_mixin.BuilderTurbulentDissipationMixin,
    coagulation_builder_mixin.BuilderFluidDensityMixin,
):
    """Turbulent DNS coagulation builder class.

    Creates and configures a `TurbulentDNSCoagulationStrategy` to simulate
    coagulation in turbulent flow fields using Direct Numerical Simulation
    parameters. This builder enforces that the required parameters
    (distribution_type, turbulent_dissipation, fluid_density, reynolds_lambda,
    relative_velocity) are set prior to building the strategy.

    Attributes:
        - distribution_type : The particle distribution type
          ("discrete", "continuous_pdf", or "particle_resolved").
        - turbulent_dissipation : Rate of turbulent energy dissipation (m²/s³).
        - fluid_density : Fluid density in kg/m³ (e.g., air density).
        - reynolds_lambda : Taylor-scale Reynolds number (dimensionless).
        - relative_velocity : Relative velocity in m/s (particle vs. airflow).

    Methods:
    - set_distribution_type : Set the distribution type.
    - set_turbulent_dissipation : Set the turbulent dissipation rate.
    - set_fluid_density : Set the fluid density.
    - set_reynolds_lambda : Set the Taylor-scale Reynolds number.
    - set_relative_velocity : Set the relative velocity.
    - build : Validate parameters and return a
      `TurbulentDNSCoagulationStrategy`.

    Examples:
        ```py title="Turbulent DNS Builder Example"
        builder = TurbulentDNSCoagulationBuilder()
        builder.set_distribution_type("discrete")
        builder.set_turbulent_dissipation(1e-3)
        builder.set_fluid_density(1.225)
        builder.set_reynolds_lambda(250.)
        builder.set_relative_velocity(0.5, "m/s")
        strategy = builder.build()
        # Now 'strategy' can be used to compute DNS-based coagulation rates.

    References:
        - Saffman, P. G., & Turner, J. S. (1956) "On the collision of drops
          in turbulent clouds." Journal of Fluid Mechanics, 1(1): 16–30.
    """

    def __init__(self):
        """Initialize the Turbulent DNS coagulation builder.

        Sets up the builder with required parameters for creating a
        TurbulentDNSCoagulationStrategy, including distribution type,
        turbulent dissipation, fluid density, Reynolds lambda, and
        relative velocity.
        """
        required_parameters = [
            "distribution_type",
            "turbulent_dissipation",
            "fluid_density",
            "reynolds_lambda",
            "relative_velocity",
        ]
        BuilderABC.__init__(self, required_parameters)
        coagulation_builder_mixin.BuilderDistributionTypeMixin.__init__(self)
        mixin_class = coagulation_builder_mixin.BuilderTurbulentDissipationMixin
        mixin_class.__init__(self)
        coagulation_builder_mixin.BuilderFluidDensityMixin.__init__(self)
        self.reynolds_lambda = None
        self.relative_velocity = None

    @validate_inputs({"reynolds_lambda": "nonnegative"})
    def set_reynolds_lambda(
        self,
        reynolds_lambda: float,
        reynolds_lambda_units: Optional[str] = None,
    ):
        """Set the Taylor-scale Reynolds number (Reλ).

        Represents a measure of turbulence intensity in DNS flows.
        When specifying units, only "dimensionless" is recognized here.
        Any other unit triggers a warning and is treated as dimensionless.

        Arguments:
            - reynolds_lambda : Numeric value for Reλ.
            - reynolds_lambda_units : String indicating units
              (default "dimensionless").

        Returns:
            - self : The builder instance for chaining.

        Examples:
            ```py
            builder.set_reynolds_lambda(250.)
            ```
        """
        if reynolds_lambda_units == "dimensionless":
            self.reynolds_lambda = reynolds_lambda
            return self
        if reynolds_lambda_units is not None:
            logger.warning("Units for reynolds_lambda are not used. ")
        self.reynolds_lambda = reynolds_lambda
        return self

    @validate_inputs({"relative_velocity": "finite"})
    def set_relative_velocity(
        self,
        relative_velocity: float,
        relative_velocity_units: str,
    ):
        """Set the relative particle-airflow velocity for DNS coagulation.

        This value is typically a background flow velocity or a
        sedimentation-adjusted velocity, excluding turbulence.

        Arguments:
            - relative_velocity : Numeric value of velocity.
            - relative_velocity_units : Units of the velocity
              (e.g., "m/s").

        Returns:
            - self : The builder instance for chaining.
        """
        if relative_velocity_units == "m/s":
            self.relative_velocity = relative_velocity
            return self
        self.relative_velocity = relative_velocity * get_unit_conversion(
            relative_velocity_units, "m/s"
        )
        return self

    def build(self) -> CoagulationStrategyABC:
        """Construct a TurbulentDNSCoagulationStrategy.

        Validates the required parameters, then instantiates a
        `TurbulentDNSCoagulationStrategy` for DNS-based coagulation
        calculations.

        Returns:
            CoagulationStrategyABC: The configured DNS coagulation
            strategy.
        """
        self.pre_build_check()
        return TurbulentDNSCoagulationStrategy(
            distribution_type=self.distribution_type,
            turbulent_dissipation=self.turbulent_dissipation,
            fluid_density=self.fluid_density,
            reynolds_lambda=self.reynolds_lambda,
            relative_velocity=self.relative_velocity,
        )
