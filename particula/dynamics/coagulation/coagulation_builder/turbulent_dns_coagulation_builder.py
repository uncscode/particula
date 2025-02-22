"""
CoagulationBuilder for Turbulent DNS Coagulation.
"""

from typing import Optional
import logging

from particula.abc_builder import BuilderABC

from particula.util.validate_inputs import validate_inputs
from particula.util.converting.convert_units import get_unit_conversion

from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from particula.dynamics.coagulation.coagulation_strategy.trubulent_dns_coagulation_strategy import (
    TurbulentDNSCoagulationStrategy,
)
from particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin import (
    BuilderDistributionTypeMixin,
    BuilderTurbulentDissipationMixin,
    BuilderFluidDensityMixin,
)

logger = logging.getLogger("particula")


# pylint: disable=duplicate-code
class TurbulentDNSCoagulationBuilder(
    BuilderABC,
    BuilderDistributionTypeMixin,
    BuilderTurbulentDissipationMixin,
    BuilderFluidDensityMixin,
):
    """
    Turbulent DNS Coagulation Builder class.

    This class is used to create coagulation strategies for turbulent DNS
    coagulation and ensures that the correct values (distribution_type,
    turbulent_dissipation, fluid_density, reynolds_lambda, relative_velocity)
    are passed.
    """

    def __init__(self):
        required_parameters = [
            "distribution_type",
            "turbulent_dissipation",
            "fluid_density",
            "reynolds_lambda",
            "relative_velocity",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionTypeMixin.__init__(self)
        BuilderTurbulentDissipationMixin.__init__(self)
        BuilderFluidDensityMixin.__init__(self)
        self.reynolds_lambda = None
        self.relative_velocity = None

    @validate_inputs({"reynolds_lambda": "nonnegative"})
    def set_reynolds_lambda(
        self,
        reynolds_lambda: float,
        reynolds_lambda_units: Optional[str] = None,
    ):
        """
        Set the Reynolds lambda. This is a measure of the turbulence length
        scale, of airflow, and is used in the turbulent DNS model.
        Shorthand for the Taylor-scale Reynolds number, Reλ.

        Args:
            - reynolds_lambda : Reynolds lambda.
            - reynolds_lambda_units : Units of the Reynolds lambda
                [dimensionless].
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
        """
        Set the relative vertical velocity. This is the relative
        velocity between particles and airflow,
        (excluding turbulence, and gravity).

        Args:
            - relative_velocity : Relative velocity.
            - relative_velocity_units : Units of the relative velocity
                [m/s].
        """
        if relative_velocity_units == "m/s":
            self.relative_velocity = relative_velocity
            return self
        self.relative_velocity = relative_velocity * get_unit_conversion(
            relative_velocity_units, "m/s"
        )
        return self

    def build(self) -> CoagulationStrategyABC:
        self.pre_build_check()
        return TurbulentDNSCoagulationStrategy(
            distribution_type=self.distribution_type,
            turbulent_dissipation=self.turbulent_dissipation,
            fluid_density=self.fluid_density,
            reynolds_lambda=self.reynolds_lambda,
            relative_velocity=self.relative_velocity,
        )
