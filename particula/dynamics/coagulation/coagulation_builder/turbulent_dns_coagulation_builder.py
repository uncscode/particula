"""
CoagulationBuilder for Turbulent DNS Coagulation.
"""

from particula.abc_builder import BuilderABC

from particula.dynamics.coagulation.coagulation_strategy import (
    CoagulationStrategyABC,
    TurbulentDNSCoagulationStrategy,
)
from particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin import (
    BuilderDistributionTypeMixin,
    BuilderTurbulentDissipationMixin,
    BuilderFluidDensityMixin,
)


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

    def set_reynolds_lambda(self, reynolds_lambda: float):
        self.reynolds_lambda = reynolds_lambda
        return self

    def set_relative_velocity(self, relative_velocity: float):
        self.relative_velocity = relative_velocity
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
