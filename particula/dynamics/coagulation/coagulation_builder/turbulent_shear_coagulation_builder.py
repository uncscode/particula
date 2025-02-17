"""
CoagulationBuilder for Turbulent Shear Coagulation.
"""

from particula.abc_builder import BuilderABC

from particula.dynamics.coagulation.coagulation_strategy import (
    CoagulationStrategyABC,
    TurbulentShearCoagulationStrategy,
)
from particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin import (
    BuilderDistributionTypeMixin,
    BuilderTurbulentDissipationMixin,
    BuilderFluidDensityMixin,
)


class TurbulentShearCoagulationBuilder(
    BuilderABC,
    BuilderDistributionTypeMixin,
    BuilderTurbulentDissipationMixin,
    BuilderFluidDensityMixin,
):
    """Turbulent Shear Coagulation Builder class.

    This class is used to create coagulation strategies for turbulent shear
    coagulation. This provides a validation layer to ensure that the correct
    values are passed to the coagulation strategy.

    Methods:
        set_distribution_type(distribution_type): Set the distribution type.
        set_turbulent_dissipation(turbulent_dissipation): Set the turbulent
            dissipation rate.
        set_fluid_density(fluid_density): Set the fluid density.
        build(): Validate and return the TurbulentShearCoagulationStrategy
            object.
    """

    def __init__(self):
        required_parameters = [
            "distribution_type",
            "turbulent_dissipation",
            "fluid_density",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionTypeMixin.__init__(self)
        BuilderTurbulentDissipationMixin.__init__(self)
        BuilderFluidDensityMixin.__init__(self)

    def build(self) -> CoagulationStrategyABC:
        """Validate and return the TurbulentShearCoagulationStrategy object.

        Returns:
            CoagulationStrategy: Instance of the CoagulationStrategy object.
        """
        return TurbulentShearCoagulationStrategy(
            distribution_type=self.distribution_type,
            turbulent_dissipation=self.turbulent_dissipation,
            fluid_density=self.fluid_density,
        )
