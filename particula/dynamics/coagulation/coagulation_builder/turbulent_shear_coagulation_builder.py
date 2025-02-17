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
)


class TurbulentShearCoagulationBuilder(
    BuilderABC,
    BuilderDistributionTypeMixin,
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
        self.turbulent_dissipation = None
        self.fluid_density = None

    def set_turbulent_dissipation(
        self,
        turbulent_dissipation: float,
        turbulent_dissipation_units: str = None,
    ):
        """Set the turbulent dissipation rate."""
        if turbulent_dissipation_units is not None:
            # In this example, just note we ignore units:
            pass
        self.turbulent_dissipation = turbulent_dissipation
        return self

    def set_fluid_density(
        self,
        fluid_density: float,
        fluid_density_units: str = None,
    ):
        """Set the fluid density."""
        if fluid_density_units is not None:
            pass
        self.fluid_density = fluid_density
        return self

    def build(self) -> CoagulationStrategyABC:
        """Validate and return the TurbulentShearCoagulationStrategy object."""
        """Validate and return the TurbulentShearCoagulationStrategy object.

        Returns:
            CoagulationStrategy: Instance of the CoagulationStrategy object.
        """
        return TurbulentShearCoagulationStrategy(
            distribution_type=self.distribution_type,
            turbulent_dissipation=self.turbulent_dissipation,
            fluid_density=self.fluid_density,
        )
