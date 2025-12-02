"""Turbulent Shear Coagulation Builder Module."""

from particula.abc_builder import BuilderABC
from particula.dynamics.coagulation.coagulation_builder import (
    coagulation_builder_mixin,
)

from ..coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from ..coagulation_strategy.turbulent_shear_coagulation_strategy import (
    TurbulentShearCoagulationStrategy,
)


# pylint: disable=duplicate-code
class TurbulentShearCoagulationBuilder(
    BuilderABC,
    coagulation_builder_mixin.BuilderDistributionTypeMixin,
    coagulation_builder_mixin.BuilderTurbulentDissipationMixin,
    coagulation_builder_mixin.BuilderFluidDensityMixin,
):
    """Turbulent shear coagulation builder.

    Creates a TurbulentShearCoagulationStrategy that calculates coagulation
    rates under turbulent flow conditions. Ensures the correct distribution
    type, turbulent dissipation, and fluid density values are provided.

    Attributes:
        - distribution_type : Type of the particle distribution
          ("discrete", "continuous_pdf", or "particle_resolved").
        - turbulent_dissipation : Turbulent energy dissipation rate (m²/s³).
        - fluid_density : Fluid density (kg/m³) for the coagulation medium.

    Methods:
        - set_distribution_type : Set the distribution type.
        - set_turbulent_dissipation : Set turbulent dissipation rate.
        - set_fluid_density : Set fluid density.
        - build : Validate parameters and return a
          TurbulentShearCoagulationStrategy.

    Examples:
        ```py title="Turbulent Shear Coagulation Builder Example"
        import particula as par
        builder = par.dynamics.TurbulentShearCoagulationBuilder()
        builder.set_distribution_type("discrete")
        builder.set_turbulent_dissipation(1e-3)
        builder.set_fluid_density(1000.)
        strategy = builder.build()
        # Now 'strategy' can be used to compute turbulent shear coagulation
        # rates.
        ```

    References:
        - Saffman, P. G., & Turner, J. S. (1956). "On the collision of drops
          in turbulent clouds." J. Fluid Mech., 1, 16-30.
    """

    def __init__(self):
        """Initialize the TurbulentShearCoagulationBuilder.

        Returns:
            - None

        Note:
            Some default values may be set by the mixins to guide the user
            toward valid operation.
        """
        required_parameters = [
            "distribution_type",
            "turbulent_dissipation",
            "fluid_density",
        ]
        BuilderABC.__init__(self, required_parameters)
        coagulation_builder_mixin.BuilderDistributionTypeMixin.__init__(self)
        mixin_class = coagulation_builder_mixin.BuilderTurbulentDissipationMixin
        mixin_class.__init__(self)
        coagulation_builder_mixin.BuilderFluidDensityMixin.__init__(self)

    def build(self) -> CoagulationStrategyABC:
        """Construct a TurbulentShearCoagulationStrategy.

        This method performs a final check to ensure all required parameters
        have been set. It then creates and returns an instance of
        TurbulentShearCoagulationStrategy.

        Returns:
            - The resulting turbulent shear coagulation strategy object.
        """
        self.pre_build_check()
        return TurbulentShearCoagulationStrategy(
            distribution_type=self.distribution_type,
            turbulent_dissipation=self.turbulent_dissipation,
            fluid_density=self.fluid_density,
        )
