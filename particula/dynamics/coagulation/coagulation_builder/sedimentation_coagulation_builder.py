"""Sedimentation Coagulation builder class."""

from particula.abc_builder import BuilderABC
from particula.dynamics.coagulation.coagulation_builder import (
    coagulation_builder_mixin,
)

from ..coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from ..coagulation_strategy.sedimentation_coagulation_strategy import (
    SedimentationCoagulationStrategy,
)


class SedimentationCoagulationBuilder(
    BuilderABC,
    coagulation_builder_mixin.BuilderDistributionTypeMixin,
):
    """Sedimentation Coagulation builder class.

    Creates a `SedimentationCoagulationStrategy` given a distribution type
    (e.g., "discrete", "continuous_pdf", or "particle_resolved"). Ensures
    the required parameters are set before building the strategy.

    Attributes:
        - distribution_type : Representation of the particle
          size distribution.

    Methods:
    - set_distribution_type : Set the distribution type.
    - build : Validate parameters and return a
      SedimentationCoagulationStrategy instance.

    Examples:
        ```py title="Example of using SedimentationCoagulationBuilder"
        import particula as par
        builder = SedimentationCoagulationBuilder()
        builder.set_distribution_type("discrete")
        strategy = builder.build()
        # strategy is now a SedimentationCoagulationStrategy instance
        ```
    """

    def __init__(self):
        """Initialize the Sedimentation coagulation builder.

        Sets up the builder with required parameters for creating a
        SedimentationCoagulationStrategy.
        """
        required_parameters = ["distribution_type"]
        BuilderABC.__init__(self, required_parameters)
        coagulation_builder_mixin.BuilderDistributionTypeMixin.__init__(self)

    def build(self) -> CoagulationStrategyABC:
        """Validate and return SedimentationCoagulationStrategy object."""
        self.pre_build_check()
        return SedimentationCoagulationStrategy(
            distribution_type=self.distribution_type,
        )
