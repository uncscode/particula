"""
Sedimentation Coagulation builder class.
"""

from particula.abc_builder import BuilderABC
from particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin import (
    BuilderDistributionTypeMixin,
)
from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from particula.dynamics.coagulation.coagulation_strategy.sedimentation_coagulation_strategy import (
    SedimentationCoagulationStrategy,
)


class SedimentationCoagulationBuilder(
    BuilderABC, BuilderDistributionTypeMixin
):
    """
    Sedimentation Coagulation builder class.

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
        required_parameters = ["distribution_type"]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionTypeMixin.__init__(self)

    def build(self) -> CoagulationStrategyABC:
        """
        Validate and return the SedimentationCoagulationStrategy object.
        """
        self.pre_build_check()
        return SedimentationCoagulationStrategy(
            distribution_type=self.distribution_type,
        )
