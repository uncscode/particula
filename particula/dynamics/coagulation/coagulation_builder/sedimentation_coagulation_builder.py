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
