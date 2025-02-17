"""
CoagulationBuilder for Brownian Coagulation.
"""

from particula.abc_builder import BuilderABC

from particula.dynamics.coagulation.coagulation_strategy import (
    CoagulationStrategyABC,
    BrownianCoagulationStrategy,
)
from particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin import (
    BuilderDistributionTypeMixin,
)


class BrownianCoagulationBuilder(
    BuilderABC,
    BuilderDistributionTypeMixin,
):
    """Brownian Coagulation Builder class for coagulation strategies.

    This class is used to create coagulation strategies based on the specified
    distribution type and kernel strategy. This provides a validation layer to
    ensure that the correct values are passed to the coagulation strategy.

    Methods:
        set_distribution_type(distribution_type): Set the distribution type.
        set_kernel_strategy(kernel_strategy): Set the kernel strategy.
        set_parameters(params): Set the parameters of the CoagulationStrategy
            object from a dictionary including optional units.
        build(): Validate and return the CoagulationStrategy object.
    """

    def __init__(self):
        required_parameters = ["distribution_type", "kernel_strategy"]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionTypeMixin.__init__(self)

    def build(self) -> CoagulationStrategyABC:
        """Validate and return the CoagulationStrategy object.

        Returns:
            CoagulationStrategy: Instance of the CoagulationStrategy object.
        """
        self.pre_build_check()

        return BrownianCoagulationStrategy(
            distribution_type=self.distribution_type,
        )
