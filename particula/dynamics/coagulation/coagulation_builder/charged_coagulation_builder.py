"""
CoagulationBuilder for Charged Coagulation.
"""

from typing import Optional

from particula.abc_builder import BuilderABC

from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy import (
    ChargedCoagulationStrategy,
)
from particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin import (
    BuilderDistributionTypeMixin,
)
from particula.dynamics.coagulation.charged_kernel_strategy import (
    ChargedKernelStrategyABC,
)


class ChargedCoagulationBuilder(
    BuilderABC,
    BuilderDistributionTypeMixin,
):
    """Charged Coagulation Builder class for coagulation strategies.

    This class is used to create charged coagulation strategies based on the
    specified distribution type and kernel strategy. This provides a validation
    layer to ensure that the correct values are passed to the coagulation
    strategy.

    Methods:
        set_distribution_type(distribution_type): Set the distribution type.
        set_kernel_strategy(kernel_strategy): Set the kernel strategy.
        set_parameters(params): Set the parameters of the CoagulationStrategy
            object from a dictionary including optional units.
        build(): Validate and return the CoagulationStrategy object.
    """

    def __init__(self):
        required_parameters = ["distribution_type", "charged_kernel_strategy"]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionTypeMixin.__init__(self)
        self.charged_kernel_strategy = None

    def set_charged_kernel_strategy(
        self,
        charged_kernel_strategy: ChargedKernelStrategyABC,
        charged_kernel_strategy_units: Optional[str] = None,
    ):
        """Set the kernel strategy.

        Args:
            charged_kernel_strategy : The kernel strategy to be set.
            charged_kernel_strategy_units : Not used.

        Raises:
            ValueError: If the kernel strategy is not valid.
        """
        # check type
        if not isinstance(charged_kernel_strategy, ChargedKernelStrategyABC):
            message = (
                f"Invalid kernel strategy: {charged_kernel_strategy}. "
                f"Valid types -> {ChargedKernelStrategyABC.__subclasses__()}."
            )
            raise ValueError(message)

        if charged_kernel_strategy_units is not None:
            message = (
                f"Units for kernel strategy are not used. "
                f"Received: {charged_kernel_strategy_units}."
            )
            raise ValueError(message)
        self.charged_kernel_strategy = charged_kernel_strategy
        return self

    def build(self) -> CoagulationStrategyABC:
        """Validate and return the ChargedCoagulationStrategy object.

        Returns:
            CoagulationStrategy: Instance of the CoagulationStrategy object.
        """
        self.pre_build_check()

        return ChargedCoagulationStrategy(
            distribution_type=self.distribution_type,
            kernel_strategy=self.charged_kernel_strategy,
        )
