"""Charged Coagulation Builder Module.

Provides a builder for creating `ChargedCoagulationStrategy` objects,
allowing electrostatic interactions in coagulation processes. Combines a
specified distribution type with a `ChargedKernelStrategyABC` to ensure
valid, flexible modeling of charged aerosol aggregation.

"""

from typing import Optional

from particula.abc_builder import BuilderABC

from ..charged_kernel_strategy import ChargedKernelStrategyABC
from ..coagulation_strategy.charged_coagulation_strategy import (
    ChargedCoagulationStrategy,
)
from ..coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from .coagulation_builder_mixin import BuilderDistributionTypeMixin


class ChargedCoagulationBuilder(
    BuilderABC,
    BuilderDistributionTypeMixin,
):
    """Charged Coagulation builder class.

    Creates a `ChargedCoagulationStrategy` based on a specified distribution
    type and a `ChargedKernelStrategyABC` instance, enforcing the correct
    parameters for modeling electrostatic interactions in aerosol coagulation.

    Attributes:
        - distribution_type : Distribution representation
          ("discrete", "continuous_pdf", or "particle_resolved").
        - charged_kernel_strategy : Instance of `ChargedKernelStrategyABC`
          for electrostatic kernel calculations.

    Methods:
    - set_distribution_type : Set the distribution type.
    - set_charged_kernel_strategy : Set the charged kernel strategy.
    - set_parameters : Configure parameters from a dictionary.
    - build : Validate inputs and return a `ChargedCoagulationStrategy`.

    Examples:
        ```py title="Example of using ChargedCoagulationBuilder"
        import particula as par
        builder = par.dynamics.ChargedCoagulationBuilder()
        builder.set_distribution_type("discrete")
        builder.set_charged_kernel_strategy(charged_kernel_strategy)
        coagulation_strategy = builder.build()
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). "Atmospheric Chemistry
          and Physics." Wiley.
    """

    def __init__(self):
        """Initialize the Charged coagulation builder.

        Sets up the builder with required parameters for creating a
        ChargedCoagulationStrategy, including distribution type and
        charged kernel strategy.
        """
        required_parameters = [
            "distribution_type",
            "charged_kernel_strategy",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionTypeMixin.__init__(self)
        self.charged_kernel_strategy = None

    def set_charged_kernel_strategy(
        self,
        charged_kernel_strategy: ChargedKernelStrategyABC,
        charged_kernel_strategy_units: Optional[str] = None,
    ):
        """Set the charged kernel strategy for electrostatic coagulation.

        Arguments:
            - charged_kernel_strategy : An instance of
              `ChargedKernelStrategyABC`.
            - charged_kernel_strategy_units : For interface consistency,
              unused.

        Raises:
            - ValueError : If the kernel strategy is invalid or units passed
              are unsupported.
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

        This method checks whether all required parameters have been
        specified (e.g., distribution type, charged kernel strategy)
        before creating a `ChargedCoagulationStrategy`.

        Returns:
            - CoagulationStrategyABC : The properly configured
              charged coagulation strategy.

        Examples:
            ```py title="Example of using ChargedCoagulationBuilder build"
            import particula as par
            builder = ChargedCoagulationBuilder()
            builder.set_distribution_type("discrete")
            builder.set_charged_kernel_strategy(charged_kernel_strategy)
            charged_strategy = builder.build()
            ```
        """
        self.pre_build_check()

        return ChargedCoagulationStrategy(
            distribution_type=self.distribution_type,
            kernel_strategy=self.charged_kernel_strategy,
        )
