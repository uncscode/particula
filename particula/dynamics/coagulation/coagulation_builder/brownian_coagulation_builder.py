"""Brownian Coagulation Builder Module.

Provides a builder for creating `BrownianCoagulationStrategy` objects
based on a chosen particle distribution type (e.g., "discrete",
"continuous_pdf", or "particle_resolved"). Validates that required
parameters are set before returning the final strategy.

References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). "Atmospheric Chemistry
      and Physics." Wiley.
"""

from particula.abc_builder import BuilderABC

from ..coagulation_strategy.brownian_coagulation_strategy import (
    BrownianCoagulationStrategy,
)
from ..coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from .coagulation_builder_mixin import (
    BuilderDistributionTypeMixin,
)


class BrownianCoagulationBuilder(
    BuilderABC,
    BuilderDistributionTypeMixin,
):
    """Brownian Coagulation builder class.

    Creates a `BrownianCoagulationStrategy` given a distribution type
    (e.g., "discrete", "continuous_pdf", or "particle_resolved"). Ensures
    the required parameters are set before building the strategy.

    Attributes:
        - distribution_type : Representation of the particle
          size distribution.

    Methods:
    - set_distribution_type : Assign the distribution type.
    - set_parameters : Inherited from BuilderABC to set multiple
      parameters via a dict.
    - build : Validate and return a `BrownianCoagulationStrategy`.

    Examples:
        ```py title="Example of using BrownianCoagulationBuilder"
        import particula as par
        builder = BrownianCoagulationBuilder()
        builder.set_distribution_type("discrete")
        strategy = builder.build()
        # strategy is now a BrownianCoagulationStrategy instance
        ```
    """

    def __init__(self):
        """Initialize the Brownian coagulation builder.

        Sets up the builder with required parameters for creating a
        BrownianCoagulationStrategy.
        """
        required_parameters = ["distribution_type"]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionTypeMixin.__init__(self)

    def build(self) -> CoagulationStrategyABC:
        """Validate and return the BrownianCoagulationStrategy object.

        Checks that all required parameters (e.g., distribution_type) are set
        before creating and returning a `BrownianCoagulationStrategy`.

        Returns:
            BrownianCoagulationStrategy : The newly created
            Brownian coagulation strategy.
        """
        self.pre_build_check()

        return BrownianCoagulationStrategy(
            distribution_type=self.distribution_type,
        )
