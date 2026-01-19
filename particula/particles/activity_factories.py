"""Activity strategy factories for calculating activity and partial pressure
of species in a mixture of liquids.
"""

from typing import Union

from particula.abc_factory import StrategyFactoryABC
from particula.particles.activity_builders import (
    ActivityIdealMassBuilder,
    ActivityIdealMolarBuilder,
    ActivityKappaParameterBuilder,
    ActivityNonIdealBinaryBuilder,
)
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityIdealMolar,
    ActivityKappaParameter,
    ActivityNonIdealBinary,
)


# There is a bit of work in the constructor, but it is necessary to set the
# types of the builders and strategies correctly.
class ActivityFactory(
    StrategyFactoryABC[
        Union[
            ActivityIdealMassBuilder,
            ActivityIdealMolarBuilder,
            ActivityKappaParameterBuilder,
            ActivityNonIdealBinaryBuilder,
        ],
        Union[
            ActivityIdealMass,
            ActivityIdealMolar,
            ActivityKappaParameter,
            ActivityNonIdealBinary,
        ],
    ]
):
    """Factory for creating activity strategy builders for liquid mixtures.

    This class supports strategies to compute activity and partial pressures
    of species based on different thermodynamic models:

    - **mass_ideal**: Raoult's Law based on mass fractions
    - **molar_ideal**: Raoult's Law based on mole fractions
    - **kappa_parameter** / **kappa**: Kappa hygroscopic parameter
      (Petters 2007)
    - **non_ideal_binary** / **binary_non_ideal**: BAT model for non-ideal
      organic–water mixtures (Gorkowski 2019)

    Methods:
        get_builders: Provides a mapping from strategy type to builder.
        get_strategy: Validates inputs and returns a strategy instance for the
            specified strategy type.

    Raises:
        ValueError: If the strategy type is unknown or if required parameters
            are missing or invalid.

    Examples:
        ```py title="Factory Usage Example"
        import particula as par
        factory = par.particles.ActivityFactory()

        # Ideal mass activity
        strategy = factory.get_strategy("mass_ideal")

        # Non-ideal binary (BAT model)
        params = {
            "molar_mass": 0.200,
            "oxygen2carbon": 0.4,
            "density": 1400.0,
        }
        bat_strategy = factory.get_strategy("non_ideal_binary", params)
        # bat_strategy.get_name() -> "ActivityNonIdealBinary"
        ```

    References:
        - Petters, M. D., & Kreidenweis, S. M. (2007).
          DOI:10.5194/acp-7-1961-2007
        - Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
          DOI:10.5194/acp-19-13383-2019
    """

    def get_builders(self):
        """Return a mapping of strategy types to their corresponding builders.

        Returns:
            dict[str, Any]: A dictionary mapping the activity strategy type
            to a builder instance.

        Supported types:
            - "mass_ideal": ActivityIdealMassBuilder (Raoult's Law, mass)
            - "molar_ideal": ActivityIdealMolarBuilder (Raoult's Law, mole)
            - "kappa_parameter": ActivityKappaParameterBuilder (kappa model)
            - "kappa": Alias for "kappa_parameter"
            - "non_ideal_binary": ActivityNonIdealBinaryBuilder (BAT model)
            - "binary_non_ideal": Alias for "non_ideal_binary"

        Examples:
            ```py title="Builders Retrieval Example"
            factory = ActivityFactory()
            builder_map = factory.get_builders()
            non_ideal_builder = builder_map["non_ideal_binary"]
            # non_ideal_builder -> ActivityNonIdealBinaryBuilder()
            ```
        """
        return {
            "mass_ideal": ActivityIdealMassBuilder(),
            "molar_ideal": ActivityIdealMolarBuilder(),
            "kappa_parameter": ActivityKappaParameterBuilder(),
            "kappa": ActivityKappaParameterBuilder(),
            "non_ideal_binary": ActivityNonIdealBinaryBuilder(),
            "binary_non_ideal": ActivityNonIdealBinaryBuilder(),
        }
