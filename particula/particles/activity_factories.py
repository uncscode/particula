"""Activity strategy factories for calculating activity and partial pressure
of species in a mixture of liquids.
"""

from typing import Union

from particula.abc_factory import StrategyFactoryABC
from particula.particles.activity_builders import (
    ActivityIdealMassBuilder,
    ActivityIdealMolarBuilder,
    ActivityKappaParameterBuilder,
)
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityIdealMolar,
    ActivityKappaParameter,
)


# There is a bit of work in the constructor, but it is necessary to set the
# types of the builders and strategies correctly.
class ActivityFactory(
    StrategyFactoryABC[
        Union[
            ActivityIdealMassBuilder,
            ActivityIdealMolarBuilder,
            ActivityKappaParameterBuilder,
        ],
        Union[ActivityIdealMass, ActivityIdealMolar, ActivityKappaParameter],
    ]
):
    """Factory for creating activity strategy builders for liquid mixtures.

    This class supports various strategies (e.g., mass-ideal, molar-ideal,
    kappa-parameter) to compute activity and partial pressures of species
    based on Raoult's Law or kappa hygroscopic parameter.

    Methods:
        get_builders:
            Provides a mapping from strategy type to its corresponding builder.
        get_strategy(strategy_type, parameters):
            Validates inputs and returns a strategy instance for the specified
            strategy type (e.g., 'mass_ideal', 'molar_ideal', or
            'kappa_parameter').

    Returns:
        - ActivityStrategy: Instance configured for the chosen activity
            approach.

    Raises:
        - ValueError: If the strategy type is unknown or if required parameters
          are missing or invalid.

    Examples:
        ```py title="Factory Usage Example"
        import particula as par
        factory = par.particles.ActivityFactory()
        strategy = factory.get_strategy("mass_ideal")
        result = strategy.activity([1.0, 2.0, 3.0])
        # result -> ...
        ```

    References:
    - "Raoult's Law,"
        [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).
    """

    def get_builders(self):
        """Return a mapping of strategy types to their corresponding builders.

        Returns:
            dict[str, Any]: A dictionary mapping the activity strategy type
            (e.g., 'mass_ideal', 'molar_ideal', 'kappa_parameter') to a builder
            instance.

        Examples:
            ```py title="Builders Retrieval Example"
            factory = ActivityFactory()
            builder_map = factory.get_builders()
            mass_ideal_builder = builder_map["mass_ideal"]
            # mass_ideal_builder -> ActivityIdealMassBuilder()
        """
        return {
            "mass_ideal": ActivityIdealMassBuilder(),
            "molar_ideal": ActivityIdealMolarBuilder(),
            "kappa_parameter": ActivityKappaParameterBuilder(),
            "kappa": ActivityKappaParameterBuilder(),
        }
