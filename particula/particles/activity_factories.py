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

    This class supports various strategies for vapor-pressure and activity
    calculations:

    - **mass_ideal**: Raoult's Law using mass fractions.
    - **molar_ideal**: Raoult's Law using mole fractions.
    - **kappa_parameter** / **kappa**: Hygroscopicity via Petters &
      Kreidenweis (2007).
    - **non_ideal_binary**: BAT-based organic-water activity (Gorkowski
      et al., 2019).

    Methods:
        get_builders:
            Provides a mapping from strategy type to its corresponding builder.
        get_strategy(strategy_type, parameters):
            Validates inputs and returns a strategy instance for the specified
            strategy type.

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

        params = {
            "molar_mass": 0.200,
            "oxygen2carbon": 0.4,
            "density": 1400.0,
        }
        bat_strategy = factory.get_strategy("non_ideal_binary", params)
        # bat_strategy -> ActivityNonIdealBinary
        ```

    References:
        - Petters, M. D., & Kreidenweis, S. M. (2007).
          "A single parameter representation of hygroscopic growth and cloud
          condensation nucleus activity." Atmospheric Chemistry and Physics,
          7(8), 1961–1971. https://doi.org/10.5194/acp-7-1961-2007
        - Gorkowski, K., Preston, T. C., & Zuend, A. (2019). Relative-
          humidity-dependent organic aerosol thermodynamics via an efficient
          reduced-complexity model. Atmospheric Chemistry and Physics,
          19(19), 13383-13410. https://doi.org/10.5194/acp-19-13383-2019
        - "Raoult's Law," [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).
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
            ```
        """
        return {
            "mass_ideal": ActivityIdealMassBuilder(),
            "molar_ideal": ActivityIdealMolarBuilder(),
            "kappa_parameter": ActivityKappaParameterBuilder(),
            "kappa": ActivityKappaParameterBuilder(),
            "non_ideal_binary": ActivityNonIdealBinaryBuilder(),
        }
