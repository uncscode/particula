"""Factory module to create a concrete LatentHeatStrategy object using
builders.
"""

from typing import Union

from particula.abc_factory import StrategyFactoryABC
from particula.gas.latent_heat_builders import (
    ConstantLatentHeatBuilder,
    LinearLatentHeatBuilder,
    PowerLawLatentHeatBuilder,
)
from particula.gas.latent_heat_strategies import (
    ConstantLatentHeat,
    LatentHeatStrategy,
    LinearLatentHeat,
    PowerLawLatentHeat,
)


class LatentHeatFactory(
    StrategyFactoryABC[
        Union[
            ConstantLatentHeatBuilder,
            LinearLatentHeatBuilder,
            PowerLawLatentHeatBuilder,
        ],
        Union[
            LatentHeatStrategy,
            ConstantLatentHeat,
            LinearLatentHeat,
            PowerLawLatentHeat,
        ],
    ]
):
    """Factory class to create latent heat strategy builders.

    This class provides a way to generate multiple latent heat calculation
    strategies (e.g., constant, linear, or power-law) by commissioning the
    appropriate builder. It is useful for scenarios requiring a flexible way
    to switch or extend latent heat calculation methods.

    Attributes:
        - None

    Methods:
    - get_builders : Returns the mapping of strategy types to builder
      instances.
    - get_strategy : Returns the selected latent heat strategy, given a
      strategy type and parameters.

    Examples:
        ```py title="Example LatentHeatFactory usage"
        import particula as par

        factory = par.gas.LatentHeatFactory()
        strategy = factory.get_strategy(
            "constant",
            {"latent_heat_ref": 2.26e6, "latent_heat_ref_units": "J/kg"},
        )
        # strategy is an instance of ConstantLatentHeat
        ```
    """

    def get_builders(self):
        """Return a dictionary mapping strategy types to builder instances.

        Returns:
            dict:
                - "constant": ConstantLatentHeatBuilder
                - "linear": LinearLatentHeatBuilder
                - "power_law": PowerLawLatentHeatBuilder

        Examples:
            ```py
            import particula as par
            builders_dict = par.gas.LatentHeatFactory().get_builders()
            builder = builders_dict["constant"]
            # builder is an instance of ConstantLatentHeatBuilder
            ```
        """
        return {
            "constant": ConstantLatentHeatBuilder(),
            "linear": LinearLatentHeatBuilder(),
            "power_law": PowerLawLatentHeatBuilder(),
        }
