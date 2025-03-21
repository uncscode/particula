"""
Factory module to create a concrete VaporPressureStrategy object using
builders.
"""

from typing import Union
from particula.abc_factory import StrategyFactoryABC
from particula.gas.vapor_pressure_builders import (
    ConstantBuilder,
    AntoineBuilder,
    ClausiusClapeyronBuilder,
    WaterBuckBuilder,
)
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    AntoineVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    WaterBuckStrategy,
)


class VaporPressureFactory(
    StrategyFactoryABC[
        Union[
            ConstantBuilder,
            AntoineBuilder,
            ClausiusClapeyronBuilder,
            WaterBuckBuilder,
        ],
        Union[
            ConstantVaporPressureStrategy,
            AntoineVaporPressureStrategy,
            ClausiusClapeyronStrategy,
            WaterBuckStrategy,
        ],
    ]
):
    """
    Factory class to create vapor pressure strategy
    builders.

    This class provides a way to generate multiple vapor pressure calculation
    strategies (e.g., constant, Antoine, Clausius-Clapeyron, or Water Buck) by
    commissioning the appropriate builder. It is useful for scenarios requiring
    a flexible way to switch or extend vapor pressure calculation methods.

    Attributes:
        - None

    Methods:
    - get_builders : Returns the mapping of strategy types to builder
      instances.
    - get_strategy : Returns the selected vapor pressure strategy,
      given a strategy type and parameters.

    Examples:
        ```py title="Example VaporPressureFactory usage"
        import particula as par

        factory = par.gas.VaporPressureFactory()
        # Create a constant vapor pressure strategy:
        strategy = factory.get_strategy(
            "constant", {"constant_vapor_pressure": 101325.0}
        )
        # strategy is an instance of ConstantVaporPressureStrategy
        ```

    References:
        - "Vapor Pressure,"
        [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).
    """

    def get_builders(self):
        """
        Return a dictionary mapping strategy types to builder instances.

        Returns:
            dict:
                - "constant": ConstantBuilder
                - "antoine": AntoineBuilder
                - "clausius_clapeyron": ClausiusClapeyronBuilder
                - "water_buck": WaterBuckBuilder

        Examples:
            ```py
            import particula as par
            builders_dict = par.gas.VaporPressureFactory().get_builders()
            builder = builders_dict["constant"]
            # builder is an instance of ConstantBuilder
            ```
        """
        return {
            "constant": ConstantBuilder(),
            "antoine": AntoineBuilder(),
            "clausius_clapeyron": ClausiusClapeyronBuilder(),
            "water_buck": WaterBuckBuilder(),
        }
