"""Factory module to create a concrete VaporPressureStrategy object using
builders."""

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
    """Factory class to create vapor pressure strategy builders

    Factory class to create vapor pressure strategy builders for calculating
    vapor pressure of gas species.

    Methods:
        - get_builders : Returns the mapping of strategy types to builder
            instances.
        - get_strategy : Gets the strategy instance
            for the specified strategy type.
            - strategy_type : Type of vapor pressure strategy to use, can be
                'constant', 'antoine', 'clausius_clapeyron', or 'water_buck'.
            - parameters : Parameters required for the
                builder, dependent on the chosen strategy type.
                    - constant: constant_vapor_pressure
                    - antoine: A, B, C
                    - clausius_clapeyron: A, B, C
                    - water_buck: No parameters are required.

    Returns:
        VaporPressureStrategy : An instance of the specified
            VaporPressureStrategy.

    Raises:
        ValueError : If an unknown strategy type is provided.
        ValueError : If any required key is missing during check_keys or
            pre_build_check, or if trying to set an invalid parameter.

    Example:
        ``` py title="constant vapor pressure strategy"
        strategy_is = VaporPressureFactory().get_strategy("constant")
        # returns ConstantVaporPressureStrategy
        ```
    """

    def get_builders(self):
        """Returns the mapping of strategy types to builder instances.

        Returns:
            A dictionary mapping strategy types to builder instances.
                - constant: ConstantBuilder
                - antoine: AntoineBuilder
                - clausius_clapeyron: ClausiusClapeyronBuilder
                - water_buck: WaterBuckBuilder
        """
        return {
            "constant": ConstantBuilder(),
            "antoine": AntoineBuilder(),
            "clausius_clapeyron": ClausiusClapeyronBuilder(),
            "water_buck": WaterBuckBuilder(),
        }
