"""Factory module to create a concrete VaporPressureStrategy object using
builders."""

from typing import Optional
from particula.next.gas import vapor_pressure_builders
from particula.next.gas.vapor_pressure_strategies import VaporPressureStrategy


def vapor_pressure_factory(
    strategy: str,
    parameters: Optional[dict] = None  # type: ignore
) -> VaporPressureStrategy:
    """Factory method to create a concrete VaporPressureStrategy object using
    builders.

    Args:
    ----
    - strategy (str): The strategy to use for vapor pressure calculations.
      Options: "constant", "antoine", "clausius_clapeyron", "water_buck".
    - **kwargs: Additional keyword arguments required for the strategy.

    Returns:
        VaporPressureStrategy: An instance of the specified
            VaporPressureStrategy.

    Raises:
        ValueError: If an unknown strategy type is provided.
        ValueError: If any required key is missing during check_keys or
            pre_build_check, or if trying to set an invalid parameter.

    Example:
    >>> strategy_is = VaporPressureFactory().get_strategy("constant")
    """

    def get_builders(self):
        """Returns the mapping of strategy types to builder instances.

        Returns:
            Dict[str, Any]: A dictionary mapping strategy types to builder
                instances.
                constant: ConstantBuilder
                antoine: AntoineBuilder
                clausius_clapeyron: ClausiusClapeyronBuilder
                water_buck: WaterBuckBuilder
        """
        return {
            "constant": ConstantBuilder(),
            "antoine": AntoineBuilder(),
            "clausius_clapeyron": ClausiusClapeyronBuilder(),
            "water_buck": WaterBuckBuilder(),
        }
