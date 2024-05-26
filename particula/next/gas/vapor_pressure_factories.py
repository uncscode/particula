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
    - parameters (dict): A dictionary containing the necessary parameters for
        the strategy. If no parameters are needed, this can be left as None.

    Returns:
    -------
    - vapor_pressure_strategy (VaporPressureStrategy): A concrete
      implementation of the VaporPressureStrategy built using the appropriate
      builder.
    """
    # Assumes all necessary parameters are passed, builder will raise error
    # if parameters are missing.
    # update to a map, like in activity_factories.py
    if strategy.lower() == "constant":
        builder = vapor_pressure_builders.ConstantBuilder()
        builder.set_parameters(parameters=parameters)  # type: ignore
        return builder.build()
    if strategy.lower() == "antoine":
        builder = vapor_pressure_builders.AntoineBuilder()
        builder.set_parameters(parameters=parameters)  # type: ignore
        return builder.build()
    if strategy.lower() == "clausius_clapeyron":
        builder = vapor_pressure_builders.ClausiusClapeyronBuilder()
        builder.set_parameters(parameters=parameters)  # type: ignore
        return builder.build()
    if strategy.lower() == "water_buck":
        builder = vapor_pressure_builders.WaterBuckBuilder()
        return builder.build()  # Assumes no parameters are needed
    raise ValueError(f"Unknown vapor pressure strategy: {strategy}")
