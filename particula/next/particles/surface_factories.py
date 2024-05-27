""""Surface tension strategies factory.

If this factory patterns holds, we can make and ABC factory class
or function that will consolidate the code and test. We can revisit this
idea in the future.
"""

from typing import Optional, Dict, Any
import logging
from particula.next.particles.surface_strategies import SurfaceStrategy
from particula.next.particles.surface_builders import (
    SurfaceStrategyMolarBuilder, SurfaceStrategyMassBuilder,
    SurfaceStrategyVolumeBuilder
)

logger = logging.getLogger("particula")


def surface_strategy_factory(
    strategy_type: str,
    parameters: Optional[Dict[str, Any]] = None
) -> SurfaceStrategy:
    """
    Factory function to create surface tension strategy builders for
    calculating surface tension and the Kelvin effect for species in
    particulate phases.

    Args:
    -----
    - strategy_type (str): Type of surface tension strategy to use,
        'volume', 'mass', or 'molar'.
    - parameters (Dict[str, Any], optional): Parameters required for the
    builder, dependent on the chosen strategy type.
        - volume: density, surface_tension
        - mass: density, surface_tension
        - molar: molar_mass, density, surface_tension

    Returns:
    --------
    - SurfaceStrategy: An instance of the specified SurfaceStrategy.

    Raises:
    -------
    - ValueError: If an unknown strategy type is provided.
    - ValueError: If any required key is missing during check_keys or
        pre_build_check, or if trying to set an invalid parameter.
    """
    builder_map = {
        "volume": SurfaceStrategyVolumeBuilder(),
        "mass": SurfaceStrategyMassBuilder(),
        "molar": SurfaceStrategyMolarBuilder()
    }
    builder = builder_map.get(strategy_type.lower())
    if builder is None:
        message = f"Unknown strategy type: {strategy_type}"
        logger.error(message)
        raise ValueError(message)

    # Set the parameters for the builder
    if parameters and hasattr(builder, 'set_parameters'):
        builder.set_parameters(parameters)

    return builder.build()  # build the surface strategy
