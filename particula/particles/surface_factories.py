""""Surface tension strategies factory.
"""

from typing import Union
from particula.abc_factory import StrategyFactory
from particula.particles.surface_builders import (
    SurfaceStrategyMolarBuilder,
    SurfaceStrategyMassBuilder,
    SurfaceStrategyVolumeBuilder,
)
from particula.particles.surface_strategies import (
    SurfaceStrategyVolume,
    SurfaceStrategyMass,
    SurfaceStrategyMolar,
)


class SurfaceFactory(
    StrategyFactory[
        Union[
            SurfaceStrategyVolumeBuilder,
            SurfaceStrategyMassBuilder,
            SurfaceStrategyMolarBuilder,
        ],
        Union[
            SurfaceStrategyVolume, SurfaceStrategyMass, SurfaceStrategyMolar
        ],
    ]
):
    """Factory class to call and create surface tension strategies.

    Factory class to create surface tension strategy builders for
    calculating surface tension and the Kelvin effect for species in
    particulate phases.

    Methods:
        get_builders(): Returns the mapping of strategy types to builder
        instances.
        get_strategy(strategy_type, parameters): Gets the strategy instance
        for the specified strategy type.
            strategy_type: Type of surface tension strategy to use, can be
            'volume', 'mass', or 'molar'.
            parameters(Dict[str, Any], optional): Parameters required for the
            builder, dependent on the chosen strategy type.
                volume: density, surface_tension
                mass: density, surface_tension
                molar: molar_mass, density, surface_tension

    Returns:
        SurfaceStrategy: An instance of the specified SurfaceStrategy.

    Raises:
        ValueError: If an unknown strategy type is provided.
        ValueError: If any required key is missing during check_keys or
            pre_build_check, or if trying to set an invalid parameter.
    """

    def get_builders(self):
        """
        Returns the mapping of strategy types to builder instances.

        Returns:
            Dict[str, BuilderT]: A dictionary mapping strategy types to
            builder instances.
                volume: SurfaceStrategyVolumeBuilder
                mass: SurfaceStrategyMassBuilder
                molar: SurfaceStrategyMolarBuilder
        """
        return {
            "volume": SurfaceStrategyVolumeBuilder(),
            "mass": SurfaceStrategyMassBuilder(),
            "molar": SurfaceStrategyMolarBuilder(),
        }
