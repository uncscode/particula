"""Surface tension strategies factory."""

from typing import Union

from particula.abc_factory import StrategyFactoryABC
from particula.particles.surface_builders import (
    SurfaceStrategyMassBuilder,
    SurfaceStrategyMolarBuilder,
    SurfaceStrategyVolumeBuilder,
)
from particula.particles.surface_strategies import (
    SurfaceStrategyMass,
    SurfaceStrategyMolar,
    SurfaceStrategyVolume,
)


class SurfaceFactory(
    StrategyFactoryABC[
        Union[
            SurfaceStrategyVolumeBuilder,
            SurfaceStrategyMassBuilder,
            SurfaceStrategyMolarBuilder,
        ],
        Union[SurfaceStrategyVolume, SurfaceStrategyMass, SurfaceStrategyMolar],
    ]
):
    """Factory for creating surface tension strategy builders.

    Creates builder instances for volume-, mass-, or molar-based
    surface tension strategies. These strategies calculate surface
    tension and the Kelvin effect for species in particulate phases.

    Methods:
    - get_builders : Return a mapping of strategy types to builder
        instances.
    - get_strategy : Return a strategy instance for the specified type
        ('volume', 'mass', or 'molar').

    Returns:
        - SurfaceStrategy : The instance of the requested surface strategy.

    Raises:
        - ValueError : If an unknown strategy type is provided or if
          required parameters are missing during check_keys/pre_build_check.
    """

    def get_builders(self):
        """Return a mapping of strategy types to builder instances.

        Returns:
            - Keys are 'volume', 'mass', or 'molar', each
              mapped to the corresponding builder class.

        Examples:
            ```py title="SurfaceFactory Example"
            import particula as par
            factory = par.particles.SurfaceFactory()
            builders = factory.get_builders()
            volume_builder = builders["volume"]
            mass_builder = builders["mass"]
            molar_builder = builders["molar"]
            ```
        """
        return {
            "volume": SurfaceStrategyVolumeBuilder(),
            "mass": SurfaceStrategyMassBuilder(),
            "molar": SurfaceStrategyMolarBuilder(),
        }
