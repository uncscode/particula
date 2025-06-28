"""Particle Representation Factories.

Provides classes for constructing ParticleRepresentation objects
based on various distribution, activity, and surface strategies.
"""

from typing import Union

from particula.abc_factory import StrategyFactoryABC
from particula.particles.representation import ParticleRepresentation
from particula.particles.representation_builders import (
    ParticleMassRepresentationBuilder,
    ParticleRadiusRepresentationBuilder,
    PresetParticleRadiusBuilder,
    PresetResolvedParticleMassBuilder,
    ResolvedParticleMassRepresentationBuilder,
)


class ParticleRepresentationFactory(
    StrategyFactoryABC[
        Union[
            ParticleMassRepresentationBuilder,
            ParticleRadiusRepresentationBuilder,
            PresetParticleRadiusBuilder,
            ResolvedParticleMassRepresentationBuilder,
            PresetResolvedParticleMassBuilder,
        ],
        ParticleRepresentation,
    ]
):
    """Factory for creating particle representation builders.

    Methods:
    - get_builders : Return a dictionary of strategy builder instances.
    - get_strategy : Obtain a ParticleRepresentation from a chosen builder.

    Examples:
        ```py title="Factory Usage Example"
        import particula as par
        factory = par.particles.ParticleRepresentationFactory()
        rep = factory.get_strategy("mass")
        # rep -> ParticleRepresentation with mass-based distribution
        ```

    References:
        - Builder Pattern,
          [Wikipedia](https://en.wikipedia.org/wiki/Builder_pattern).

    """

    def get_builders(self):
        """Return a mapping of strategy types to builder instances.

        Returns:
            - A dictionary where each key is a strategy type
              ("mass", "radius", etc.) and each value is the
              corresponding builder instance.

        Examples:
            ```py title="get_builders usage"
            import particula as par
            factory = par.particles.ParticleRepresentationFactory()
            builders = factory.get_builders()
            mass_builder = builders["mass"]
            # mass_builder -> ParticleMassRepresentationBuilder()
            ```
        """
        return {
            "mass": ParticleMassRepresentationBuilder(),
            "radius": ParticleRadiusRepresentationBuilder(),
            "preset_radius": PresetParticleRadiusBuilder(),
            "resolved_mass": ResolvedParticleMassRepresentationBuilder(),
            "preset_resolved_mass": PresetResolvedParticleMassBuilder(),
        }
