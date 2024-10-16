"""
This module contains the representation factories for the particles.
"""

from typing import Union
from particula.abc_factory import StrategyFactory
from particula.particles.representation_builders import (
    ParticleMassRepresentationBuilder,
    ParticleRadiusRepresentationBuilder,
    PresetParticleRadiusBuilder,
    ResolvedParticleMassRepresentationBuilder,
    PresetResolvedParticleMassBuilder,
)
from particula.particles.representation import ParticleRepresentation


class ParticleRepresentationFactory(
    StrategyFactory[
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
    """Factory class to create particle representation builders.

    Methods:
        get_builders: Returns the mapping of strategy types to builder
            instances.
        get_strategy: Gets the strategy instance for the specified strategy.
            strategy_type: Type of particle representation strategy to use,
            can be 'radius' (default) or 'mass'.
            parameters: Parameters required for
            the builder

    Returns:
        ParticleRepresentation: An instance of the specified
        ParticleRepresentation.

    Raises:
        ValueError: If an unknown strategy type is provided.
        ValueError: If any required key is missing during check_keys or
            pre_build_check, or if trying to set an invalid parameter.
    """

    def get_builders(self):
        """Returns the mapping of strategy types to builder instances.

        Returns:
            dict[str, Any]: A dictionary with the strategy types as keys and
            the builder instances as values.
            - 'mass': MassParticleRepresentationBuilder
            - 'radius': RadiusParticleRepresentationBuilder
            - 'preset_radius': LimitedRadiusParticleBuilder
            - 'resolved_mass': ResolvedMassParticleRepresentationBuilder
            - 'preset_resolved_mass': PresetResolvedMassParticleBuilder
        """
        return {
            "mass": ParticleMassRepresentationBuilder(),
            "radius": ParticleRadiusRepresentationBuilder(),
            "preset_radius": PresetParticleRadiusBuilder(),
            "resolved_mass": ResolvedParticleMassRepresentationBuilder(),
            "preset_resolved_mass": PresetResolvedParticleMassBuilder(),
        }
