"""Factory Classes for selecting the distribution strategy."""

from typing import Union

from particula.abc_factory import StrategyFactoryABC
from particula.particles.distribution_builders import (
    MassBasedMovingBinBuilder,
    ParticleResolvedSpeciatedMassBuilder,
    RadiiBasedMovingBinBuilder,
    SpeciatedMassMovingBinBuilder,
)
from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    ParticleResolvedSpeciatedMass,
    RadiiBasedMovingBin,
    SpeciatedMassMovingBin,
)


class DistributionFactory(
    StrategyFactoryABC[
        Union[
            MassBasedMovingBinBuilder,
            RadiiBasedMovingBinBuilder,
            SpeciatedMassMovingBinBuilder,
            ParticleResolvedSpeciatedMassBuilder,
        ],
        Union[
            MassBasedMovingBin,
            RadiiBasedMovingBin,
            SpeciatedMassMovingBin,
            ParticleResolvedSpeciatedMass,
        ],
    ]
):
    """Factory class to create distribution strategies from builders.

    This factory is used to obtain particle distribution strategies
    based on the specified representation type (mass-based, radius-based,
    speciated, or particle-resolved).

    Methods:
        - get_builders : Return a mapping of strategy types to builder
            instances.
        - get_strategy : Return a strategy instance for a given strategy type.

    Returns:
        - DistributionStrategy : An instance configured for the chosen
          distribution representation.

    Raises:
        - ValueError : If an unknown strategy type is provided or if
          required parameters are missing or invalid.

    Examples:
        ```py title="DistributionFactory Example"
        import particula as par
        factory = par.particles.DistributionFactory()
        strategy = factory.get_strategy("mass_based_moving_bin")
        # strategy -> MassBasedMovingBin()
        ```
    """

    def get_builders(self):
        """Return a mapping of strategy types to builder instances.

        Returns:
            - A dictionary where each key is a string identifying the strategy
                type, and each value is the corresponding builder object.

        Examples:
            ```py title="get_builders Example"
            import particula as par
            factory = par.particles.DistributionFactory()
            builder_map = factory.get_builders()
            # builder_map["mass_based_moving_bin"] -> MassBasedMovingBinBuilder
            ```
        """
        return {
            "mass_based_moving_bin": MassBasedMovingBinBuilder(),
            "radii_based_moving_bin": RadiiBasedMovingBinBuilder(),
            "speciated_mass_moving_bin": SpeciatedMassMovingBinBuilder(),
            "particle_resolved_speciated_mass": (
                ParticleResolvedSpeciatedMassBuilder()
            ),
        }
