"""Factory Classes for selecting the distribution strategy.
"""

from typing import Union
from particula.next.abc_factory import StrategyFactory
from particula.next.particles.distribution_strategies import (
    MassBasedMovingBin, RadiiBasedMovingBin, SpeciatedMassMovingBin,
)
from particula.next.particles.distribution_builders import (
    MassBasedMovingBinBuilder, RadiiBasedMovingBinBuilder,
    SpeciatedMassMovingBinBuilder,
)


class DistributionFactory(
    StrategyFactory[
        Union[
            MassBasedMovingBinBuilder,
            RadiiBasedMovingBinBuilder,
            SpeciatedMassMovingBinBuilder
        ],
        Union[
            MassBasedMovingBin,
            RadiiBasedMovingBin,
            SpeciatedMassMovingBin
        ]]
):
    """
    Factory class to create distribution strategy builders for
    calculating particle distributions based on the specified
    representation type.

    Methods
    -------
    - get_builders(): Returns the mapping of strategy types to builder
    instances.
    - get_strategy(strategy_type, parameters): Gets the strategy instance
    for the specified strategy type.
        - strategy_type: Type of distribution strategy to use, can be
        'mass_based_moving_bin', 'radii_based_moving_bin', or
        'speciated_mass_moving_bin'.
        - parameters(Dict[str, Any], optional): Parameters required for the
        builder, dependent on the chosen strategy type.
            - mass_based_moving_bin: None
            - radii_based_moving_bin: None
            - speciated_mass_moving_bin: None

    Returns:
    --------
    - DistributionStrategy: An instance of the specified DistributionStrategy.

    Raises:
    -------
    - ValueError: If an unknown strategy type is provided.
    - ValueError: If any required key is missing during check_keys or
        pre_build_check, or if trying to set an invalid parameter.
    """

    def get_builders(self):
        """
        Returns the mapping of strategy types to builder instances.

        Returns:
        --------
        - Dict[str, BuilderABC]: Mapping of strategy types to builder
        instances.
            - 'mass_based_moving_bin': MassBasedMovingBinBuilder
            - 'radii_based_moving_bin': RadiiBasedMovingBinBuilder
            - 'speciated_mass_moving_bin': SpeciatedMassMovingBinBuilder
        """
        return {
            "mass_based_moving_bin": MassBasedMovingBinBuilder(),
            "radii_based_moving_bin": RadiiBasedMovingBinBuilder(),
            "speciated_mass_moving_bin": SpeciatedMassMovingBinBuilder()
        }
