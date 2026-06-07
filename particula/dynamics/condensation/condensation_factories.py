"""Factory for building condensation strategies."""

from typing import Dict, Union

from particula.abc_factory import StrategyFactoryABC

from .condensation_builder import (
    CondensationIsothermalBuilder,
    CondensationIsothermalStaggeredBuilder,
    CondensationLatentHeatBuilder,
)
from .condensation_strategies import CondensationStrategy


class CondensationFactory(
    StrategyFactoryABC[
        Union[
            CondensationLatentHeatBuilder,
            CondensationIsothermalBuilder,
            CondensationIsothermalStaggeredBuilder,
        ],
        CondensationStrategy,
    ]
):
    """Factory class for condensation strategies.

    Supports strategy types:
        - "isothermal": Standard isothermal condensation.
        - "isothermal_staggered": Staggered isothermal condensation with
          batch stepping for stability.
        - "latent_heat": Condensation with latent-heat coupling.
    """

    def get_builders(
        self,
    ) -> Dict[
        str,
        Union[
            CondensationLatentHeatBuilder,
            CondensationIsothermalBuilder,
            CondensationIsothermalStaggeredBuilder,
        ],
    ]:
        """Return the mapping of strategy types to builder instances.

        Returns:
            Dictionary mapping condensation strategy names to builders.
        """
        return {
            "isothermal": CondensationIsothermalBuilder(),
            "isothermal_staggered": CondensationIsothermalStaggeredBuilder(),
            "latent_heat": CondensationLatentHeatBuilder(),
        }
