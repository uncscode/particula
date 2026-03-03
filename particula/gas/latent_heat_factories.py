"""Factories for latent heat strategies.

This module exposes :class:`LatentHeatFactory`, which builds constant,
linear, and power-law latent heat strategies via their respective builders.
"""

from particula.abc_factory import StrategyFactoryABC
from particula.gas.latent_heat_builders import (
    ConstantLatentHeatBuilder,
    LinearLatentHeatBuilder,
    PowerLawLatentHeatBuilder,
)
from particula.gas.latent_heat_strategies import (
    ConstantLatentHeat,
    LinearLatentHeat,
    PowerLawLatentHeat,
)

LatentHeatBuilderType = (
    ConstantLatentHeatBuilder
    | LinearLatentHeatBuilder
    | PowerLawLatentHeatBuilder
)

LatentHeatStrategyType = (
    ConstantLatentHeat | LinearLatentHeat | PowerLawLatentHeat
)


class LatentHeatFactory(
    StrategyFactoryABC[LatentHeatBuilderType, LatentHeatStrategyType]
):
    """Factory for latent heat strategies.

    The factory builds constant, linear, or power-law latent heat strategies
    by delegating to the corresponding builder class.

    Examples:
        >>> from particula.gas import LatentHeatFactory
        >>> factory = LatentHeatFactory()
        >>> factory.get_strategy(
        ...     "constant",
        ...     {
        ...         "latent_heat_ref": 2.26e6,
        ...         "latent_heat_ref_units": "J/kg",
        ...     },
        ... ).latent_heat(300.0)
        2260000.0
    """

    def get_builders(self) -> dict[str, LatentHeatBuilderType]:
        """Return latent heat builders keyed by strategy name.

        Returns:
            Builder instances keyed by strategy type.

        Examples:
            >>> from particula.gas import LatentHeatFactory
            >>> builders = LatentHeatFactory().get_builders()
            >>> isinstance(builders["constant"], ConstantLatentHeatBuilder)
            True
        """
        return {
            "constant": ConstantLatentHeatBuilder(),
            "linear": LinearLatentHeatBuilder(),
            "power_law": PowerLawLatentHeatBuilder(),
        }
