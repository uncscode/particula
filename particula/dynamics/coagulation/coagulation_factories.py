"""Coagulation Factory Module."""

from typing import Any, Dict, Union

from particula.abc_factory import StrategyFactoryABC

from .coagulation_builder.brownian_coagulation_builder import (
    BrownianCoagulationBuilder,
)
from .coagulation_builder.charged_coagulation_builder import (
    ChargedCoagulationBuilder,
)
from .coagulation_builder.combine_coagulation_strategy_builder import (
    CombineCoagulationStrategyBuilder,
)
from .coagulation_builder.turbulent_dns_coagulation_builder import (
    TurbulentDNSCoagulationBuilder,
)
from .coagulation_builder.turbulent_shear_coagulation_builder import (
    TurbulentShearCoagulationBuilder,
)
from .coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)


class CoagulationFactory(
    StrategyFactoryABC[
        Union[
            BrownianCoagulationBuilder,
            ChargedCoagulationBuilder,
            TurbulentShearCoagulationBuilder,
            TurbulentDNSCoagulationBuilder,
            CombineCoagulationStrategyBuilder,
        ],
        CoagulationStrategyABC,
    ]
):
    """Factory class for creating coagulation strategy instances
    based on a given type string. Supported types include:
        - 'brownian'
        - 'charged'
        - 'turbulent_shear'
        - 'turbulent_dns'
        - 'combine'.

    Methods:
    - get_builders() : Returns the mapping of strategy types to builder
        instances.
    - get_strategy(strategy_type, parameters): Gets the strategy instance
        for the specified strategy type.
        - strategy_type: Type of coagulation strategy to use, can be
            'brownian', 'charged', 'turbulent_shear', 'turbulent_dns', or
            'combine'.
    """

    def get_builders(self) -> Dict[str, Any]:
        """Return the mapping of strategy types to builder instances.

        Returns:
            Dictionary mapping strategy names to their corresponding builders.
        """
        return {
            "brownian": BrownianCoagulationBuilder(),
            "charged": ChargedCoagulationBuilder(),
            "turbulent_shear": TurbulentShearCoagulationBuilder(),
            "turbulent_dns": TurbulentDNSCoagulationBuilder(),
            "combine": CombineCoagulationStrategyBuilder(),
        }
