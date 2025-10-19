"""Coagulation Factory Module."""

from typing import Any, Dict, Union

from particula.abc_factory import StrategyFactoryABC
from particula.dynamics.coagulation.coagulation_builder import (
    brownian_coagulation_builder,
    charged_coagulation_builder,
    combine_coagulation_strategy_builder,
    turbulent_dns_coagulation_builder,
    turbulent_shear_coagulation_builder,
)
from particula.dynamics.coagulation.coagulation_strategy import (
    coagulation_strategy_abc,
)


class CoagulationFactory(
    StrategyFactoryABC[
        Union[
            brownian_coagulation_builder.BrownianCoagulationBuilder,
            charged_coagulation_builder.ChargedCoagulationBuilder,
            turbulent_shear_coagulation_builder
            .TurbulentShearCoagulationBuilder,
            turbulent_dns_coagulation_builder
            .TurbulentDNSCoagulationBuilder,
            combine_coagulation_strategy_builder
            .CombineCoagulationStrategyBuilder,
        ],
        coagulation_strategy_abc.CoagulationStrategyABC,
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
        return {
            "brownian": (
                brownian_coagulation_builder.BrownianCoagulationBuilder()
            ),
            "charged": (
                charged_coagulation_builder.ChargedCoagulationBuilder()
            ),
            "turbulent_shear": (
                turbulent_shear_coagulation_builder
                .TurbulentShearCoagulationBuilder()
            ),
            "turbulent_dns": (
                turbulent_dns_coagulation_builder
                .TurbulentDNSCoagulationBuilder()
            ),
            "combine": (
                combine_coagulation_strategy_builder
                .CombineCoagulationStrategyBuilder()
            ),
        }
