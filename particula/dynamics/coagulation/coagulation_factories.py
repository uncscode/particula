from typing import Dict, Any
from particula.abc_factory import StrategyFactoryABC
from particula.dynamics.coagulation.coagulation_builder import (
    BrownianCoagulationBuilder,
    ChargedCoagulationBuilder,
    TurbulentShearCoagulationBuilder,
    TurbulentDNSCoagulationBuilder,
    CombineCoagulationStrategyBuilder,
)
from particula.dynamics.coagulation.coagulation_strategy import (
    CoagulationStrategyABC,
)


class CoagulationFactory(
    StrategyFactoryABC[
        BrownianCoagulationBuilder
        | ChargedCoagulationBuilder
        | TurbulentShearCoagulationBuilder
        | TurbulentDNSCoagulationBuilder
        | CombineCoagulationStrategyBuilder,
        CoagulationStrategyABC,
    ]
):
    """
    Factory class for creating coagulation strategy instances
    based on a given type string. Supported types include:
        - 'brownian'
        - 'charged'
        - 'turbulent_shear'
        - 'turbulent_dns'
        - 'combine'

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
            "brownian": BrownianCoagulationBuilder(),
            "charged": ChargedCoagulationBuilder(),
            "turbulent_shear": TurbulentShearCoagulationBuilder(),
            "turbulent_dns": TurbulentDNSCoagulationBuilder(),
            "combine": CombineCoagulationStrategyBuilder(),
        }
