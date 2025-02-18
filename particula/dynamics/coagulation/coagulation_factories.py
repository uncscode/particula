from typing import Dict, Any
from particula.abc_factory import StrategyFactory
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


"""
Factory class for creating coagulation strategy instances
based on a given type string. Supported types include:
  - 'brownian'
  - 'charged'
  - 'turbulent_shear'
  - 'turbulent_dns'
  - 'combine'
"""

class CoagulationFactory(
    StrategyFactory[
        BrownianCoagulationBuilder
        | ChargedCoagulationBuilder
        | TurbulentShearCoagulationBuilder
        | TurbulentDNSCoagulationBuilder
        | CombineCoagulationStrategyBuilder,
        CoagulationStrategyABC,
    ]
):

    def get_builders(self) -> Dict[str, Any]:
        return {
            "brownian": BrownianCoagulationBuilder(),
            "charged": ChargedCoagulationBuilder(),
            "turbulent_shear": TurbulentShearCoagulationBuilder(),
            "turbulent_dns": TurbulentDNSCoagulationBuilder(),
            "combine": CombineCoagulationStrategyBuilder(),
        }
