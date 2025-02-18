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

    def get_strategy(
        self,
        strategy_type: str,
        parameters: Dict[str, Any],
    ) -> CoagulationStrategyABC:
        builder = self.get_builders().get(strategy_type)
        if builder is None:
            raise ValueError(
                f"Unknown coagulation strategy type: {strategy_type}"
            )

        builder.check_keys(parameters)
        for param_name, param_value in parameters.items():
            if hasattr(builder, f"set_{param_name}"):
                getattr(builder, f"set_{param_name}")(param_value)
            else:
                raise ValueError(
                    f"Invalid parameter '{param_name}' for this builder."
                )

        return builder.build()
