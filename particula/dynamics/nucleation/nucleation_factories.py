"""Fresh-builder factory for bounded P4 nucleation strategies."""

from particula.abc_factory import StrategyFactoryABC
from particula.dynamics.nucleation.nucleation_builders import (
    ActivationNucleationBuilder,
    KineticNucleationBuilder,
)
from particula.dynamics.nucleation.nucleation_strategies import (
    ActivationNucleationStrategy,
    KineticNucleationStrategy,
)


class NucleationFactory(
    StrategyFactoryABC[
        ActivationNucleationBuilder | KineticNucleationBuilder,
        ActivationNucleationStrategy | KineticNucleationStrategy,
    ]
):
    """Build activation or kinetic P4 potential-rate strategies by name."""

    def get_builders(
        self,
    ) -> dict[str, ActivationNucleationBuilder | KineticNucleationBuilder]:
        """Return new builders for the two supported identifiers."""
        return {
            "activation": ActivationNucleationBuilder(),
            "kinetic": KineticNucleationBuilder(),
        }

    def get_strategy(
        self,
        strategy_type: str,
        parameters: dict[str, object] | None = None,
    ) -> ActivationNucleationStrategy | KineticNucleationStrategy:
        """Build a validated strategy from a copied strict-schema payload."""
        if not isinstance(strategy_type, str):
            raise ValueError("strategy_type must be a string")
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a dict")
        builder = self.get_builders().get(strategy_type.lower())
        if builder is None:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        builder.set_parameters(dict(parameters))
        return builder.build()
