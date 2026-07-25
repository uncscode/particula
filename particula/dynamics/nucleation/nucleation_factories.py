"""Create bounded P4 nucleation strategies with fresh strict builders.

The factory dispatches only activation and kinetic potential-rate strategies.
It does not construct source-selection metadata or expose concrete P2/P3
``particle_source`` finalization and mutation helpers.
"""

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
    """Build activation or kinetic P4 potential-rate strategies by name.

    Each request gets a new builder, so failed or successful requests cannot
    retain configuration into subsequent calls. Identifiers are case-insensitive
    but are limited to ``"activation"`` and ``"kinetic"``.
    """

    def get_builders(
        self,
    ) -> dict[str, ActivationNucleationBuilder | KineticNucleationBuilder]:
        """Return fresh builders for the two supported identifiers.

        Returns:
            New activation and kinetic builders under their canonical lowercase
            identifiers.
        """
        return {
            "activation": ActivationNucleationBuilder(),
            "kinetic": KineticNucleationBuilder(),
        }

    def get_strategy(
        self,
        strategy_type: str,
        parameters: dict[str, object] | None = None,
    ) -> ActivationNucleationStrategy | KineticNucleationStrategy:
        """Build a validated strategy from a copied strict-schema payload.

        Args:
            strategy_type: Case-insensitive ``"activation"`` or ``"kinetic"``
                identifier.
            parameters: Required dictionary accepted by the selected builder.

        Returns:
            Newly built immutable activation or kinetic potential-rate strategy.

        Raises:
            ValueError: If the identifier or payload is invalid, unsupported, or
                fails the selected builder's schema validation.
        """
        if not isinstance(strategy_type, str):
            raise ValueError("strategy_type must be a string")
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a dict")
        builder = self.get_builders().get(strategy_type.lower())
        if builder is None:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        builder.set_parameters(dict(parameters))
        return builder.build()
