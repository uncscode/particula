"""Runnable wrapper for equilibria strategies.

Provides the ``Equilibria`` runnable to compose thermodynamic equilibrium
solvers with other runnables using the shared ``RunnableABC`` interface. The
runnable mirrors the sub-step semantics used by dynamics processes (e.g.,
condensation, coagulation, wall loss) while delegating equilibrium resolution
to a configured ``EquilibriaStrategy``.

Examples:
    Basic usage with a strategy::

        >>> import numpy as np
        >>> from particula.equilibria import (
        ...     Equilibria,
        ...     LiquidVaporPartitioningStrategy,
        ... )
        >>> strategy = LiquidVaporPartitioningStrategy(water_activity=0.75)
        >>> runnable = Equilibria(strategy=strategy)
        >>> aerosol = runnable.execute(aerosol, time_step=1.0)

    Composing with other runnables::

        >>> pipeline = runnable | other_runnable
        >>> aerosol = pipeline.execute(aerosol, time_step=1.0, sub_steps=4)

Note:
    Equilibria solves are typically instantaneous, but the runnable still uses
    sub-step semantics to align with ``RunnableSequence`` composition. A future
    TODO is richer aerosol data mapping once broader data flow is finalized.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Mapping

from particula.aerosol import Aerosol
from particula.equilibria.equilibria_strategies import EquilibriaStrategy
from particula.runnable import RunnableABC

if TYPE_CHECKING:
    from particula.equilibria.equilibria_strategies import (
        EquilibriumResult,
        PhaseConcentrations,
    )


class Equilibria(RunnableABC):
    """Execute an equilibria strategy within the runnable interface.

    Args:
        strategy: Concrete ``EquilibriaStrategy`` used to solve equilibrium.
    """

    def __init__(self, strategy: EquilibriaStrategy) -> None:
        """Initialize with the provided equilibria strategy.

        Args:
            strategy: Concrete ``EquilibriaStrategy`` used during execution.
        """
        self.strategy = strategy

    def rate(self, aerosol: Aerosol) -> Any:  # noqa: ARG002
        """Return the strategy identifier for non-rate processes."""
        return self.strategy.get_name()

    def execute(
        self, aerosol: Aerosol, time_step: float, sub_steps: int = 1
    ) -> Aerosol:
        """Run equilibria over the provided time step.

        Args:
            aerosol: Aerosol state passed to the strategy and updated in place.
            time_step: Simulation interval in seconds.
            sub_steps: Number of sub-divisions of ``time_step``. Must be > 0.

        Returns:
            The updated aerosol after equilibrium application.

        Raises:
            ValueError: If ``sub_steps`` is not positive.
            AttributeError: When required partitioning inputs are missing.
            TypeError: If the strategy returns ``None``.
        """
        if sub_steps <= 0:
            raise ValueError("sub_steps must be positive")

        partitioning_inputs = self._extract_partitioning_inputs(aerosol)
        dt = time_step / sub_steps
        solve_params = inspect.signature(self.strategy.solve).parameters

        for _ in range(sub_steps):
            solve_kwargs = {
                **partitioning_inputs,
                "partition_coefficient_guess": None,
            }
            if "time_step" in solve_params:
                solve_kwargs["time_step"] = dt
            result = self.strategy.solve(**solve_kwargs)
            aerosol = self._apply_equilibrium_result(aerosol, result)

        return aerosol

    def _extract_partitioning_inputs(self, aerosol: Aerosol) -> dict[str, Any]:
        """Extract required inputs for partitioning strategies.

        Looks for a ``partitioning_inputs`` mapping on the aerosol first. If
        absent, falls back to attributes of the aerosol itself. Raises with a
        clear message when a required field is unavailable to keep failures
        explicit during testing.
        """
        required_keys = [
            "c_star_j_dry",
            "concentration_organic_matter",
            "molar_mass",
            "oxygen2carbon",
            "density",
        ]

        if hasattr(aerosol, "partitioning_inputs"):
            candidate = aerosol.partitioning_inputs
            if isinstance(candidate, Mapping):
                missing = [k for k in required_keys if k not in candidate]
                if missing:
                    raise AttributeError(
                        "Missing required partitioning_inputs keys: "
                        + ", ".join(missing)
                    )
                return {k: candidate[k] for k in required_keys}

        inputs: dict[str, Any] = {}
        missing_attrs: list[str] = []
        for key in required_keys:
            if hasattr(aerosol, key):
                inputs[key] = getattr(aerosol, key)
            else:
                missing_attrs.append(key)

        if missing_attrs:
            raise AttributeError(
                "Aerosol is missing required attributes: "
                + ", ".join(missing_attrs)
            )

        return inputs

    def _apply_equilibrium_result(
        self, aerosol: Aerosol, result: Any
    ) -> Aerosol:
        """Apply strategy results to the aerosol or attach for downstream use.

        Accepts either an updated ``Aerosol`` instance, a mapping/struct with
        recognizable concentration fields, or raises when ``None`` is returned.
        """
        if result is None:
            raise TypeError("Equilibria strategy returned None")

        # Local import to prevent import cycles during typing checks.
        from particula.equilibria.equilibria_strategies import (
            EquilibriumResult,
            PhaseConcentrations,
        )

        if isinstance(result, Aerosol):
            return result

        if isinstance(result, EquilibriumResult):
            return self._attach_equilibrium_result_dataclass(aerosol, result)

        if isinstance(result, Mapping):
            return self._apply_mapping_result(aerosol, result)

        if hasattr(result, "phase_concentrations") or hasattr(
            result, "mass_concentrations"
        ):
            return self._apply_attribute_result(aerosol, result)

        if isinstance(result, PhaseConcentrations):
            return self._attach_phase_concentrations(aerosol, result)

        return aerosol

    @staticmethod
    def _attach_equilibrium_result_dataclass(
        aerosol: Aerosol, result: EquilibriumResult
    ) -> Aerosol:
        aerosol.equilibria_result = result  # type: ignore[attr-defined]
        return aerosol

    @staticmethod
    def _apply_mapping_result(
        aerosol: Aerosol, result: Mapping[str, Any]
    ) -> Aerosol:
        if "phase_concentrations" in result:
            aerosol.phase_concentrations = result["phase_concentrations"]  # type: ignore[attr-defined]
        if "mass_concentrations" in result:
            aerosol.mass_concentrations = result["mass_concentrations"]  # type: ignore[attr-defined]
        aerosol.equilibria_result = result  # type: ignore[attr-defined]
        return aerosol

    @staticmethod
    def _apply_attribute_result(aerosol: Aerosol, result: Any) -> Aerosol:
        aerosol.equilibria_result = result  # type: ignore[attr-defined]
        if hasattr(result, "phase_concentrations"):
            aerosol.phase_concentrations = result.phase_concentrations  # type: ignore[attr-defined]
        if hasattr(result, "mass_concentrations"):
            aerosol.mass_concentrations = result.mass_concentrations  # type: ignore[attr-defined]
        return aerosol

    @staticmethod
    def _attach_phase_concentrations(
        aerosol: Aerosol, result: PhaseConcentrations
    ) -> Aerosol:
        aerosol.phase_concentrations = result  # type: ignore[attr-defined]
        aerosol.equilibria_result = result  # type: ignore[attr-defined]
        return aerosol
