"""Tests for the Equilibria runnable."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from particula.equilibria.equilibria import Equilibria
from particula.equilibria.equilibria_strategies import (
    EquilibriaStrategy,
    EquilibriumResult,
    LiquidVaporPartitioningStrategy,
    PhaseConcentrations,
)
from particula.runnable import RunnableABC, RunnableSequence


class _AerosolStub:
    """Simple stub exposing required fields for testing."""

    def __init__(self, inputs: dict[str, Any] | None = None) -> None:
        if inputs is None:
            inputs = {
                "c_star_j_dry": np.array([1.0, 2.0]),
                "concentration_organic_matter": np.array([0.5, 0.6]),
                "molar_mass": np.array([100.0, 200.0]),
                "oxygen2carbon": np.array([0.2, 0.4]),
                "density": np.array([1200.0, 1000.0]),
            }
        self.partitioning_inputs = inputs


def _equilibrium_result() -> EquilibriumResult:
    phase = PhaseConcentrations(
        species_concentrations=np.array([1.0]),
        water_concentration=0.0,
        total_concentration=1.0,
    )
    return EquilibriumResult(
        alpha_phase=phase,
        beta_phase=None,
        partition_coefficients=np.array([1.0]),
        water_content=(0.0, 0.0),
        error=0.0,
    )


def test_init_stores_strategy():
    """Ensure the initializer keeps the provided strategy."""
    strategy = MagicMock(spec=EquilibriaStrategy)
    runnable = Equilibria(strategy)
    assert runnable.strategy is strategy


def test_rate_returns_strategy_name():
    """Rate reports the configured strategy identifier."""
    strategy = MagicMock(spec=EquilibriaStrategy)
    strategy.get_name.return_value = "ReturnAerosol"
    runnable = Equilibria(strategy)
    assert runnable.rate(_AerosolStub()) == "ReturnAerosol"  # type: ignore[arg-type]


def test_execute_returns_aerosol_instance():
    """Strategy results that are aerosols are returned directly."""
    aerosol = _AerosolStub()
    strategy = MagicMock(spec=EquilibriaStrategy)
    strategy.solve.return_value = aerosol
    runnable = Equilibria(strategy)
    result = runnable.execute(aerosol, time_step=1.0)  # type: ignore[arg-type]
    assert result is aerosol


def test_execute_applies_mapping_result():
    """Mappings with concentration fields are attached to the aerosol."""
    aerosol = _AerosolStub()
    strategy = MagicMock(spec=EquilibriaStrategy)
    strategy.solve.return_value = {
        "phase_concentrations": "phase",
        "mass_concentrations": "mass",
    }
    runnable = Equilibria(strategy)
    result = runnable.execute(aerosol, time_step=1.0)  # type: ignore[arg-type]
    assert result.equilibria_result["phase_concentrations"] == "phase"  # type: ignore[attr-defined]
    assert result.equilibria_result["mass_concentrations"] == "mass"  # type: ignore[attr-defined]


def test_execute_raises_on_none_result():
    """None results raise to avoid silent failures."""
    aerosol = _AerosolStub()
    strategy = MagicMock(spec=EquilibriaStrategy)
    strategy.solve.return_value = None
    runnable = Equilibria(strategy)
    with pytest.raises(TypeError):
        runnable.execute(aerosol, time_step=1.0)  # type: ignore[arg-type]


class _CountingStrategy(EquilibriaStrategy):
    """Strategy counting calls and capturing dt."""

    def __init__(self):
        self.calls = 0
        self.dts: list[float] = []

    def solve(
        self,
        c_star_j_dry: Any,
        concentration_organic_matter: Any,
        molar_mass: Any,
        oxygen2carbon: Any,
        density: Any,
        partition_coefficient_guess: Any = None,
        time_step: Any = None,
    ):  # type: ignore[override]
        self.calls += 1
        self.dts.append(float(time_step))
        return _equilibrium_result()


def test_sub_steps_calls_strategy_each_time_and_passes_dt():
    """Verify each sub-step calls the strategy with the delta time."""
    aerosol = _AerosolStub()
    strategy = _CountingStrategy()
    runnable = Equilibria(strategy)
    runnable.execute(aerosol, time_step=3.0, sub_steps=3)  # type: ignore[arg-type]
    assert strategy.calls == 3
    assert strategy.dts == [1.0, 1.0, 1.0]


def test_invalid_sub_steps_raises():
    """Non-positive sub_steps are rejected."""
    aerosol = _AerosolStub()
    strategy = MagicMock(spec=EquilibriaStrategy)
    runnable = Equilibria(strategy)
    with pytest.raises(ValueError):
        runnable.execute(aerosol, time_step=1.0, sub_steps=0)  # type: ignore[arg-type]


def test_pipe_operator_returns_runnable_sequence():
    """Composition via | yields a RunnableSequence containing both processes."""
    strategy = MagicMock(spec=EquilibriaStrategy)
    runnable = Equilibria(strategy)
    other = MagicMock(spec=RunnableABC)
    sequence = runnable | other
    assert isinstance(sequence, RunnableSequence)
    assert len(sequence.processes) == 2


def test_missing_partitioning_input_raises_attribute_error():
    """AttributeError surfaces when required inputs are missing."""
    strategy = MagicMock(spec=EquilibriaStrategy)
    runnable = Equilibria(strategy)
    aerosol = _AerosolStub(inputs={"c_star_j_dry": np.array([1.0])})
    with pytest.raises(AttributeError):
        runnable.execute(aerosol, time_step=1.0)  # type: ignore[arg-type]


def test_liquid_vapor_partitioning_strategy_runs():
    """Ensure the real LiquidVaporPartitioningStrategy coherently runs."""
    strategy = LiquidVaporPartitioningStrategy(water_activity=0.5)
    runnable = Equilibria(strategy)
    aerosol = _AerosolStub()
    result = runnable.execute(aerosol, time_step=1.0)  # type: ignore[arg-type]
    assert result is aerosol


def test_phase_concentrations_result_is_attached():
    """PhaseConcentrations results are stored on the aerosol."""
    aerosol = _AerosolStub()
    phase = PhaseConcentrations(
        species_concentrations=np.array([1.0]),
        water_concentration=0.0,
        total_concentration=1.0,
    )

    def _solve(**kwargs: Any):
        return phase

    strategy = MagicMock(spec=EquilibriaStrategy)
    strategy.solve.side_effect = _solve
    runnable = Equilibria(strategy)
    result = runnable.execute(aerosol, time_step=1.0)  # type: ignore[arg-type]
    assert result.phase_concentrations is phase  # type: ignore[attr-defined]


def test_equilibrium_result_is_attached_and_attribute_inputs_used():
    """Attribute-backed aerosols are accepted and store results."""

    class _AttrAerosol:
        def __init__(self):
            self.c_star_j_dry = np.array([3.0])
            self.concentration_organic_matter = np.array([4.0])
            self.molar_mass = np.array([5.0])
            self.oxygen2carbon = np.array([6.0])
            self.density = np.array([7.0])

    aerosol = _AttrAerosol()
    strategy = _CountingStrategy()
    runnable = Equilibria(strategy)

    result = runnable.execute(aerosol, time_step=2.0, sub_steps=2)  # type: ignore[arg-type]

    assert strategy.calls == 2
    assert strategy.dts == [1.0, 1.0]
    assert result is aerosol
    assert result.equilibria_result == _equilibrium_result()  # type: ignore[attr-defined]


def test_object_with_phase_concentrations_attribute_is_attached():
    """Objects exposing concentration attributes get persisted."""
    aerosol = _AerosolStub()

    class _ResultWithAttrs:
        def __init__(self):
            self.phase_concentrations = "phase"
            self.mass_concentrations = "mass"

    strategy = MagicMock(spec=EquilibriaStrategy)
    strategy.solve.return_value = _ResultWithAttrs()
    runnable = Equilibria(strategy)

    result = runnable.execute(aerosol, time_step=1.0)  # type: ignore[arg-type]

    assert result is aerosol
    assert result.equilibria_result.phase_concentrations == "phase"  # type: ignore[attr-defined]
    assert result.equilibria_result.mass_concentrations == "mass"  # type: ignore[attr-defined]
