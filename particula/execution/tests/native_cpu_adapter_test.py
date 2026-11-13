"""Native CPU adapter ownership, dispatch and scientific integration checks."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from particula.dynamics import (
    Coagulation,
    Dilution,
    DilutionStrategy,
    MassCondensation,
)
from particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy import (
    BrownianCoagulationStrategy,
)
from particula.dynamics.tests.native_runnable_test import condensation, state
from particula.equilibria import Equilibria
from particula.equilibria.equilibria_strategies import EquilibriaStrategy
from particula.execution import (
    CPUExecutionAdapter,
    CPUExecutionState,
    MutationScope,
)
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    CPUCoagulationExecutionAdapter,
    CPUCoagulationExecutionState,
    CPUCoagulationResult,
    CPUCoagulationState,
)
from particula.execution.adapters.condensation import (
    CondensationExecutionConfig,
    CPUCondensationExecutionAdapter,
    CPUCondensationExecutionState,
    CPUCondensationState,
)
from particula.execution.tests.condensation_adapter_test import _configuration


@pytest.fixture(autouse=True)
def forbid_facades(forbid_legacy_facades):
    """Apply the removal-safe native execution boundary guard."""


def selected(family, aerosol):
    """Create each selected CPU carrier using real native runnables."""
    if family == "condensation":
        request = CPUCondensationExecutionState(
            CPUCondensationState(
                CondensationExecutionConfig(_configuration()), aerosol
            ),
            4e8,
            2,
            MassCondensation(condensation(aerosol.gas)),
        )
        return CPUCondensationExecutionAdapter(), request
    request = CPUCoagulationExecutionState(
        CPUCoagulationState(BrownianCoagulationConfig(), aerosol),
        1e30,
        1,
        Coagulation(BrownianCoagulationStrategy("particle_resolved")),
    )
    return CPUCoagulationExecutionAdapter(), request


@pytest.mark.parametrize("family", ["condensation", "coagulation"])
def test_native_selected_adapter_dispatches_once_and_conserves(
    family, monkeypatch
):
    """The adapter passes original controls once and preserves native carriers."""
    aerosol = state()
    p, g = aerosol.particles, aerosol.gas
    original_arrays = (p.masses, p.concentration, p.charge, g.concentration)
    before_particle = (
        np.sum(p.masses * p.concentration[..., None], axis=1)
        / p.volume[:, None]
    )
    before_gas = g.concentration.copy()
    adapter, request = selected(family, aerosol)
    original = request.runnable.execute
    calls = []

    def execute(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(request.runnable, "execute", execute)
    result = adapter.execute(request)
    assert calls == [(aerosol, request.time_step, request.sub_steps)]
    assert result.state is request
    assert result.metadata == ()
    assert result.mutation.scopes == frozenset({MutationScope.STATE})
    value = result.backend_result.value
    if family == "coagulation":
        assert isinstance(value, CPUCoagulationResult)
        assert value.aerosol is aerosol
        assert value.state is request.state
        np.testing.assert_array_equal(g.concentration, before_gas)
    else:
        assert value is aerosol
        assert np.any(g.concentration[:, 1:] < before_gas[:, 1:])
    after_particle = (
        np.sum(p.masses * p.concentration[..., None], axis=1)
        / p.volume[:, None]
    )
    np.testing.assert_allclose(
        after_particle + g.concentration[:, [2, 1]],
        before_particle + before_gas[:, [2, 1]],
        rtol=1e-12,
        atol=1e-30,
    )
    assert all(
        left is right
        for left, right in zip(
            original_arrays,
            (p.masses, p.concentration, p.charge, g.concentration),
            strict=True,
        )
    )


@pytest.mark.parametrize("family", ["condensation", "coagulation"])
@pytest.mark.parametrize("behavior", ["raise", "replace"])
def test_native_adapter_preserves_delegate_error_and_identity_contract(
    family, behavior, monkeypatch
):
    """Delegate failures propagate; successful replacement objects reject."""
    aerosol = state()
    adapter, request = selected(family, aerosol)
    failure = RuntimeError("native delegate failed")
    replacement = state()

    def execute(*args):
        if behavior == "raise":
            raise failure
        return replacement

    monkeypatch.setattr(request.runnable, "execute", execute)
    with pytest.raises(
        RuntimeError if behavior == "raise" else ValueError
    ) as caught:
        adapter.execute(request)
    if behavior == "raise":
        assert caught.value is failure


def test_generic_cpu_adapter_runs_native_sequence():
    """Generic CPU execution preserves sequence ordering and original ownership."""
    aerosol = state()
    sequence = Dilution(DilutionStrategy(0.1)) | Dilution(DilutionStrategy(0.2))
    request = CPUExecutionState(aerosol, 2.0, 2)
    before = aerosol.gas.concentration.copy()
    result = CPUExecutionAdapter(sequence).execute(request)
    assert result.state is request
    assert result.backend_result.value is aerosol
    np.testing.assert_allclose(
        aerosol.gas.concentration, before * np.exp(-0.6), rtol=1e-12, atol=0
    )


def test_equilibria_recognizes_native_return_and_preserves_input_mapping():
    """Native recognition retains Equilibria's existing solver/result contract."""
    aerosol = state()
    replacement = state()
    inputs = {
        name: np.array([value])
        for name, value in (
            ("c_star_j_dry", 1.0),
            ("concentration_organic_matter", 0.5),
            ("molar_mass", 100.0),
            ("oxygen2carbon", 0.2),
            ("density", 1200.0),
        )
    }
    aerosol.partitioning_inputs = inputs
    strategy = MagicMock(spec=EquilibriaStrategy)
    strategy.solve.return_value = replacement
    assert Equilibria(strategy).execute(aerosol, 2.0, 2) is replacement
    assert strategy.solve.call_count == 2
    for call in strategy.solve.call_args_list:
        for name, value in inputs.items():
            assert call.kwargs[name] is value
        assert call.kwargs["partition_coefficient_guess"] is None
    strategy.solve.return_value = {"mass_concentrations": np.array([0.2])}
    assert Equilibria(strategy).execute(aerosol, 1.0) is aerosol
    assert (
        aerosol.mass_concentrations
        is strategy.solve.return_value["mass_concentrations"]
    )


@pytest.mark.parametrize("family", ["condensation", "coagulation"])
def test_selected_adapter_delegates_multibox_context_once(family, monkeypatch):
    """Selected adapters forward context without introducing another box loop."""
    from dataclasses import replace

    from particula.dynamics.tests.multibox_runnable_test import batch
    from particula.execution import CPUExecutionContext

    aerosol = batch(state())
    adapter, request = selected(family, aerosol)
    context = CPUExecutionContext(17, ("a", "b"))
    request = replace(request, time_step=0.01, context=context)
    execute = request.runnable.execute
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return execute(*args, **kwargs)

    monkeypatch.setattr(request.runnable, "execute", record)
    result = adapter.execute(request)
    assert result.state is request
    assert calls == [((aerosol, 0.01, request.sub_steps), {"context": context})]
