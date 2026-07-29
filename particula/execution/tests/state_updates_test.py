"""Tests for direct-only resident environment and gas state updates."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest


def _state_updates() -> Any:
    """Import the Warp-dependent update boundary only for device tests."""
    pytest.importorskip("warp")
    import particula.execution.state_updates as state_updates

    return state_updates


def _session(boxes: int = 1, species: int = 1) -> Any:
    """Build a canonical resident session through the shared test helper."""
    pytest.importorskip("warp")
    from particula.execution.tests.gpu_resources_test import _session as build

    return build(boxes, 2, species)


def _request_node(name: str) -> Any:
    """Resolve and return one exact canonical update node."""
    from particula.execution import CapabilityRequirements
    from particula.execution.process_graph import (
        InvalidatedState,
        NodeKind,
        ProcessNode,
        ResourceRequirement,
        TimestepPlan,
        resolve_timestep_plan,
    )

    environment = ProcessNode(
        "environment_update",
        NodeKind.ENVIRONMENT_UPDATE,
        None,
        CapabilityRequirements(frozenset()),
        frozenset({ResourceRequirement.ENVIRONMENT}),
        frozenset(
            {InvalidatedState.VAPOR_PRESSURE, InvalidatedState.SATURATION_RATIO}
        ),
    )
    gas = ProcessNode(
        "gas_update",
        NodeKind.GAS_UPDATE,
        None,
        CapabilityRequirements(frozenset()),
        frozenset({ResourceRequirement.GAS}),
        frozenset({InvalidatedState.SATURATION_RATIO}),
    )
    graph = resolve_timestep_plan(TimestepPlan((environment, gas), ()))
    return graph, next(node for node in graph.nodes if node.node_id == name)


@pytest.mark.warp
def test_environment_update_copies_only_prescribed_fields_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment updates retain identities and copy temperature then pressure."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session(boxes=2, species=2)
    registry = GPUResourceRegistry(session)
    graph, node = _request_node("environment_update")
    temperature = wp.array([280.0, 285.0], dtype=wp.float64, device="cpu")
    pressure = wp.array([100000.0, 99000.0], dtype=wp.float64, device="cpu")
    request = updates.ResidentEnvironmentUpdateRequest(
        session, registry, graph, node, temperature, pressure
    )
    assert request.session is session
    assert request.registry is registry
    assert request.graph is graph
    assert request.node is node
    assert request.temperature is temperature
    assert request.pressure is pressure
    with pytest.raises(FrozenInstanceError):
        request.temperature = pressure  # type: ignore[misc]
    original_copy = updates.wp.copy
    calls: list[tuple[object, object]] = []

    def tracked_copy(
        destination: object, source: object, *args: object, **kwargs: object
    ) -> Any:
        calls.append((destination, source))
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(updates.wp, "copy", tracked_copy)
    environment = updates.ResidentStateUpdateExecutor().execute(request)

    assert environment is session.environment
    assert environment.temperature is cast(Any, session.environment).temperature
    npt.assert_array_equal(environment.temperature.numpy(), [280.0, 285.0])
    npt.assert_array_equal(environment.pressure.numpy(), [100000.0, 99000.0])
    assert calls == [
        (environment.temperature, temperature),
        (environment.pressure, pressure),
    ]


@pytest.mark.warp
def test_gas_update_preserves_non_target_primaries_and_accepts_zero_values() -> (
    None
):
    """Gas updates replace concentration only and permit finite zero values."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session(boxes=2, species=2)
    registry = GPUResourceRegistry(session)
    graph, node = _request_node("gas_update")
    gas = cast(Any, session.gas)
    vapor_pressure = gas.vapor_pressure.numpy().copy()
    volume = cast(Any, session.particles).volume
    concentration = wp.array(
        [[0.0, 2.0], [3.0, 4.0]], dtype=wp.float64, device="cpu"
    )
    request = updates.ResidentGasUpdateRequest(
        session, registry, graph, node, concentration
    )

    returned = updates.ResidentStateUpdateExecutor().execute(request)

    assert returned is gas
    assert cast(Any, session.particles).volume is volume
    npt.assert_array_equal(gas.concentration.numpy(), concentration.numpy())
    npt.assert_array_equal(gas.vapor_pressure.numpy(), vapor_pressure)


@pytest.mark.warp
def test_invalid_environment_payload_rejects_before_any_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both environment fields scan before the prescribed two-copy commit."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session()
    registry = GPUResourceRegistry(session)
    graph, node = _request_node("environment_update")
    request = updates.ResidentEnvironmentUpdateRequest(
        session,
        registry,
        graph,
        node,
        wp.array([300.0], dtype=wp.float64, device="cpu"),
        wp.array([np.nan], dtype=wp.float64, device="cpu"),
    )
    calls = 0

    def no_copy(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(updates.wp, "copy", no_copy)
    with pytest.raises(ValueError, match="pressure values"):
        updates.ResidentStateUpdateExecutor().execute(request)
    assert calls == 0


@pytest.mark.warp
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", 0.0, "temperature values"),
        ("temperature", -1.0, "temperature values"),
        ("temperature", np.inf, "temperature values"),
        ("pressure", 0.0, "pressure values"),
        ("pressure", -np.inf, "pressure values"),
    ],
)
def test_environment_invalid_scalar_values_leave_targets_unchanged(
    field: str,
    value: float,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment scalar validation rejects before either resident write."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session()
    registry = GPUResourceRegistry(session)
    graph, node = _request_node("environment_update")
    temperature = wp.array([298.15], dtype=wp.float64, device="cpu")
    pressure = wp.array([101325.0], dtype=wp.float64, device="cpu")
    if field == "temperature":
        temperature = wp.array([value], dtype=wp.float64, device="cpu")
    else:
        pressure = wp.array([value], dtype=wp.float64, device="cpu")
    target = cast(Any, session.environment)
    before = (target.temperature.numpy().copy(), target.pressure.numpy().copy())
    monkeypatch.setattr(
        updates.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("invalid payload must not copy"),
    )

    request = updates.ResidentEnvironmentUpdateRequest(
        session, registry, graph, node, temperature, pressure
    )
    with pytest.raises(ValueError, match=message):
        updates.ResidentStateUpdateExecutor().execute(request)

    npt.assert_array_equal(target.temperature.numpy(), before[0])
    npt.assert_array_equal(target.pressure.numpy(), before[1])


@pytest.mark.warp
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_gas_invalid_scalar_values_reject_before_copy(
    value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gas concentration must be finite and nonnegative before committing."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session()
    registry = GPUResourceRegistry(session)
    graph, node = _request_node("gas_update")
    target = cast(Any, session.gas).concentration
    before = target.numpy().copy()
    request = updates.ResidentGasUpdateRequest(
        session,
        registry,
        graph,
        node,
        wp.array([[value]], dtype=wp.float64, device="cpu"),
    )
    monkeypatch.setattr(
        updates.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("invalid payload must not copy"),
    )

    with pytest.raises(ValueError, match="concentration values"):
        updates.ResidentStateUpdateExecutor().execute(request)

    npt.assert_array_equal(target.numpy(), before)


@pytest.mark.warp
@pytest.mark.parametrize(
    ("request_type", "payload", "message"),
    [
        ("temperature", [298.15, np.nan, 0.0], "temperature values"),
        ("pressure", [101325.0, np.inf, -1.0], "pressure values"),
        (
            "concentration",
            [[1.0, np.nan], [-1.0, np.inf], [0.0, 2.0]],
            "concentration values",
        ),
    ],
)
def test_multilane_invalid_payloads_reject_before_copy(
    request_type: str,
    payload: list[float] | list[list[float]],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent invalid scan lanes reject without committing a writer."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session(boxes=3, species=2)
    registry = GPUResourceRegistry(session)
    if request_type == "concentration":
        graph, node = _request_node("gas_update")
        target = cast(Any, session.gas).concentration
        request = updates.ResidentGasUpdateRequest(
            session,
            registry,
            graph,
            node,
            wp.array(payload, dtype=wp.float64, device="cpu"),
        )
    else:
        graph, node = _request_node("environment_update")
        target = cast(Any, session.environment)
        temperature = wp.full(3, 298.15, dtype=wp.float64, device="cpu")
        pressure = wp.full(3, 101325.0, dtype=wp.float64, device="cpu")
        if request_type == "temperature":
            temperature = wp.array(payload, dtype=wp.float64, device="cpu")
        else:
            pressure = wp.array(payload, dtype=wp.float64, device="cpu")
        request = updates.ResidentEnvironmentUpdateRequest(
            session, registry, graph, node, temperature, pressure
        )
    if request_type == "concentration":
        before_concentration = target.numpy().copy()
    else:
        before_temperature = target.temperature.numpy().copy()
        before_pressure = target.pressure.numpy().copy()
    monkeypatch.setattr(
        updates.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("invalid payload must not copy"),
    )

    with pytest.raises(ValueError, match=message):
        updates.ResidentStateUpdateExecutor().execute(request)

    if request_type == "concentration":
        npt.assert_array_equal(target.numpy(), before_concentration)
    else:
        npt.assert_array_equal(target.temperature.numpy(), before_temperature)
        npt.assert_array_equal(target.pressure.numpy(), before_pressure)


@pytest.mark.warp
def test_update_rejects_manually_constructed_resolved_graph_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A field-identical graph must originate from the plan resolver."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry
    from particula.execution.process_graph import ResolvedProcessGraph

    session = _session()
    registry = GPUResourceRegistry(session)
    graph, node = _request_node("gas_update")
    forged_node = type(node)(
        node.node_id,
        node.kind,
        node.process,
        node.requirements,
        node.resources,
        node.invalidates,
    )
    forged_graph = ResolvedProcessGraph((forged_node,), ())
    target = cast(Any, session.gas).concentration
    before = target.numpy().copy()
    request = updates.ResidentGasUpdateRequest(
        session,
        registry,
        forged_graph,
        forged_node,
        wp.array([[1.0]], dtype=wp.float64, device="cpu"),
    )
    monkeypatch.setattr(
        updates.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("forged graph must not copy"),
    )

    with pytest.raises(ValueError, match="produced by plan resolution"):
        updates.ResidentStateUpdateExecutor().execute(request)

    npt.assert_array_equal(target.numpy(), before)


@pytest.mark.warp
def test_nonempty_null_pointer_rejects_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonempty Warp input with a null pointer cannot authorize a copy."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session()
    registry = GPUResourceRegistry(session)
    graph, node = _request_node("gas_update")
    target = cast(Any, session.gas).concentration
    before = target.numpy().copy()
    null_input = wp.array(
        ptr=0,
        capacity=8,
        dtype=wp.float64,
        shape=(1, 1),
        strides=(8, 8),
        device="cpu",
        copy=False,
    )
    request = updates.ResidentGasUpdateRequest(
        session, registry, graph, node, null_input
    )
    monkeypatch.setattr(
        updates.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("null input must not copy"),
    )

    with pytest.raises(ValueError, match="valid pointer"):
        updates.ResidentStateUpdateExecutor().execute(request)

    npt.assert_array_equal(target.numpy(), before)


@pytest.mark.warp
def test_update_rejects_primary_alias_and_noncanonical_node_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Updates reject protected storage and an incompatible canonical role."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session()
    registry = GPUResourceRegistry(session)
    environment_graph, environment_node = _request_node("environment_update")
    gas_graph, gas_node = _request_node("gas_update")
    alias_request = updates.ResidentEnvironmentUpdateRequest(
        session,
        registry,
        environment_graph,
        environment_node,
        cast(Any, session.environment).temperature,
        wp.array([101325.0], dtype=wp.float64, device="cpu"),
    )
    wrong_role_request = updates.ResidentGasUpdateRequest(
        session,
        registry,
        gas_graph,
        gas_node,
        wp.array([[1.0]], dtype=wp.float64, device="cpu"),
    )
    object.__setattr__(gas_node, "kind", environment_node.kind)
    monkeypatch.setattr(
        updates.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("invalid request must not copy"),
    )

    with pytest.raises(ValueError, match="must not alias"):
        updates.ResidentStateUpdateExecutor().execute(alias_request)
    with pytest.raises(ValueError, match="canonical update role"):
        updates.ResidentStateUpdateExecutor().execute(wrong_role_request)


@pytest.mark.warp
def test_environment_update_rejects_input_alias_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment update inputs must retain independent storage ownership."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session()
    registry = GPUResourceRegistry(session)
    graph, node = _request_node("environment_update")
    shared = wp.array([300.0], dtype=wp.float64, device="cpu")
    request = updates.ResidentEnvironmentUpdateRequest(
        session, registry, graph, node, shared, shared
    )
    monkeypatch.setattr(
        updates.wp,
        "copy",
        lambda *_args, **_kwargs: pytest.fail("aliased inputs must not copy"),
    )

    with pytest.raises(ValueError, match="must not alias each other"):
        updates.ResidentStateUpdateExecutor().execute(request)


@pytest.mark.warp
@pytest.mark.parametrize("boxes,species", [(0, 1), (1, 0)])
def test_empty_update_schemas_are_write_free_noops(
    boxes: int, species: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical empty environment and gas schemas preserve all identities."""
    updates = _state_updates()
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    session = _session(boxes=boxes, species=species)
    registry = GPUResourceRegistry(session)
    environment_graph, environment_node = _request_node("environment_update")
    gas_graph, gas_node = _request_node("gas_update")
    calls = 0
    original_copy = updates.wp.copy

    def tracked_copy(*args: object, **kwargs: object) -> Any:
        nonlocal calls
        calls += 1
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(updates.wp, "copy", tracked_copy)
    environment = cast(Any, session.environment)
    gas = cast(Any, session.gas)
    environment_request = updates.ResidentEnvironmentUpdateRequest(
        session,
        registry,
        environment_graph,
        environment_node,
        wp.ones((boxes,), dtype=wp.float64, device="cpu"),
        wp.ones((boxes,), dtype=wp.float64, device="cpu"),
    )
    gas_request = updates.ResidentGasUpdateRequest(
        session,
        registry,
        gas_graph,
        gas_node,
        wp.zeros((boxes, species), dtype=wp.float64, device="cpu"),
    )

    assert (
        updates.ResidentStateUpdateExecutor().execute(environment_request)
        is environment
    )
    assert updates.ResidentStateUpdateExecutor().execute(gas_request) is gas
    assert calls == 0 if boxes == 0 else 2


def test_request_carriers_are_frozen_and_execution_import_is_isolated() -> None:
    """Concrete imports retain package import isolation and carrier immutability."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import sys
import particula.execution as execution
assert execution.__all__ == [
    'Backend',
    'Device',
    'Process',
    'Capability',
    'CapabilityRequirements',
    'CapabilityDeclaration',
    'CapabilityMatrix',
    'ExecutionRequest',
    'ExecutionAdapter',
    'ExecutionContext',
]
assert 'particula.execution.state_updates' not in sys.modules
assert 'particula.execution.gpu_resources' not in sys.modules
assert 'particula.gpu' not in sys.modules
assert 'warp' not in sys.modules
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-Werror", "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    updates = _state_updates()
    with pytest.raises(TypeError, match="exact ResidentSession"):
        updates.ResidentGasUpdateRequest(
            object(), object(), object(), object(), object()
        )
