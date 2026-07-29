"""Tests for concrete-only resident direct-process adapters."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

import particula.execution.process_adapters as process_adapters
from particula.execution.gpu_resources import (
    GPUResourceRegistry,
    NucleationResources,
    WallLossResources,
)
from particula.execution.gpu_session import ResidentSession
from particula.execution.process_adapters import (
    ResidentDilutionAdapter,
    ResidentDilutionRequest,
    ResidentNucleationAdapter,
    ResidentNucleationRequest,
    ResidentWallLossAdapter,
    ResidentWallLossRequest,
)


def _registry(session: ResidentSession) -> GPUResourceRegistry:
    """Construct a Warp registry only for a Warp-dependent test."""
    pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry

    return GPUResourceRegistry(session)


def _session() -> ResidentSession:
    """Build a Warp session only for a Warp-dependent test."""
    pytest.importorskip("warp")
    from particula.execution.tests.gpu_resources_test import _session as build

    return build()


@pytest.mark.warp
def test_request_carriers_retain_exact_dependencies_and_opaque_inputs() -> None:
    """Test valid carriers retain references and reject invalid dependencies."""
    session = _session()
    registry = _registry(session)
    wall_loss = registry.acquire_wall_loss()
    nucleation = registry.acquire_nucleation()
    opaque: Any = object()

    dilution = ResidentDilutionRequest(session, registry, opaque, opaque)
    wall = ResidentWallLossRequest(session, registry, wall_loss, opaque, opaque)
    nucleation_request = ResidentNucleationRequest(
        session, registry, nucleation, opaque, opaque, opaque
    )

    assert dilution.session is session
    assert dilution.coefficient is opaque
    assert wall.resources is wall_loss
    assert wall.config is opaque
    assert nucleation_request.resources is nucleation
    assert nucleation_request.exhaustion_controls is opaque
    assert dilution != ResidentDilutionRequest(
        session, registry, opaque, opaque
    )
    with pytest.raises(FrozenInstanceError):
        dilution.time_step = 1  # type: ignore[misc]
    with pytest.raises(TypeError, match="exact ResidentSession"):
        ResidentDilutionRequest(cast(Any, object()), registry, opaque, opaque)
    with pytest.raises(TypeError, match="exact GPUResourceRegistry"):
        ResidentDilutionRequest(session, cast(Any, object()), opaque, opaque)
    with pytest.raises(TypeError, match="exact WallLossResources"):
        ResidentWallLossRequest(
            session, registry, cast(Any, object()), opaque, opaque
        )
    with pytest.raises(TypeError, match="exact NucleationResources"):
        ResidentNucleationRequest(
            session,
            registry,
            cast(Any, object()),
            opaque,
            opaque,
            opaque,
        )


@pytest.mark.warp
def test_request_carriers_enforce_exact_types_in_documented_order() -> None:
    """Test subclasses reject while opaque kernel inputs remain uninspected."""
    session = _session()
    registry = _registry(session)
    nucleation = registry.acquire_nucleation()
    opaque: Any = object()

    class SessionSubclass(ResidentSession):
        """Provide an inexact session type without invoking its constructor."""

    class RegistrySubclass(GPUResourceRegistry):
        """Provide an inexact registry type without invoking its constructor."""

    class WallLossSubclass(WallLossResources):
        """Provide an inexact established wall-loss view type."""

    class NucleationSubclass(NucleationResources):
        """Provide an inexact established nucleation view type."""

    inexact_session = object.__new__(SessionSubclass)
    inexact_registry = object.__new__(RegistrySubclass)
    inexact_wall_loss = object.__new__(WallLossSubclass)
    inexact_nucleation = object.__new__(NucleationSubclass)

    with pytest.raises(TypeError, match="exact ResidentSession"):
        ResidentWallLossRequest(
            cast(Any, inexact_session),
            cast(Any, object()),
            cast(Any, object()),
            opaque,
            opaque,
        )
    with pytest.raises(TypeError, match="exact GPUResourceRegistry"):
        ResidentWallLossRequest(
            session,
            cast(Any, inexact_registry),
            cast(Any, object()),
            opaque,
            opaque,
        )
    with pytest.raises(TypeError, match="exact WallLossResources"):
        ResidentWallLossRequest(
            session,
            registry,
            cast(Any, inexact_wall_loss),
            opaque,
            opaque,
        )
    with pytest.raises(TypeError, match="exact NucleationResources"):
        ResidentNucleationRequest(
            session,
            registry,
            cast(Any, inexact_nucleation),
            opaque,
            opaque,
            opaque,
        )

    request = ResidentNucleationRequest(
        session, registry, nucleation, opaque, opaque, opaque
    )
    assert request.config is opaque
    assert request.time_step is opaque


@pytest.mark.warp
def test_dilution_adapter_delegates_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dilution dispatch preserves containers and opaque inputs by identity."""
    session = _session()
    registry = _registry(session)
    coefficient: Any = object()
    time_step: Any = object()
    request = ResidentDilutionRequest(session, registry, coefficient, time_step)
    result = object()
    calls: list[tuple[object, ...]] = []

    def step(*args: object) -> object:
        calls.append(args)
        return result

    monkeypatch.setattr(
        process_adapters, "_get_dilution_step_gpu", lambda: step
    )

    assert ResidentDilutionAdapter().execute(request) is result
    assert calls == [(session.particles, session.gas, coefficient, time_step)]


@pytest.mark.warp
def test_wall_loss_adapter_delegates_published_rng_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test wall-loss dispatch preserves its established persistent RNG sidecar."""
    session = _session()
    registry = _registry(session)
    resources = registry.acquire_wall_loss()
    config: Any = object()
    request = ResidentWallLossRequest(
        session,
        registry,
        resources,
        config,
        0,
        rng_seed=41,
        initialize_rng=True,
    )
    result = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def step(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(
        process_adapters, "_get_wall_loss_step_gpu", lambda: step
    )

    assert ResidentWallLossAdapter().execute(request) is result
    args, kwargs = calls[0]
    assert args == (session.particles, None, None, 0)
    assert kwargs["config"] is config
    assert kwargs["rng_states"] is resources.rng_states
    assert kwargs["rng_seed"] == 41
    assert kwargs["initialize_rng"] is True
    assert kwargs["environment"] is session.environment


@pytest.mark.warp
def test_nucleation_adapter_delegates_all_published_sidecars_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test nucleation dispatch preserves each established sidecar identity."""
    session = _session()
    registry = _registry(session)
    resources = registry.acquire_nucleation()
    config: Any = object()
    controls: Any = object()
    request = ResidentNucleationRequest(
        session, registry, resources, config, object(), controls
    )
    result = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def step(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(
        process_adapters, "_get_nucleation_step_gpu", lambda: step
    )

    assert ResidentNucleationAdapter().execute(request) is result
    args, kwargs = calls[0]
    assert args[:3] == (session.particles, session.gas, config)
    assert args[3] is request.time_step
    assert kwargs["scratch"] is resources.scratch
    assert kwargs["finalized_demand"] is resources.finalized_demand
    assert kwargs["diagnostics"] is resources.diagnostics
    assert kwargs["exhaustion_buffers"] is resources.exhaustion
    assert kwargs["exhaustion_controls"] is controls
    assert kwargs["temperature"] is None
    assert kwargs["saturation"] is None
    assert kwargs["environment"] is session.environment


@pytest.mark.warp
@pytest.mark.parametrize(
    ("adapter", "candidate_request"),
    [
        (ResidentDilutionAdapter(), object()),
        (ResidentWallLossAdapter(), object()),
        (ResidentNucleationAdapter(), object()),
    ],
)
def test_adapters_reject_wrong_request_before_kernel_resolution(
    adapter: Any, candidate_request: object
) -> None:
    """Test adapters reject wrong request types before their direct import."""
    with pytest.raises(TypeError, match="request must be an exact"):
        adapter.execute(candidate_request)


@pytest.mark.warp
def test_resource_view_validators_reject_unacquired_and_replaced_views() -> (
    None
):
    """Test established-view seams neither acquire nor accept replacements."""
    session = _session()
    registry = _registry(session)
    other_registry = _registry(_session())
    other_wall_loss = other_registry.acquire_wall_loss()
    other_nucleation = other_registry.acquire_nucleation()

    with pytest.raises(ValueError, match="have not been acquired"):
        registry.validate_wall_loss_resources(session, other_wall_loss)
    with pytest.raises(ValueError, match="have not been acquired"):
        registry.validate_nucleation_resources(session, other_nucleation)
    wall_loss = registry.acquire_wall_loss()
    nucleation = registry.acquire_nucleation()
    with pytest.raises(ValueError, match="published wall_loss"):
        registry.validate_wall_loss_resources(session, other_wall_loss)
    with pytest.raises(ValueError, match="published nucleation"):
        registry.validate_nucleation_resources(session, other_nucleation)
    assert registry._views["wall_loss"] is wall_loss
    assert registry._views["nucleation"] is nucleation


@pytest.mark.warp
def test_direct_kernel_error_escapes_without_adapter_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test an adapter calls once and leaves direct writer failure untouched."""
    session = _session()
    registry = _registry(session)
    request = ResidentDilutionRequest(session, registry, 0, 0)
    failure = RuntimeError("direct writer failed")
    calls = 0

    def step(*_args: object) -> object:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(
        process_adapters, "_get_dilution_step_gpu", lambda: step
    )
    with pytest.raises(RuntimeError) as caught:
        ResidentDilutionAdapter().execute(request)

    assert caught.value is failure
    assert calls == 1


@pytest.mark.warp
@pytest.mark.parametrize("family", ["wall_loss", "nucleation"])
def test_view_adapter_direct_error_escapes_without_recovery(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test view adapters call a failing direct boundary exactly once."""
    session = _session()
    registry = _registry(session)
    opaque: Any = object()
    failure = RuntimeError(f"{family} direct writer failed")
    calls = 0
    primaries = (session.particles, session.gas, session.environment)
    request: Any
    adapter: Any
    resources: Any

    def step(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise failure

    if family == "wall_loss":
        resources = registry.acquire_wall_loss()
        request = ResidentWallLossRequest(
            session, registry, resources, opaque, opaque
        )
        adapter = ResidentWallLossAdapter()
        target = "_get_wall_loss_step_gpu"
    else:
        resources = registry.acquire_nucleation()
        request = ResidentNucleationRequest(
            session, registry, resources, opaque, opaque, opaque
        )
        adapter = ResidentNucleationAdapter()
        target = "_get_nucleation_step_gpu"
    bindings = registry._bindings.copy()
    views = registry._views.copy()
    capacities = registry._capacities.copy()
    monkeypatch.setattr(process_adapters, target, lambda: step)

    with pytest.raises(RuntimeError) as caught:
        adapter.execute(request)

    assert caught.value is failure
    assert calls == 1
    assert registry._bindings == bindings
    assert registry._views == views
    assert registry._capacities == capacities
    assert session.particles is primaries[0]
    assert session.gas is primaries[1]
    assert session.environment is primaries[2]


@pytest.mark.warp
@pytest.mark.parametrize("family", ["wall_loss", "nucleation"])
def test_established_view_rejection_resolves_no_kernel_and_mutates_nothing(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test unavailable published views reject before resolver or writer calls."""
    session = _session()
    registry = _registry(session)
    foreign_registry = _registry(_session())
    opaque: Any = object()
    resolver_calls = 0
    request: Any
    adapter: Any
    resources: Any

    def resolver() -> object:
        """Fail if adapter preflight attempts to import a direct boundary."""
        nonlocal resolver_calls
        resolver_calls += 1
        return object()

    if family == "wall_loss":
        resources = foreign_registry.acquire_wall_loss()
        request = ResidentWallLossRequest(
            session, registry, resources, opaque, opaque
        )
        adapter = ResidentWallLossAdapter()
        target = "_get_wall_loss_step_gpu"
    else:
        resources = foreign_registry.acquire_nucleation()
        request = ResidentNucleationRequest(
            session, registry, resources, opaque, opaque, opaque
        )
        adapter = ResidentNucleationAdapter()
        target = "_get_nucleation_step_gpu"
    monkeypatch.setattr(process_adapters, target, resolver)
    bindings = registry._bindings.copy()
    views = registry._views.copy()
    capacities = registry._capacities.copy()
    primaries = (session.particles, session.gas, session.environment)

    with pytest.raises(ValueError, match="have not been acquired"):
        adapter.execute(request)

    assert resolver_calls == 0
    assert registry._bindings == bindings
    assert registry._views == views
    assert registry._capacities == capacities
    assert (session.particles, session.gas, session.environment) == primaries


@pytest.mark.warp
def test_dilution_binding_rejection_resolves_no_kernel_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a request bound to another registry rejects before resolution."""
    session = _session()
    registry = _registry(session)
    other_registry = _registry(_session())
    request = ResidentDilutionRequest(session, other_registry, 0, 0)
    calls = 0

    def resolver() -> object:
        """Record an unexpected lazy resolver invocation."""
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(process_adapters, "_get_dilution_step_gpu", resolver)
    with pytest.raises(ValueError, match="pinned ResidentSession"):
        ResidentDilutionAdapter().execute(request)

    assert calls == 0
    assert registry._bindings == {}
    assert other_registry._bindings == {}


def test_process_adapter_import_and_opaque_carrier_construction_are_isolated() -> (
    None
):
    """Test importing adapters alone does not eagerly import Warp or GPU code."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import sys
import particula.execution.process_adapters as adapters
assert 'warp' not in sys.modules
assert not any(name == 'particula.gpu' or name.startswith('particula.gpu.') for name in sys.modules)
assert 'particula.execution.gpu_resources' not in sys.modules
assert adapters.ResidentDilutionAdapter is not None
try:
    adapters.ResidentDilutionRequest(object(), object(), object(), object())
except TypeError as error:
    assert str(error) == 'session must be an exact ResidentSession.'
else:
    raise AssertionError('invalid opaque carrier data must reject')
assert 'particula.execution.gpu_resources' not in sys.modules
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
