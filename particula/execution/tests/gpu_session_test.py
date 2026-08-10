"""Tests for concrete-only GPU-resident session P1 carriers."""

import os
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from particula.execution import Backend, Device
from particula.execution.gpu_session import (
    ResidentDimensions,
    ResidentLifecycle,
    ResidentMetadata,
    ResidentSession,
    ResidentStepGuard,
    ResidentStepToken,
    ResidentStreamMetadata,
    _handle_failed_resident_operation,
    _ResidentOperationOutcome,
    _validate_contiguous_array,
    setup_resident_session,
)
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData


def _metadata(species: int = 1) -> ResidentMetadata:
    """Create valid CPU-owned metadata for a Warp CPU fixture."""
    return ResidentMetadata(
        Device(Backend.WARP, "cpu"),
        tuple(f"species_{index}" for index in range(species)),
    )


@pytest.mark.parametrize(
    ("pointer", "capacity", "match"),
    (
        (0, 8, "valid pointer"),
        (4, 8, "8-byte aligned"),
        (8, 7, "sufficient integral storage capacity"),
        (8, 9, "sufficient integral storage capacity"),
    ),
)
def test_primary_array_metadata_requires_valid_pointer_alignment_and_capacity(
    pointer: int, capacity: int, match: str
) -> None:
    """Test primary metadata rejects unsafe nonempty native backing storage."""
    array = type(
        "PrimaryArray",
        (),
        {"strides": (8,), "ptr": pointer, "capacity": capacity},
    )()

    with pytest.raises(ValueError, match=match):
        _validate_contiguous_array("primary", array, (1,), 8)


def _warp_resources(
    boxes: int = 1, particles_count: int = 2, species: int = 1
) -> tuple[Any, Any, Any]:
    """Construct a tiny schema-valid Warp CPU fixture lazily."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    particles = WarpParticleData()
    particles.masses = wp.ones(
        (boxes, particles_count, species), dtype=wp.float64, device="cpu"
    )
    particles.concentration = wp.ones(
        (boxes, particles_count), dtype=wp.float64, device="cpu"
    )
    particles.charge = wp.zeros(
        (boxes, particles_count), dtype=wp.float64, device="cpu"
    )
    particles.density = wp.ones(species, dtype=wp.float64, device="cpu")
    particles.volume = wp.ones(boxes, dtype=wp.float64, device="cpu")
    gas = WarpGasData()
    gas.molar_mass = wp.ones(species, dtype=wp.float64, device="cpu")
    gas.concentration = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    gas.vapor_pressure = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    gas.partitioning = wp.ones((boxes, species), dtype=wp.int32, device="cpu")
    environment = WarpEnvironmentData()
    environment.temperature = wp.ones(boxes, dtype=wp.float64, device="cpu")
    environment.pressure = wp.ones(boxes, dtype=wp.float64, device="cpu")
    environment.saturation_ratio = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    return particles, gas, environment


def _guard() -> ResidentStepGuard:
    """Construct a valid P4 guard with its exact pinned registry."""
    from particula.execution.gpu_resources import GPUResourceRegistry

    particles, gas, environment = _warp_resources()
    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 2, 1),
        _metadata(),
        ResidentLifecycle.ACTIVE,
    )
    return ResidentStepGuard(session, GPUResourceRegistry(session))


class _MetadataArray:
    """Expose only array metadata and reject every payload operation."""

    def __init__(self, dtype: object, shape: object, device: object) -> None:
        self.dtype = dtype
        self.shape = shape
        self.device = device

    def numpy(self) -> object:
        """Fail if P1 attempts a host readback."""
        raise AssertionError("P1 must not read resident array payloads.")


def _primary_fields() -> tuple[tuple[str, str, tuple[int, ...], str], ...]:
    """Return the primary array schema for the canonical one-box fixture."""
    return (
        ("particles", "masses", (1, 2, 1), "float64"),
        ("particles", "concentration", (1, 2), "float64"),
        ("particles", "charge", (1, 2), "float64"),
        ("particles", "density", (1,), "float64"),
        ("particles", "volume", (1,), "float64"),
        ("gas", "molar_mass", (1,), "float64"),
        ("gas", "concentration", (1, 1), "float64"),
        ("gas", "vapor_pressure", (1, 1), "float64"),
        ("gas", "partitioning", (1, 1), "int32"),
        ("environment", "temperature", (1,), "float64"),
        ("environment", "pressure", (1,), "float64"),
        ("environment", "saturation_ratio", (1, 1), "float64"),
    )


def _snapshot_resources(
    particles: Any, gas: Any, environment: Any
) -> tuple[tuple[int, object, object, object, tuple[object, ...]], ...]:
    """Capture primary-array identity, metadata, and tiny fixture values."""
    snapshot = []
    for carrier, field, _, _ in _primary_fields():
        value = getattr(
            {"particles": particles, "gas": gas, "environment": environment}[
                carrier
            ],
            field,
        )
        snapshot.append(
            (
                id(value),
                value.dtype,
                value.shape,
                value.device,
                tuple(value.numpy().flat),
            )
        )
    return tuple(snapshot)


def test_carriers_validate_and_remain_immutable() -> None:
    """Test CPU-only dimensions, metadata, and lifecycle vocabulary."""
    dimensions = ResidentDimensions(1, 0, 0)
    metadata = ResidentMetadata(Device(Backend.WARP, "cpu"), ())

    assert dimensions == ResidentDimensions(1, 0, 0)
    assert metadata.gas_names is metadata.gas_names
    assert set(ResidentLifecycle) == {
        ResidentLifecycle.ACTIVE,
        ResidentLifecycle.FAULTED,
        ResidentLifecycle.FINALIZED,
        ResidentLifecycle.CLOSED,
    }
    with pytest.raises(FrozenInstanceError):
        dimensions.n_boxes = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        metadata.gas_names = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value", "name", "positive", "error", "message"),
    [
        (True, "n_boxes", True, TypeError, "integral, not bool"),
        (1.0, "n_boxes", True, TypeError, "integral, not bool"),
        (-1, "n_particles", False, ValueError, "nonnegative"),
        (-1, "n_species", False, ValueError, "nonnegative"),
    ],
)
def test_dimensions_reject_invalid_values(
    value: object,
    name: str,
    positive: bool,
    error: type[Exception],
    message: str,
) -> None:
    """Test dimension validation has stable condition-specific messages."""
    values: dict[str, object] = {
        "n_boxes": 1,
        "n_particles": 0,
        "n_species": 0,
    }
    values[name] = value
    with pytest.raises(error, match=message):
        ResidentDimensions(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("device", "names", "error", "message"),
    [
        (object(), (), TypeError, "exact Device"),
        (Device(Backend.CPU, "cpu"), (), ValueError, "Backend.WARP"),
        (Device(Backend.WARP, "cpu"), [], TypeError, "exact tuple"),
    ],
)
def test_metadata_rejects_invalid_values(
    device: Device, names: object, error: type[Exception], message: str
) -> None:
    """Test metadata does not normalize invalid device or name declarations."""
    with pytest.raises(error, match=message):
        ResidentMetadata(device, names)  # type: ignore[arg-type]


def test_metadata_preserves_valid_names_by_identity() -> None:
    """Test metadata retains valid tuple and string names by identity."""
    names = ("ok", "still_ok")

    metadata = ResidentMetadata(Device(Backend.WARP, "cpu"), names)

    assert metadata.gas_names is names


def test_resident_stream_metadata_validates_and_is_retained_by_metadata() -> (
    None
):
    """Test stream metadata preserves an explicit nontrivial lane manifest."""
    stream = ResidentStreamMetadata(2, 41, ("north", "south"), (1, 0))
    metadata = ResidentMetadata(Device(Backend.WARP, "cpu"), (), stream)

    assert metadata.stream is stream
    assert stream.root_seed == 41
    assert stream.logical_box_ids == ("north", "south")
    assert stream.lanes == (1, 0)
    with pytest.raises(ValueError, match="permutation"):
        ResidentStreamMetadata(2, 41, ("north", "south"), (0, 0))
    with pytest.raises(ValueError, match="unique"):
        ResidentStreamMetadata(2, 41, ("north", "north"), (0, 1))


@pytest.mark.parametrize("names", [("ok", object()), ("ok", 1)])
def test_metadata_rejects_non_string_names(names: tuple[object, ...]) -> None:
    """Test metadata requires exact string gas-name entries."""
    with pytest.raises(TypeError, match="exact str instances"):
        ResidentMetadata(Device(Backend.WARP, "cpu"), names)  # type: ignore[arg-type]


def test_module_import_does_not_load_warp_or_gpu() -> None:
    """Test importing carriers and using metadata stays optional-backend free."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import sys
from particula.execution import Backend, Device
from particula.execution.gpu_session import ResidentDimensions, ResidentLifecycle, ResidentMetadata
ResidentDimensions(1, 0, 0)
ResidentMetadata(Device(Backend.WARP, 'cpu'), ())
assert ResidentLifecycle.ACTIVE.value == 'active'
assert not any(name == 'warp' or name.startswith('particula.gpu') for name in sys.modules)
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.warp
@pytest.mark.parametrize(
    "shape",
    [(1, 2, 1), (2, 1, 2), (1, 0, 0), (1, 0, 1), (1, 2, 0)],
)
def test_session_retains_valid_resources_by_identity(
    shape: tuple[int, int, int],
) -> None:
    """Test every declared lifecycle is accepted as immutable P1 state.

    P4 owns transition legality; P1 only accepts each declared value.
    """
    boxes, particle_count, species = shape
    particles, gas, environment = _warp_resources(
        boxes, particle_count, species
    )
    dimensions = ResidentDimensions(*shape)
    metadata = _metadata(species)
    before = _snapshot_resources(particles, gas, environment)
    for lifecycle in ResidentLifecycle:
        session = ResidentSession(
            particles, gas, environment, dimensions, metadata, lifecycle
        )
        assert session.particles is particles
        assert session.gas is gas
        assert session.environment is environment
        assert session.dimensions is dimensions
        assert session.metadata is metadata
        assert session.lifecycle is lifecycle
        assert session.metadata.gas_names is metadata.gas_names
        assert all(
            actual is expected
            for actual, expected in zip(
                session.metadata.gas_names, metadata.gas_names, strict=True
            )
        )
        assert session != ResidentSession(
            particles, gas, environment, dimensions, metadata, lifecycle
        )
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_session_preserves_gas_name_identity() -> None:
    """Test valid session construction preserves valid gas-name tuples."""
    particles, gas, environment = _warp_resources(species=2)
    names = ("species_0", "species_1")
    metadata = ResidentMetadata(Device(Backend.WARP, "cpu"), names)

    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 2, 2),
        metadata,
        ResidentLifecycle.ACTIVE,
    )

    assert session.metadata.gas_names is names
    assert session.metadata.gas_names == names


@pytest.mark.warp
def test_session_rejects_schema_and_metadata_mismatches() -> None:
    """Test isolated schema failures are deterministic and read-only."""
    particles, gas, environment = _warp_resources()
    dimensions = ResidentDimensions(1, 2, 1)

    with pytest.raises(ValueError, match="gas_names length"):
        ResidentSession(
            particles,
            gas,
            environment,
            dimensions,
            ResidentMetadata(Device(Backend.WARP, "cpu"), ("a", "b")),
            ResidentLifecycle.ACTIVE,
        )
    with pytest.raises(ValueError, match="metadata.device.native"):
        ResidentSession(
            particles,
            gas,
            environment,
            dimensions,
            ResidentMetadata(Device(Backend.WARP, "cuda:0"), ("a",)),
            ResidentLifecycle.ACTIVE,
        )
    with pytest.raises(ValueError, match="masses shape must match"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 1, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("carrier", "field", "aliased_carrier", "aliased_field"),
    [
        ("particles", "charge", "particles", "concentration"),
        ("environment", "saturation_ratio", "gas", "concentration"),
    ],
)
def test_session_rejects_overlapping_primary_arrays(
    carrier: str,
    field: str,
    aliased_carrier: str,
    aliased_field: str,
) -> None:
    """Primary arrays with compatible schemas cannot reuse resident storage."""
    particles, gas, environment = _warp_resources()
    carriers = {
        "particles": particles,
        "gas": gas,
        "environment": environment,
    }
    object.__setattr__(
        carriers[carrier],
        field,
        getattr(carriers[aliased_carrier], aliased_field),
    )

    with pytest.raises(
        ValueError, match="resident primary arrays must not alias"
    ):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )


def test_preflight_rejects_cpu_carriers_before_warp_import() -> None:
    """Test carrier and identity failures precede optional Warp import."""
    dimensions = ResidentDimensions(1, 0, 0)
    metadata = ResidentMetadata(Device(Backend.WARP, "cpu"), ())
    with pytest.raises(TypeError, match="exact ResidentDimensions"):
        ResidentSession(
            object(),
            object(),
            object(),
            cast(ResidentDimensions, object()),
            metadata,
            ResidentLifecycle.ACTIVE,
        )
    for first, second, message in (
        ("particles", "gas", "particles and gas"),
        ("particles", "environment", "particles and environment"),
        ("gas", "environment", "gas and environment"),
    ):
        shared = object()
        resources = {
            "particles": object(),
            "gas": object(),
            "environment": object(),
        }
        resources[first] = shared
        resources[second] = shared
        with pytest.raises(ValueError, match=message):
            ResidentSession(
                resources["particles"],
                resources["gas"],
                resources["environment"],
                dimensions,
                metadata,
                ResidentLifecycle.ACTIVE,
            )


def test_missing_warp_runtime_has_stable_error_after_cpu_preflight() -> None:
    """Test only resident validation requires the optional Warp runtime."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import builtins
from particula.execution import Backend, Device
from particula.execution.gpu_session import (
    ResidentDimensions, ResidentLifecycle, ResidentMetadata, ResidentSession,
)
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'warp' or name.startswith('warp.'):
        raise ModuleNotFoundError("No module named 'warp'", name='warp')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
metadata = ResidentMetadata(Device(Backend.WARP, 'cpu'), ())
try:
    ResidentSession(object(), object(), object(), object(), metadata, ResidentLifecycle.ACTIVE)
except TypeError as error:
    assert str(error) == 'dimensions must be an exact ResidentDimensions.'
else:
    raise AssertionError('carrier error did not precede Warp import')
shared = object()
try:
    ResidentSession(shared, shared, object(), ResidentDimensions(1, 0, 0), metadata, ResidentLifecycle.ACTIVE)
except ValueError as error:
    assert str(error) == 'particles and gas must not be identical.'
else:
    raise AssertionError('identity error did not precede Warp import')
try:
    ResidentSession(object(), object(), object(), ResidentDimensions(1, 0, 0), metadata, ResidentLifecycle.ACTIVE)
except RuntimeError as error:
    assert str(error) == 'ResidentSession requires the optional Warp runtime.'
else:
    raise AssertionError('resident validation did not require Warp')
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_warp_runtime_is_covered_without_loading_generated_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test resident validation converts only a missing top-level Warp import."""
    import builtins

    original_import: Any = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        """Raise only for the documented optional runtime import."""
        if name == "warp":
            raise ModuleNotFoundError("No module named 'warp'", name="warp")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(
        RuntimeError,
        match="ResidentSession requires the optional Warp runtime",
    ):
        ResidentSession(
            object(),
            object(),
            object(),
            ResidentDimensions(1, 0, 0),
            _metadata(0),
            ResidentLifecycle.ACTIVE,
        )


def test_session_carriers_remain_concrete_only() -> None:
    """Test P1 names do not alter public execution, adapter, or root exports."""
    import particula
    import particula.execution as execution
    import particula.execution.adapters as adapters

    names = {
        "ResidentDimensions",
        "ResidentLifecycle",
        "ResidentMetadata",
        "ResidentSession",
        "setup_resident_session",
        "ResidentStepGuard",
        "ResidentStepToken",
    }
    assert names.isdisjoint(execution.__all__)
    assert all(not hasattr(particula, name) for name in names)
    assert all(not hasattr(execution, name) for name in names)
    assert all(not hasattr(adapters, name) for name in names)


@pytest.mark.warp
def test_step_guard_completes_identity_tokens_and_preserves_duration_type() -> (
    None
):
    """Test matching completions alone advance concrete guard bookkeeping."""
    guard = _guard()

    zero = guard.begin_step(0)
    assert type(zero) is ResidentStepToken
    assert zero is not ResidentStepToken(guard, 0)
    guard.complete_step(zero)
    rational = Fraction(3, 2)
    token = guard.begin_step(rational)
    guard.complete_step(token)

    assert guard.completed_steps == 2
    assert guard.simulated_time == rational
    guard.assert_step_closed()
    with pytest.raises(FrozenInstanceError):
        token._duration = 0  # type: ignore[misc]


@pytest.mark.warp
def test_step_guard_rejections_preserve_open_token_and_bookkeeping() -> None:
    """Test invalid durations and completions do not mutate guard state."""
    guard = _guard()
    for value, error in (
        (True, TypeError),
        (object(), TypeError),
        (-1, ValueError),
    ):
        with pytest.raises(error):
            guard.begin_step(value)  # type: ignore[arg-type]
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            guard.begin_step(value)
    with pytest.raises(RuntimeError, match="No resident timestep"):
        guard.complete_step(object())  # type: ignore[arg-type]

    token = guard.begin_step(1.0)
    with pytest.raises(RuntimeError, match="already open"):
        guard.begin_step(0)
    with pytest.raises(ValueError, match="does not match"):
        guard.complete_step(ResidentStepToken(guard, 1.0))
    with pytest.raises(RuntimeError, match="timestep is open"):
        guard.assert_step_closed()
    assert guard.completed_steps == 0
    assert guard.simulated_time == 0
    guard.complete_step(token)
    with pytest.raises(RuntimeError, match="No resident timestep"):
        guard.complete_step(token)


@pytest.mark.warp
def test_step_guards_share_one_open_token_per_registry_binding() -> None:
    """Test guards sharing one binding cannot overlap resident timesteps."""
    from particula.execution.gpu_resources import GPUResourceRegistry

    particles, gas, environment = _warp_resources()
    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 2, 1),
        _metadata(),
        ResidentLifecycle.ACTIVE,
    )
    registry = GPUResourceRegistry(session)
    first_guard = ResidentStepGuard(session, registry)
    second_guard = ResidentStepGuard(session, registry)

    token = first_guard.begin_step(1.0)
    with pytest.raises(RuntimeError, match="already open"):
        second_guard.begin_step(1.0)

    assert first_guard._open_token is token
    assert second_guard._open_token is None
    assert registry._open_step_token is token
    first_guard.complete_step(token)

    second_token = second_guard.begin_step(2.0)
    assert registry._open_step_token is second_token
    second_guard.complete_step(second_token)


@pytest.mark.warp
def test_step_guard_rejects_nonfinite_completed_time_atomically() -> None:
    """Test overflowing completed time preserves the open-step ownership."""
    from particula.execution.gpu_resources import GPUResourceRegistry

    particles, gas, environment = _warp_resources()
    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 2, 1),
        _metadata(),
        ResidentLifecycle.ACTIVE,
    )
    registry = GPUResourceRegistry(session)
    guard = ResidentStepGuard(session, registry)
    other_guard = ResidentStepGuard(session, registry)
    object.__setattr__(guard, "_simulated_time", sys.float_info.max)
    token = guard.begin_step(sys.float_info.max)

    with pytest.raises(
        ValueError, match="completed simulated time must be finite"
    ):
        guard.complete_step(token)

    assert guard.completed_steps == 0
    assert guard.simulated_time == sys.float_info.max
    assert guard._open_token is token
    assert registry._open_step_token is token
    with pytest.raises(RuntimeError, match="already open"):
        other_guard.begin_step(1.0)


@pytest.mark.warp
def test_step_guard_completion_time_failure_is_atomic() -> None:
    """Test failed time arithmetic preserves guard and binding ownership."""
    from particula.execution.gpu_resources import GPUResourceRegistry

    class ExplodingDuration:
        """Provide an accepted real duration that fails time addition."""

        def __float__(self) -> float:
            return 0.5

        def __lt__(self, _: object) -> bool:
            return False

        def __radd__(self, _: object) -> object:
            raise RuntimeError("time addition failed")

    Real.register(ExplodingDuration)

    particles, gas, environment = _warp_resources()
    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 2, 1),
        _metadata(),
        ResidentLifecycle.ACTIVE,
    )
    registry = GPUResourceRegistry(session)
    guard = ResidentStepGuard(session, registry)
    other_guard = ResidentStepGuard(session, registry)
    token = guard.begin_step(ExplodingDuration())

    with pytest.raises(RuntimeError, match="time addition failed"):
        guard.complete_step(token)

    assert guard.completed_steps == 0
    assert guard.simulated_time == 0
    assert guard._open_token is token
    assert registry._open_step_token is token
    with pytest.raises(RuntimeError, match="already open"):
        other_guard.begin_step(1.0)


@pytest.mark.warp
def test_step_guard_completion_drift_preserves_open_bookkeeping() -> None:
    """Test completion binding drift leaves the token and counts unchanged."""
    guard = _guard()
    token = guard.begin_step(1.0)
    object.__setattr__(guard._session, "lifecycle", ResidentLifecycle.CLOSED)

    with pytest.raises(ValueError, match="ACTIVE"):
        guard.complete_step(token)

    assert guard._open_token is token
    assert guard._registry._open_step_token is token
    assert guard.completed_steps == 0
    assert guard.simulated_time == 0


@pytest.mark.warp
def test_step_guard_valid_transitions_perform_no_runtime_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test valid guard transitions remain metadata-only bookkeeping."""
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry
    from particula.gpu import conversion

    guard = _guard()

    def forbidden(*_: object, **__: object) -> None:
        """Fail when a guard transition attempts runtime work."""
        raise AssertionError("step transitions must not perform runtime work")

    for name in (
        "acquire_condensation",
        "acquire_coagulation",
        "acquire_wall_loss",
        "acquire_nucleation",
        "_allocate",
    ):
        monkeypatch.setattr(GPUResourceRegistry, name, forbidden)
    if hasattr(wp, "synchronize"):
        monkeypatch.setattr(wp, "synchronize", forbidden)
    for name in (
        "to_warp_particle_data",
        "to_warp_gas_data",
        "to_warp_environment_data",
        "from_warp_particle_data",
        "from_warp_gas_data",
        "from_warp_environment_data",
    ):
        if hasattr(conversion, name):
            monkeypatch.setattr(conversion, name, forbidden)

    token = guard.begin_step(1.0)
    guard.complete_step(token)

    assert guard.completed_steps == 1
    assert guard.simulated_time == 1.0


@pytest.mark.warp
def test_step_guard_validates_duration_before_registry_or_token_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invalid duration rejects before registry work or opening a step."""
    guard = _guard()
    calls = 0

    def forbidden_validation(_: object) -> None:
        """Fail if duration validation incorrectly reaches the registry."""
        nonlocal calls
        calls += 1
        raise AssertionError("registry validation must not run")

    monkeypatch.setattr(
        guard._registry,
        "validate_pinned_session",
        forbidden_validation,
    )

    with pytest.raises(ValueError, match="finite and nonnegative"):
        guard.begin_step(-1)

    assert calls == 0
    assert guard._open_token is None
    assert guard.completed_steps == 0
    assert guard.simulated_time == 0


@pytest.mark.warp
def test_step_guard_rejects_token_from_another_guard_without_mutation() -> None:
    """Test only the exact outstanding token can complete a guard."""
    guard = _guard()
    other_guard = _guard()
    token = guard.begin_step(1.0)
    other_token = other_guard.begin_step(2.0)

    with pytest.raises(ValueError, match="does not match"):
        guard.complete_step(other_token)

    assert guard._open_token is token
    assert guard.completed_steps == 0
    assert guard.simulated_time == 0
    guard.complete_step(token)
    other_guard.complete_step(other_token)


@pytest.mark.warp
def test_step_guard_rejects_terminal_session_without_metadata_change() -> None:
    """Test registry lifecycle validation precedes guard state transitions."""
    guard = _guard()
    object.__setattr__(guard._session, "lifecycle", ResidentLifecycle.CLOSED)

    with pytest.raises(ValueError, match="ACTIVE"):
        guard.begin_step(1.0)

    assert guard.completed_steps == 0
    assert guard.simulated_time == 0


def test_session_revalidates_fabricated_cpu_carriers_before_warp_import() -> (
    None
):
    """Test bypassed carrier, metadata, name, and lifecycle errors are ordered."""
    dimensions = object.__new__(ResidentDimensions)
    object.__setattr__(dimensions, "n_boxes", -1)
    object.__setattr__(dimensions, "n_particles", 0)
    object.__setattr__(dimensions, "n_species", 0)
    metadata = _metadata(0)
    with pytest.raises(ValueError, match="n_boxes must be nonnegative"):
        ResidentSession(
            object(),
            object(),
            object(),
            dimensions,
            metadata,
            ResidentLifecycle.ACTIVE,
        )

    invalid_metadata = object.__new__(ResidentMetadata)
    object.__setattr__(invalid_metadata, "device", Device(Backend.CPU, "cpu"))
    object.__setattr__(invalid_metadata, "gas_names", ())
    with pytest.raises(ValueError, match="device.backend must be Backend.WARP"):
        ResidentSession(
            object(),
            object(),
            object(),
            ResidentDimensions(1, 0, 0),
            invalid_metadata,
            ResidentLifecycle.ACTIVE,
        )

    invalid_names = object.__new__(ResidentMetadata)
    object.__setattr__(invalid_names, "device", Device(Backend.WARP, "cpu"))
    object.__setattr__(invalid_names, "gas_names", ("ok", object()))
    with pytest.raises(TypeError, match="exact str instances"):
        ResidentSession(
            object(),
            object(),
            object(),
            ResidentDimensions(1, 0, 0),
            invalid_names,
            ResidentLifecycle.ACTIVE,
        )

    with pytest.raises(TypeError, match="lifecycle must be an exact"):
        ResidentSession(
            object(),
            object(),
            object(),
            ResidentDimensions(1, 0, 0),
            _metadata(0),
            "active",  # type: ignore[arg-type]
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("carrier", "message"),
    [
        ("particles", "particles must be a WarpParticleData"),
        ("gas", "gas must be a WarpGasData"),
        ("environment", "environment must be a WarpEnvironmentData"),
    ],
)
def test_session_rejects_each_invalid_generated_container(
    carrier: str, message: str
) -> None:
    """Test each generated-container form has its own stable failure."""
    particles, gas, environment = _warp_resources()
    resources = {"particles": particles, "gas": gas, "environment": environment}
    resources[carrier] = object()
    with pytest.raises(TypeError, match=message):
        ResidentSession(
            resources["particles"],
            resources["gas"],
            resources["environment"],
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )


@pytest.mark.warp
@pytest.mark.parametrize("carrier,field,shape,dtype_name", _primary_fields())
def test_session_rejects_each_primary_array_wrong_shape_read_only(
    carrier: str, field: str, shape: tuple[int, ...], dtype_name: str
) -> None:
    """Test every primary field rejects one isolated wrong-rank or shape schema."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()
    before = _snapshot_resources(particles, gas, environment)
    resource = {"particles": particles, "gas": gas, "environment": environment}[
        carrier
    ]
    original = getattr(resource, field)
    wrong_shape = shape[:-1] if field == "masses" else shape + (1,)
    object.__setattr__(
        resource,
        field,
        _MetadataArray(getattr(wp, dtype_name), wrong_shape, original.device),
    )

    expected = (
        "rank 3" if field == "masses" else f"{carrier}.{field} must have shape"
    )
    with pytest.raises(ValueError, match=expected):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(resource, field, original)
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
@pytest.mark.parametrize("carrier,field,shape,dtype_name", _primary_fields())
def test_session_rejects_each_primary_array_wrong_dtype_read_only(
    carrier: str, field: str, shape: tuple[int, ...], dtype_name: str
) -> None:
    """Test every primary field rejects an isolated wrong dtype without writes."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()
    before = _snapshot_resources(particles, gas, environment)
    resource = {"particles": particles, "gas": gas, "environment": environment}[
        carrier
    ]
    original = getattr(resource, field)
    wrong_dtype = wp.int32 if dtype_name == "float64" else wp.float64
    object.__setattr__(
        resource, field, _MetadataArray(wrong_dtype, shape, original.device)
    )

    with pytest.raises(ValueError, match=f"{carrier}.{field} must use dtype"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(resource, field, original)
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
@pytest.mark.parametrize(
    "carrier,field,shape,dtype_name", _primary_fields()[1:]
)
def test_session_rejects_each_primary_array_wrong_device_read_only(
    carrier: str, field: str, shape: tuple[int, ...], dtype_name: str
) -> None:
    """Test every non-anchor primary field requires the masses device exactly."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()
    before = _snapshot_resources(particles, gas, environment)
    resource = {"particles": particles, "gas": gas, "environment": environment}[
        carrier
    ]
    original = getattr(resource, field)
    object.__setattr__(
        resource,
        field,
        _MetadataArray(getattr(wp, dtype_name), shape, object()),
    )

    with pytest.raises(
        ValueError, match="device must match particles.masses device"
    ):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(resource, field, original)
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_session_rejects_masses_metadata_and_native_device_read_only() -> None:
    """Test the anchor array validates array form, dtype, shape, and native id."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()
    original = particles.masses
    before = _snapshot_resources(particles, gas, environment)
    object.__setattr__(particles, "masses", object())
    with pytest.raises(
        ValueError, match="particles.masses must be a Warp array"
    ):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(
        particles,
        "masses",
        _MetadataArray(wp.int32, (1, 2, 1), original.device),
    )
    with pytest.raises(ValueError, match="particles.masses must use dtype"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(
        particles,
        "masses",
        _MetadataArray(wp.float64, [1, 2, 1], original.device),
    )
    with pytest.raises(ValueError, match="tuple shape"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(particles, "masses", original)
    with pytest.raises(ValueError, match="metadata.device.native"):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            ResidentMetadata(Device(Backend.WARP, "not-cpu"), ("species_0",)),
            ResidentLifecycle.ACTIVE,
        )
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_session_rejects_primary_array_without_metadata_read_only() -> None:
    """Test non-anchor arrays must expose all three required metadata fields."""
    particles, gas, environment = _warp_resources()
    before = _snapshot_resources(particles, gas, environment)
    original = gas.concentration
    object.__setattr__(gas, "concentration", object())
    with pytest.raises(
        ValueError, match="gas.concentration must be a Warp array"
    ):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )
    object.__setattr__(gas, "concentration", original)
    assert _snapshot_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_session_schema_validation_never_performs_runtime_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test valid and invalid schema construction performs metadata-only work."""
    wp = pytest.importorskip("warp")
    particles, gas, environment = _warp_resources()

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("P1 must not perform runtime work.")

    for name in (
        "synchronize",
        "launch",
        "copy",
        "from_numpy",
        "zeros",
        "ones",
        "empty",
        "full",
        "zeros_like",
        "ones_like",
        "empty_like",
        "clone",
    ):
        if hasattr(wp, name):
            monkeypatch.setattr(wp, name, forbidden)
    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 2, 1),
        _metadata(),
        ResidentLifecycle.ACTIVE,
    )
    assert session.particles is particles

    object.__setattr__(
        particles,
        "concentration",
        _MetadataArray(wp.float64, (1, 1), particles.masses.device),
    )
    with pytest.raises(
        ValueError, match="particles.concentration must have shape"
    ):
        ResidentSession(
            particles,
            gas,
            environment,
            ResidentDimensions(1, 2, 1),
            _metadata(),
            ResidentLifecycle.ACTIVE,
        )


def _cpu_resources(
    boxes: int = 1,
    particle_count: int = 2,
    species: int = 1,
) -> tuple[ParticleData, GasData, EnvironmentData]:
    """Create matching float64 CPU carriers for resident-session setup."""
    particles = ParticleData(
        masses=np.ones((boxes, particle_count, species), dtype=np.float64),
        concentration=np.ones((boxes, particle_count), dtype=np.float64),
        charge=np.zeros((boxes, particle_count), dtype=np.float64),
        density=np.ones(species, dtype=np.float64),
        volume=np.ones(boxes, dtype=np.float64),
    )
    gas = GasData(
        name=[f"species_{index}" for index in range(species)],
        molar_mass=np.ones(species, dtype=np.float64),
        concentration=np.ones((boxes, species), dtype=np.float64),
        partitioning=np.ones(species, dtype=np.bool_),
    )
    if boxes == 0:
        environment = object.__new__(EnvironmentData)
        object.__setattr__(
            environment,
            "temperature",
            np.full(0, 298.15, dtype=np.float64),
        )
        object.__setattr__(
            environment,
            "pressure",
            np.full(0, 101325.0, dtype=np.float64),
        )
        object.__setattr__(
            environment,
            "saturation_ratio",
            np.ones((0, species), dtype=np.float64),
        )
    else:
        environment = EnvironmentData(
            temperature=np.full(boxes, 298.15, dtype=np.float64),
            pressure=np.full(boxes, 101325.0, dtype=np.float64),
            saturation_ratio=np.ones((boxes, species), dtype=np.float64),
        )
    return particles, gas, environment


def test_setup_resident_session_rejects_zero_box_schema_before_upload() -> None:
    """Test unsupported empty sessions fail before importing Warp converters."""
    particles, gas, environment = _cpu_resources(boxes=0)

    with pytest.raises(ValueError, match="n_boxes must be greater than zero"):
        setup_resident_session(
            particles, gas, environment, Device(Backend.WARP, "cpu")
        )


@pytest.mark.warp
def test_setup_resident_session_converts_once_and_retains_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setup performs the sole ordered upload after CPU-only preflight."""
    particles, gas, environment = _cpu_resources(boxes=2, species=2)
    before = _snapshot_cpu_resources(particles, gas, environment)
    warp_particles, warp_gas, warp_environment = _warp_resources(2, 2, 2)
    from particula.gpu import conversion

    calls: list[tuple[object, str]] = []

    def particle_upload(value: object, *, device: str) -> object:
        calls.append((value, device))
        return warp_particles

    def gas_upload(value: object, *, device: str) -> object:
        calls.append((value, device))
        return warp_gas

    def environment_upload(value: object, *, device: str) -> object:
        calls.append((value, device))
        return warp_environment

    monkeypatch.setattr(conversion, "to_warp_particle_data", particle_upload)
    monkeypatch.setattr(conversion, "to_warp_gas_data", gas_upload)
    monkeypatch.setattr(
        conversion, "to_warp_environment_data", environment_upload
    )

    session = setup_resident_session(
        particles,
        gas,
        environment,
        Device(Backend.WARP, "cpu"),
    )

    assert calls == [(particles, "cpu"), (gas, "cpu"), (environment, "cpu")]
    assert session.particles is warp_particles
    assert session.gas is warp_gas
    assert session.environment is warp_environment
    assert session.dimensions == ResidentDimensions(2, 2, 2)
    assert session.metadata.gas_names == tuple(gas.name)
    assert session.lifecycle is ResidentLifecycle.ACTIVE
    assert not hasattr(warp_gas, "name")
    assert _snapshot_cpu_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_setup_resident_session_retains_explicit_stream_manifest() -> None:
    """Test setup publishes caller-selected stream identity without coercion."""
    particles, gas, environment = _cpu_resources(boxes=2)

    session = setup_resident_session(
        particles,
        gas,
        environment,
        Device(Backend.WARP, "cpu"),
        root_seed=41,
        logical_box_ids=("north", "south"),
        lanes=(1, 0),
    )

    assert session.metadata.stream == ResidentStreamMetadata(
        2, 41, ("north", "south"), (1, 0)
    )


@pytest.mark.parametrize(
    ("argument", "error", "message"),
    [
        ("device", TypeError, "exact Device"),
        ("cpu_device", ValueError, "Backend.WARP"),
        ("particles", TypeError, "ParticleData"),
        ("gas", TypeError, "GasData"),
        ("environment", TypeError, "EnvironmentData"),
    ],
)
def test_setup_resident_session_rejects_local_types_before_conversion(
    argument: str,
    error: type[Exception],
    message: str,
) -> None:
    """Test local device and carrier failures precede optional conversion."""
    particles, gas, environment = _cpu_resources()
    device: object = Device(Backend.WARP, "cpu")
    if argument == "device":
        device = object()
    elif argument == "cpu_device":
        device = Device(Backend.CPU, "cpu")
    elif argument == "particles":
        particles = cast(ParticleData, object())
    elif argument == "gas":
        gas = cast(GasData, object())
    else:
        environment = cast(EnvironmentData, object())

    with pytest.raises(error, match=message):
        setup_resident_session(particles, gas, environment, device)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "concentration",
        "charge",
        "density",
        "volume",
    ],
)
def test_setup_resident_session_rejects_particle_shape_before_conversion(
    field: str,
) -> None:
    """Test each cross-container particle schema field is checked locally."""
    particles, gas, environment = _cpu_resources()
    wrong_shape = (1,) if field in {"concentration", "charge"} else (2,)
    setattr(particles, field, np.zeros(wrong_shape, dtype=np.float64))
    before = _snapshot_cpu_resources(particles, gas, environment)

    with pytest.raises(ValueError, match=f"particles.{field} must have shape"):
        setup_resident_session(
            particles,
            gas,
            environment,
            Device(Backend.WARP, "cpu"),
        )

    assert _snapshot_cpu_resources(particles, gas, environment) == before


def _snapshot_cpu_resources(
    particles: ParticleData,
    gas: GasData,
    environment: EnvironmentData,
) -> tuple[tuple[object, ...], ...]:
    """Capture CPU carrier arrays and ordered gas names without mutation."""
    values = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.concentration,
        gas.partitioning,
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
    )
    return tuple(
        (id(value), value.shape, tuple(value.flat)) for value in values
    ) + ((id(gas.name), tuple(gas.name)),)


@pytest.mark.parametrize(
    ("carrier", "field", "wrong_shape", "message"),
    [
        ("particles", "masses", (1, 2), "particles.masses must have rank 3"),
        ("gas", "molar_mass", (2,), "gas.molar_mass must have shape"),
        ("gas", "partitioning", (2,), "gas.partitioning must have shape"),
        ("gas", "concentration", (1,), "gas.concentration must have shape"),
        (
            "environment",
            "temperature",
            (2,),
            "environment.temperature must have shape",
        ),
        (
            "environment",
            "pressure",
            (2,),
            "environment.pressure must have shape",
        ),
        (
            "environment",
            "saturation_ratio",
            (1,),
            "environment.saturation_ratio must have shape",
        ),
    ],
)
def test_setup_resident_session_rejects_remaining_cpu_schema_before_upload(
    carrier: str,
    field: str,
    wrong_shape: tuple[int, ...],
    message: str,
) -> None:
    """Test every remaining cross-carrier schema rejection is read-only."""
    particles, gas, environment = _cpu_resources()
    target = {
        "particles": particles,
        "gas": gas,
        "environment": environment,
    }[carrier]
    setattr(target, field, np.zeros(wrong_shape, dtype=np.float64))
    before = _snapshot_cpu_resources(particles, gas, environment)

    with pytest.raises(ValueError, match=message):
        setup_resident_session(
            particles, gas, environment, Device(Backend.WARP, "cpu")
        )

    assert _snapshot_cpu_resources(particles, gas, environment) == before


@pytest.mark.parametrize(
    ("names", "error", "message"),
    [
        (["species_0", "extra"], ValueError, "gas.name length"),
        ([object()], TypeError, "exact str instances"),
        ("ab", TypeError, "ordered collection of strings"),
    ],
)
def test_setup_resident_session_rejects_gas_names_before_upload(
    names: object, error: type[Exception], message: str
) -> None:
    """Test invalid CPU gas-name metadata cannot reach conversion helpers."""
    particles, gas, environment = _cpu_resources()
    gas.name = names  # type: ignore[assignment]
    before = _snapshot_cpu_resources(particles, gas, environment)

    with pytest.raises(error, match=message):
        setup_resident_session(
            particles, gas, environment, Device(Backend.WARP, "cpu")
        )

    assert _snapshot_cpu_resources(particles, gas, environment) == before


def test_setup_resident_session_rejects_device_subclass_before_upload() -> None:
    """Test setup requires an exact Device rather than a compatible subclass."""

    class DeviceSubclass(Device):
        """Provide a distinct Device runtime type for exactness coverage."""

    particles, gas, environment = _cpu_resources()
    with pytest.raises(TypeError, match="exact Device"):
        setup_resident_session(
            particles, gas, environment, DeviceSubclass(Backend.WARP, "cpu")
        )


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("device = object()", "TypeError", "exact Device"),
        (
            "device = type('DeviceSubclass', (Device,), {})(Backend.WARP, 'cpu')",
            "TypeError",
            "exact Device",
        ),
        ("device = Device(Backend.CPU, 'cpu')", "ValueError", "Backend.WARP"),
        ("particles = object()", "TypeError", "ParticleData"),
        ("gas = object()", "TypeError", "GasData"),
        ("environment = object()", "TypeError", "EnvironmentData"),
        ("particles.masses = np.ones((1, 2))", "ValueError", "rank 3"),
        (
            "particles.concentration = np.ones((1,))",
            "ValueError",
            "concentration",
        ),
        ("particles.charge = np.ones((1,))", "ValueError", "charge"),
        ("particles.density = np.ones((2,))", "ValueError", "density"),
        ("particles.volume = np.ones((2,))", "ValueError", "volume"),
        ("gas.molar_mass = np.ones((2,))", "ValueError", "molar_mass"),
        (
            "gas.partitioning = np.ones((2,), dtype=bool)",
            "ValueError",
            "partitioning",
        ),
        ("gas.concentration = np.ones((1,))", "ValueError", "concentration"),
        ("gas.name = ['species_0', 'extra']", "ValueError", "name length"),
        ("gas.name = [object()]", "TypeError", "exact str instances"),
        (
            "gas.name = 'ab'",
            "TypeError",
            "ordered collection of strings",
        ),
        (
            "environment.temperature = np.ones((2,))",
            "ValueError",
            "temperature",
        ),
        ("environment.pressure = np.ones((2,))", "ValueError", "pressure"),
        (
            "environment.saturation_ratio = np.ones((1,))",
            "ValueError",
            "saturation_ratio",
        ),
    ],
)
def test_setup_resident_session_local_preflight_never_imports_gpu_subprocess(
    mutation: str, error: str, message: str
) -> None:
    """Test every local rejection precedes optional GPU and Warp imports."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = f"""
import builtins
import numpy as np
import sys
from particula.execution import Backend, Device
from particula.execution.gpu_session import setup_resident_session
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData

particles = ParticleData(
    masses=np.ones((1, 2, 1)), concentration=np.ones((1, 2)),
    charge=np.zeros((1, 2)), density=np.ones((1,)), volume=np.ones((1,)),
)
gas = GasData(
    name=['species_0'], molar_mass=np.ones((1,)),
    concentration=np.ones((1, 1)), partitioning=np.ones((1,), dtype=bool),
)
environment = EnvironmentData(
    temperature=np.ones((1,)), pressure=np.ones((1,)),
    saturation_ratio=np.ones((1, 1)),
)
device = Device(Backend.WARP, 'cpu')
{mutation}
empty = np.array([])
before = (
    tuple((id(value), value.shape, tuple(value.flat)) for value in (
        getattr(particles, 'masses', empty),
        getattr(particles, 'concentration', empty),
        getattr(particles, 'charge', empty),
        getattr(particles, 'density', empty),
        getattr(particles, 'volume', empty),
        getattr(gas, 'molar_mass', empty),
        getattr(gas, 'concentration', empty),
        getattr(gas, 'partitioning', empty),
        getattr(environment, 'temperature', empty),
        getattr(environment, 'pressure', empty),
        getattr(environment, 'saturation_ratio', empty),
    )),
    tuple(getattr(gas, 'name', [])),
)
original_import = builtins.__import__
builtins.__import__ = lambda name, *args, **kwargs: (_ for _ in ()).throw(AssertionError(f'unexpected optional import: {{name}}')) if (name == 'warp' or name.startswith('warp.') or name.startswith('particula.gpu')) else original_import(name, *args, **kwargs)
try:
    setup_resident_session(particles, gas, environment, device)
except {error} as caught:
    assert {message!r} in str(caught)
else:
    raise AssertionError('local preflight unexpectedly succeeded')
assert not any(name == 'warp' or name.startswith('particula.gpu') for name in sys.modules)
after = (
    tuple((id(value), value.shape, tuple(value.flat)) for value in (
        getattr(particles, 'masses', empty),
        getattr(particles, 'concentration', empty),
        getattr(particles, 'charge', empty),
        getattr(particles, 'density', empty),
        getattr(particles, 'volume', empty),
        getattr(gas, 'molar_mass', empty),
        getattr(gas, 'concentration', empty),
        getattr(gas, 'partitioning', empty),
        getattr(environment, 'temperature', empty),
        getattr(environment, 'pressure', empty),
        getattr(environment, 'saturation_ratio', empty),
    )),
    tuple(getattr(gas, 'name', [])),
)
assert after == before
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.warp
@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_setup_resident_session_propagates_upload_failure_in_order(
    monkeypatch: pytest.MonkeyPatch, failure_index: int
) -> None:
    """Test an upload failure publishes no session or later upload calls."""
    particles, gas, environment = _cpu_resources()
    before = _snapshot_cpu_resources(particles, gas, environment)
    warp_resources = _warp_resources()
    from particula.gpu import conversion

    calls: list[str] = []
    sentinel = RuntimeError(f"upload failure {failure_index}")

    def upload(index: int, name: str, result: object) -> Any:
        """Return one helper that records order and fails at its assigned slot."""

        def helper(value: object, *, device: str) -> object:
            assert device == "cpu"
            calls.append(name)
            if index == failure_index:
                raise sentinel
            return result

        return helper

    monkeypatch.setattr(
        conversion,
        "to_warp_particle_data",
        upload(0, "particles", warp_resources[0]),
    )
    monkeypatch.setattr(
        conversion, "to_warp_gas_data", upload(1, "gas", warp_resources[1])
    )
    monkeypatch.setattr(
        conversion,
        "to_warp_environment_data",
        upload(2, "environment", warp_resources[2]),
    )

    with pytest.raises(RuntimeError) as error:
        setup_resident_session(
            particles, gas, environment, Device(Backend.WARP, "cpu")
        )

    assert error.value is sentinel
    assert calls == ["particles", "gas", "environment"][: failure_index + 1]
    assert _snapshot_cpu_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_setup_resident_session_propagates_final_schema_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test all uploads complete before invalid data rejects publication."""
    particles, gas, environment = _cpu_resources()
    before = _snapshot_cpu_resources(particles, gas, environment)
    warp_particles, warp_gas, warp_environment = _warp_resources()
    from particula.gpu import conversion

    calls: list[str] = []

    def record(name: str, result: object) -> Any:
        """Build an upload stub that records its single invocation."""

        def helper(_: object, *, device: str) -> object:
            assert device == "cpu"
            calls.append(name)
            return result

        return helper

    object.__setattr__(warp_gas, "concentration", object())
    monkeypatch.setattr(
        conversion,
        "to_warp_particle_data",
        record("particles", warp_particles),
    )
    monkeypatch.setattr(
        conversion,
        "to_warp_gas_data",
        record("gas", warp_gas),
    )
    monkeypatch.setattr(
        conversion,
        "to_warp_environment_data",
        record("environment", warp_environment),
    )

    with pytest.raises(
        ValueError, match="gas.concentration must be a Warp array"
    ):
        setup_resident_session(
            particles, gas, environment, Device(Backend.WARP, "cpu")
        )

    assert calls == ["particles", "gas", "environment"]
    assert _snapshot_cpu_resources(particles, gas, environment) == before


@pytest.mark.warp
def test_failed_read_only_operation_releases_token_and_remains_reusable() -> (
    None
):
    """Test an explicitly read-only failure leaves a session reusable."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    token = guard.begin_step(1.0)

    _handle_failed_resident_operation(
        session,
        registry,
        guard,
        token,
        _ResidentOperationOutcome.READ_ONLY,
    )

    assert session.lifecycle is ResidentLifecycle.ACTIVE
    assert guard._open_token is None
    assert registry._open_step_token is None
    assert guard.completed_steps == 0
    assert guard.simulated_time == 0
    retry = guard.begin_step(2.0)
    guard.complete_step(retry)


@pytest.mark.warp
def test_failed_writer_operation_faults_and_can_close() -> None:
    """Test a possible launched writer faults without rolling back state."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    token = guard.begin_step(1.0)

    _handle_failed_resident_operation(
        session,
        registry,
        guard,
        token,
        _ResidentOperationOutcome.WRITER_MAY_HAVE_LAUNCHED,
    )

    assert session.lifecycle is ResidentLifecycle.FAULTED
    assert guard._open_token is None
    assert registry._open_step_token is None
    with pytest.raises(ValueError, match="ACTIVE"):
        guard.begin_step(1.0)
    session.close(registry, guard)
    assert session.lifecycle is ResidentLifecycle.CLOSED


@pytest.mark.warp
def test_abort_rejection_preserves_the_open_token() -> None:
    """Test abort validates before changing token ownership or counters."""
    guard = _guard()
    token = guard.begin_step(1.0)

    with pytest.raises(ValueError, match="does not match"):
        guard._abort_step(ResidentStepToken(guard, 1.0))

    assert guard._open_token is token
    assert guard._registry._open_step_token is token
    assert guard.completed_steps == 0
    guard.complete_step(token)


@pytest.mark.warp
def test_close_requires_closed_active_guard_and_is_idempotent() -> None:
    """Test active close is gated and closed close performs no validation."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    token = guard.begin_step(1.0)

    with pytest.raises(RuntimeError, match="timestep is open"):
        session.close(registry, guard)
    assert session.lifecycle is ResidentLifecycle.ACTIVE
    guard._abort_step(token)
    session.discard(registry, guard)
    assert session.lifecycle is ResidentLifecycle.CLOSED
    session.close(cast(Any, object()), cast(Any, object()))
    assert session.lifecycle is ResidentLifecycle.CLOSED


@pytest.mark.warp
def test_close_rejects_step_opened_by_another_shared_guard() -> None:
    """Test close observes registry-wide rather than only local step state."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    other_guard = ResidentStepGuard(session, registry)
    token = other_guard.begin_step(1.0)

    with pytest.raises(RuntimeError, match="timestep is open"):
        session.close(registry, guard)

    assert session.lifecycle is ResidentLifecycle.ACTIVE
    assert registry._open_step_token is token
    other_guard._abort_step(token)
    session.close(registry, guard)
    assert session.lifecycle is ResidentLifecycle.CLOSED


@pytest.mark.warp
@pytest.mark.parametrize(
    ("outcome", "expected_lifecycle"),
    [
        (_ResidentOperationOutcome.READ_ONLY, ResidentLifecycle.ACTIVE),
        (
            _ResidentOperationOutcome.WRITER_MAY_HAVE_LAUNCHED,
            ResidentLifecycle.FAULTED,
        ),
    ],
)
def test_direct_owner_failure_preserves_original_exception_and_cleans_token(
    outcome: _ResidentOperationOutcome,
    expected_lifecycle: ResidentLifecycle,
) -> None:
    """Test an owner reraises its original error after explicit cleanup."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    sentinel = RuntimeError("direct operation failed")

    def direct_owner() -> None:
        """Open a step, classify an injected failure, and preserve that error."""
        token = guard.begin_step(1.0)
        try:
            raise sentinel
        except BaseException:
            _handle_failed_resident_operation(
                session,
                registry,
                guard,
                token,
                outcome,
            )
            raise

    with pytest.raises(RuntimeError) as error:
        direct_owner()

    assert error.value is sentinel
    assert error.tb is not None
    assert session.lifecycle is expected_lifecycle
    assert guard._open_token is None
    assert registry._open_step_token is None
    assert guard.completed_steps == 0
    assert guard.simulated_time == 0
    if outcome is _ResidentOperationOutcome.READ_ONLY:
        retry = guard.begin_step(0.0)
        guard.complete_step(retry)
    else:
        with pytest.raises(ValueError, match="ACTIVE"):
            guard.begin_step(0.0)


@pytest.mark.warp
def test_failed_operation_rejections_leave_valid_open_token_unchanged() -> None:
    """Test failure-seam type and token errors do not silently abort a step."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    token = guard.begin_step(1.0)

    with pytest.raises(TypeError, match="exact _ResidentOperationOutcome"):
        _handle_failed_resident_operation(
            session,
            registry,
            guard,
            token,
            cast(Any, object()),
        )
    with pytest.raises(ValueError, match="does not match"):
        _handle_failed_resident_operation(
            session,
            registry,
            guard,
            ResidentStepToken(guard, 1.0),
            _ResidentOperationOutcome.READ_ONLY,
        )

    assert session.lifecycle is ResidentLifecycle.ACTIVE
    assert guard._open_token is token
    assert registry._open_step_token is token
    assert guard.completed_steps == 0
    assert guard.simulated_time == 0
    guard.complete_step(token)


@pytest.mark.warp
def test_faulted_close_uses_identity_binding_without_active_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test faulted close skips active-only registry validation."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    token = guard.begin_step(1.0)
    _handle_failed_resident_operation(
        session,
        registry,
        guard,
        token,
        _ResidentOperationOutcome.WRITER_MAY_HAVE_LAUNCHED,
    )

    def forbidden(_: object) -> None:
        """Fail if terminal fault disposal revalidates an active session."""
        raise AssertionError("faulted close must not validate active binding")

    monkeypatch.setattr(registry, "validate_pinned_session", forbidden)
    session.discard(registry, guard)

    assert session.lifecycle is ResidentLifecycle.CLOSED
    assert guard._open_token is None
    assert registry._open_step_token is None


@pytest.mark.warp
def test_active_close_validates_once_and_preserves_payload_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test active close validates once and changes only lifecycle state."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    before = _snapshot_resources(
        session.particles,
        session.gas,
        session.environment,
    )
    original_validation = registry.validate_pinned_session
    calls = 0

    def validate_once(candidate: ResidentSession) -> None:
        """Record the one active close binding validation."""
        nonlocal calls
        calls += 1
        original_validation(candidate)

    monkeypatch.setattr(registry, "validate_pinned_session", validate_once)
    session.close(registry, guard)

    assert calls == 1
    assert session.lifecycle is ResidentLifecycle.CLOSED
    assert (
        _snapshot_resources(
            session.particles,
            session.gas,
            session.environment,
        )
        == before
    )


@pytest.mark.warp
def test_abort_rejects_closed_token_without_changing_bookkeeping() -> None:
    """Test abort cannot release a token after it has already completed."""
    guard = _guard()
    token = guard.begin_step(1.5)
    guard.complete_step(token)

    with pytest.raises(RuntimeError, match="No resident timestep is open"):
        guard._abort_step(token)

    assert guard._open_token is None
    assert guard._registry._open_step_token is None
    assert guard.completed_steps == 1
    assert guard.simulated_time == 1.5


@pytest.mark.warp
def test_failure_seam_validation_rejection_preserves_open_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test failed writer validation faults while retaining its open token."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    token = guard.begin_step(1.0)
    original_validation = registry.validate_pinned_session

    def reject(_: ResidentSession) -> None:
        """Model a pinned-session validation failure before cleanup."""
        raise ValueError("pinned binding drift")

    monkeypatch.setattr(registry, "validate_pinned_session", reject)
    with pytest.raises(ValueError, match="pinned binding drift"):
        _handle_failed_resident_operation(
            session,
            registry,
            guard,
            token,
            _ResidentOperationOutcome.WRITER_MAY_HAVE_LAUNCHED,
        )

    assert session.lifecycle is ResidentLifecycle.FAULTED
    assert guard._open_token is token
    assert registry._open_step_token is token
    assert guard.completed_steps == 0
    assert guard.simulated_time == 0
    monkeypatch.setattr(
        registry, "validate_pinned_session", original_validation
    )
    with pytest.raises(ValueError, match="ACTIVE"):
        guard.begin_step(1.0)


@pytest.mark.warp
@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("session", "exact ResidentSession"),
        ("registry", "exact GPUResourceRegistry"),
        ("guard", "exact ResidentStepGuard"),
        ("token", "exact ResidentStepToken"),
    ],
)
def test_failure_seam_type_rejections_preserve_valid_open_token(
    argument: str, message: str
) -> None:
    """Test each exact-type rejection occurs before failure cleanup mutates."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    token = guard.begin_step(1.0)
    values: dict[str, object] = {
        "session": session,
        "registry": registry,
        "guard": guard,
        "token": token,
    }
    values[argument] = object()

    with pytest.raises(TypeError, match=message):
        _handle_failed_resident_operation(
            cast(ResidentSession, values["session"]),
            cast(Any, values["registry"]),
            cast(ResidentStepGuard, values["guard"]),
            cast(ResidentStepToken, values["token"]),
            _ResidentOperationOutcome.READ_ONLY,
        )

    assert session.lifecycle is ResidentLifecycle.ACTIVE
    assert guard._open_token is token
    assert registry._open_step_token is token
    guard.complete_step(token)


@pytest.mark.warp
def test_writer_cleanup_failure_does_not_replace_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a failed mutated writer faults without masking its error."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    sentinel = RuntimeError("writer failure")
    particles: Any = session.particles
    masses: Any = particles.masses
    before = masses.numpy().copy()
    masses.assign(np.full((1, 2, 1), 7.0, dtype=np.float64))

    def failed_cleanup(_: ResidentStepToken) -> None:
        """Inject a cleanup failure after a token has been opened."""
        raise ValueError("cleanup failed")

    monkeypatch.setattr(guard, "_abort_step", failed_cleanup)

    def direct_owner() -> None:
        """Model operation-error preservation with cleanup diagnostics."""
        token = guard.begin_step(1.0)
        try:
            raise sentinel
        except BaseException as operation_error:
            try:
                _handle_failed_resident_operation(
                    session,
                    registry,
                    guard,
                    token,
                    _ResidentOperationOutcome.WRITER_MAY_HAVE_LAUNCHED,
                )
            except BaseException as cleanup_error:
                raise operation_error from cleanup_error
            raise

    with pytest.raises(RuntimeError) as error:
        direct_owner()

    assert error.value is sentinel
    assert error.tb is not None
    assert str(error.value.__cause__) == "cleanup failed"
    # Incomplete cleanup retains the open token but faults the session.
    assert session.lifecycle is ResidentLifecycle.FAULTED
    assert guard._open_token is not None
    assert registry._open_step_token is guard._open_token
    np.testing.assert_array_equal(
        cast(Any, session.particles).masses.numpy(), np.full_like(before, 7.0)
    )
    np.testing.assert_array_equal(before, np.ones_like(before))
    with pytest.raises(ValueError, match="ACTIVE"):
        guard.begin_step(1.0)
    with pytest.raises(ValueError, match="ACTIVE"):
        registry.validate_pinned_session(session)


@pytest.mark.warp
def test_close_rejections_preserve_active_binding_and_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invalid active close leaves lifecycle, identities, and guard intact."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    before = _snapshot_resources(
        session.particles,
        session.gas,
        session.environment,
    )

    def reject(_: ResidentSession) -> None:
        """Reject the one required active-binding validation."""
        raise ValueError("active validation failed")

    monkeypatch.setattr(registry, "validate_pinned_session", reject)
    with pytest.raises(ValueError, match="active validation failed"):
        session.close(registry, guard)

    assert session.lifecycle is ResidentLifecycle.ACTIVE
    assert guard._open_token is None
    assert guard.completed_steps == 0
    assert guard.simulated_time == 0
    assert (
        _snapshot_resources(
            session.particles,
            session.gas,
            session.environment,
        )
        == before
    )


@pytest.mark.warp
def test_finalized_close_is_write_free_without_binding_validation() -> None:
    """Test finalized close ignores supplied bindings and retains final state."""
    guard = _guard()
    session = guard._session
    object.__setattr__(session, "lifecycle", ResidentLifecycle.FINALIZED)

    session.close(cast(Any, object()), cast(Any, object()))
    session.discard(cast(Any, object()), cast(Any, object()))

    assert session.lifecycle is ResidentLifecycle.FINALIZED
    assert guard._open_token is None
    assert guard.completed_steps == 0


@pytest.mark.warp
def test_session_stream_lifecycle_requires_closed_active_binding() -> None:
    """Test stream inspection/reset compose only through an exact closed guard."""
    from particula.execution.gpu_resources import GPUResourceRegistry

    particles, gas, environment = _warp_resources()
    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 2, 1),
        _metadata(),
        ResidentLifecycle.ACTIVE,
    )
    registry = GPUResourceRegistry(session)
    guard = ResidentStepGuard(session, registry)
    resources = registry.acquire_coagulation(1)

    manifest = session.inspect_streams(registry, guard)
    assert manifest.published_process_ids == ("coagulation",)
    session.reset_streams(registry, guard, logical_box_ids=("0",))
    token = guard.begin_step(1.0)
    before = resources.rng_states.numpy().copy()
    with pytest.raises(RuntimeError, match="open"):
        session.initialize_streams(registry, guard)
    np.testing.assert_array_equal(resources.rng_states.numpy(), before)
    guard.complete_step(token)


@pytest.mark.warp
def test_session_stream_inspection_never_synchronizes_or_reads_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test inspection returns frozen metadata without device synchronization."""
    wp = pytest.importorskip("warp")
    guard = _guard()
    session = guard._session
    registry = guard._registry
    registry.acquire_coagulation(1)

    def forbidden_sync(*_args: object, **_kwargs: object) -> None:
        """Fail if metadata inspection attempts a device synchronization."""
        raise AssertionError("stream inspection must not synchronize")

    monkeypatch.setattr(wp, "synchronize", forbidden_sync)
    manifest = session.inspect_streams(registry, guard)

    assert manifest.published_process_ids == ("coagulation",)


@pytest.mark.warp
def test_session_stream_lifecycle_rejects_invalid_binding_and_selectors() -> (
    None
):
    """Test stream lifecycle calls validate before changing published words."""
    guard = _guard()
    session = guard._session
    registry = guard._registry

    empty_manifest = session.inspect_streams(registry, guard)
    assert empty_manifest.published_process_ids == ()
    resources = registry.acquire_coagulation(1)
    before = resources.rng_states.numpy().copy()

    other_guard = _guard()
    with pytest.raises(ValueError, match="guard must match"):
        session.inspect_streams(registry, other_guard)
    with pytest.raises(ValueError, match="has not been acquired"):
        session.initialize_streams(registry, guard, process_ids=("wall_loss",))
    with pytest.raises(LookupError, match="No lane"):
        session.reset_streams(registry, guard, logical_box_ids=("missing",))

    np.testing.assert_array_equal(resources.rng_states.numpy(), before)


@pytest.mark.warp
def test_stream_lifecycle_binding_rejection_precedes_registry_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a mismatched guard cannot inspect or reset through the registry."""
    guard = _guard()
    session = guard._session
    registry = guard._registry
    other_guard = _guard()
    calls: list[str] = []

    monkeypatch.setattr(
        registry,
        "inspect_published_streams",
        lambda _session: calls.append("inspect"),
    )
    monkeypatch.setattr(
        registry,
        "initialize_published_streams",
        lambda *_args, **_kwargs: calls.append("initialize"),
    )

    with pytest.raises(ValueError, match="guard must match"):
        session.inspect_streams(registry, other_guard)
    with pytest.raises(ValueError, match="guard must match"):
        session.reset_streams(registry, other_guard)

    assert calls == []
