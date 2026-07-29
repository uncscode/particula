"""Focused boundary checks for concrete resident checkpoint imports."""

from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from particula.execution import Backend, Device
from particula.execution.checkpoint import (
    CheckpointPayload,
    ResidentCheckpoint,
    _payload,
    _validate_payload,
    restart_resident_session,
)
from particula.execution.gpu_session import (
    ResidentDimensions,
    ResidentLifecycle,
    ResidentMetadata,
    ResidentSession,
    ResidentStepGuard,
)

if TYPE_CHECKING:
    from particula.execution.gpu_resources import GPUResourceRegistry


def _resident_binding() -> tuple[
    ResidentSession,
    "GPUResourceRegistry",
    ResidentStepGuard,
]:
    """Create a tiny active Warp CPU session, registry, and closed guard."""
    wp = pytest.importorskip("warp")
    from particula.execution.gpu_resources import GPUResourceRegistry
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    particles = WarpParticleData()
    particles.masses = wp.array([[[1.0]]], dtype=wp.float64, device="cpu")
    particles.concentration = wp.array([[2.0]], dtype=wp.float64, device="cpu")
    particles.charge = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    particles.density = wp.array([1000.0], dtype=wp.float64, device="cpu")
    particles.volume = wp.array([1.0], dtype=wp.float64, device="cpu")
    gas = WarpGasData()
    gas.molar_mass = wp.array([0.018], dtype=wp.float64, device="cpu")
    gas.concentration = wp.array([[3.0]], dtype=wp.float64, device="cpu")
    gas.vapor_pressure = wp.array([[42.0]], dtype=wp.float64, device="cpu")
    gas.partitioning = wp.array([[1]], dtype=wp.int32, device="cpu")
    environment = WarpEnvironmentData()
    environment.temperature = wp.array([298.15], dtype=wp.float64, device="cpu")
    environment.pressure = wp.array([101325.0], dtype=wp.float64, device="cpu")
    environment.saturation_ratio = wp.array(
        [[0.5]], dtype=wp.float64, device="cpu"
    )
    session = ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(1, 1, 1),
        ResidentMetadata(Device(Backend.WARP, "cpu"), ("water",)),
        ResidentLifecycle.ACTIVE,
    )
    registry = GPUResourceRegistry(session)
    return session, registry, ResidentStepGuard(session, registry)


def test_checkpoint_payload_is_frozen() -> None:
    """Payload records retain canonical immutable bytes."""
    payload = CheckpointPayload(
        "gas", "vapor_pressure", "<f8", (1,), b"12345678"
    )
    with pytest.raises(AttributeError):
        payload.role = "other"  # type: ignore[misc]


def test_checkpoint_carrier_retains_uncoerced_fraction_time() -> None:
    """Checkpoint metadata permits the guard's finite rational time value."""
    checkpoint = ResidentCheckpoint(
        1,
        "ResidentSession",
        object(),
        object(),
        (),
        0,
        Fraction(1, 3),
        ResidentLifecycle.ACTIVE,
        object(),
        object(),
        object(),
        (),
    )
    assert checkpoint.simulated_time == Fraction(1, 3)


def test_checkpoint_carrier_is_frozen() -> None:
    """Checkpoint records cannot have lifecycle metadata reassigned."""
    checkpoint = ResidentCheckpoint(
        1,
        "ResidentSession",
        object(),
        object(),
        (),
        0,
        Fraction(0, 1),
        ResidentLifecycle.ACTIVE,
        object(),
        object(),
        object(),
        (),
    )

    with pytest.raises(AttributeError):
        checkpoint.lifecycle = ResidentLifecycle.FINALIZED  # type: ignore[misc]


def test_payload_copies_a_contiguous_immutable_array_representation() -> None:
    """Payload capture preserves dtype, shape, and values independently."""

    class _Array:
        def numpy(self) -> np.ndarray:
            return np.array([[1, 2]], dtype=np.int32)

    payload = _payload("gas", "partitioning", _Array())

    assert payload.family == "gas"
    assert payload.role == "partitioning"
    assert payload.dtype == np.dtype(np.int32).str
    assert payload.shape == (1, 2)
    assert payload.data == np.array([[1, 2]], dtype=np.int32).tobytes()
    assert type(payload.data) is bytes


@pytest.mark.parametrize(
    ("payload", "exception", "match"),
    [
        (
            CheckpointPayload("gas", "vapor_pressure", "<f8", (1,), b"bad"),
            ValueError,
            "byte length",
        ),
        (
            CheckpointPayload("gas", "vapor_pressure", "not-a-dtype", (), b""),
            ValueError,
            "dtype",
        ),
        (
            CheckpointPayload("gas", "vapor_pressure", "<f8", (True,), b""),
            ValueError,
            "shape",
        ),
        (
            CheckpointPayload("gas", "vapor_pressure", "<f8", (), bytearray()),
            TypeError,
            "immutable bytes",
        ),
        (
            CheckpointPayload("gas", "vapor_pressure", "<f8", (), b"", 0),
            ValueError,
            "capacity",
        ),
    ],
)
def test_validate_payload_rejects_malformed_immutable_descriptors(
    payload: CheckpointPayload,
    exception: type[Exception],
    match: str,
) -> None:
    """Payload validation rejects malformed metadata before restart setup."""
    with pytest.raises(exception, match=match):
        _validate_payload(payload)


@pytest.mark.warp
def test_checkpoint_is_detached_and_preserves_active_session() -> None:
    """A checkpoint copies primaries while leaving its source usable and active."""
    session, registry, guard = _resident_binding()
    masses = cast(Any, session.particles).masses

    checkpoint = session.checkpoint(registry, guard)

    assert checkpoint.lifecycle is ResidentLifecycle.ACTIVE
    assert session.lifecycle is ResidentLifecycle.ACTIVE
    assert checkpoint.dimensions == ResidentDimensions(1, 1, 1)
    assert checkpoint.gas_names == ("water",)
    assert checkpoint.particles is not session.particles
    assert checkpoint.gas is not session.gas
    assert checkpoint.environment is not session.environment
    assert [item.role for item in checkpoint.payloads] == [
        "masses",
        "concentration",
        "charge",
        "density",
        "volume",
        "molar_mass",
        "concentration",
        "vapor_pressure",
        "partitioning",
        "temperature",
        "pressure",
        "saturation_ratio",
    ]
    np.testing.assert_array_equal(masses.numpy(), np.array([[[1.0]]]))
    np.testing.assert_array_equal(
        checkpoint.gas.concentration, np.array([[3.0]])
    )


@pytest.mark.warp
def test_finalize_is_cached_and_restart_restores_primary_state() -> None:
    """Finalization caches one snapshot and restart creates independent state."""
    session, registry, guard = _resident_binding()

    finalized = session.finalize(registry, guard)
    restored, restored_registry, restored_guard = restart_resident_session(
        finalized,
        Device(Backend.WARP, "cpu"),
    )

    assert session.finalize(registry, guard) is finalized
    assert session.lifecycle is ResidentLifecycle.FINALIZED
    assert restored is not session
    assert restored_registry is not registry
    assert restored_guard is not guard
    assert restored.metadata.gas_names == ("water",)
    np.testing.assert_array_equal(
        cast(Any, restored.particles).masses.numpy(),
        cast(Any, session.particles).masses.numpy(),
    )
    np.testing.assert_array_equal(
        cast(Any, restored.gas).vapor_pressure.numpy(),
        cast(Any, session.gas).vapor_pressure.numpy(),
    )


@pytest.mark.warp
def test_restart_restores_each_acquired_resource_family() -> None:
    """Restart reconstructs all resource families through registry APIs."""
    session, registry, guard = _resident_binding()
    registry.acquire_condensation()
    registry.acquire_coagulation(1)
    registry.acquire_wall_loss()
    registry.acquire_nucleation()

    checkpoint = session.checkpoint(registry, guard)
    _, restored_registry, _ = restart_resident_session(
        checkpoint, Device(Backend.WARP, "cpu")
    )

    assert tuple(restored_registry._bindings) == (
        "condensation",
        "coagulation",
        "wall_loss",
        "nucleation",
    )
    assert restored_registry._capacities["coagulation"] == 1


def test_restart_rejects_non_checkpoint_and_terminal_checkpoint() -> None:
    """Restart rejects invalid carriers before importing or allocating Warp."""
    with pytest.raises(TypeError, match="exact ResidentCheckpoint"):
        restart_resident_session(
            cast(ResidentCheckpoint, object()), Device(Backend.WARP, "cpu")
        )

    terminal = ResidentCheckpoint(
        1,
        "ResidentSession",
        object(),
        Device(Backend.WARP, "cpu"),
        (),
        0,
        Fraction(0, 1),
        ResidentLifecycle.FINALIZED,
        object(),
        object(),
        object(),
        (),
    )
    with pytest.raises(ValueError, match="active session"):
        restart_resident_session(terminal, Device(Backend.WARP, "cpu"))


@pytest.mark.warp
def test_restart_rejects_duplicate_payload_before_setup() -> None:
    """Restart rejects non-unique canonical descriptors before upload setup."""
    session, registry, guard = _resident_binding()
    checkpoint = session.checkpoint(registry, guard)
    malformed = replace(
        checkpoint, payloads=checkpoint.payloads + (checkpoint.payloads[0],)
    )

    with pytest.raises(ValueError, match="unique"):
        restart_resident_session(malformed, Device(Backend.WARP, "cpu"))


@pytest.mark.warp
def test_checkpoint_rejects_open_step_without_mutating_session() -> None:
    """Checkpoint performs its lifecycle gate before readback activity."""
    session, registry, guard = _resident_binding()
    token = guard.begin_step(Fraction(1, 2))

    with pytest.raises(RuntimeError, match="open"):
        session.checkpoint(registry, guard)

    assert session.lifecycle is ResidentLifecycle.ACTIVE
    guard.complete_step(token)
