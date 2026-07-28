"""Tests for concrete, non-dispatching Brownian coagulation carriers."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

import particula.execution as execution
import particula.execution.adapters as adapters
from particula.aerosol import Aerosol
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    CPUCoagulationResult,
    CPUCoagulationState,
    WarpBrownianCoagulationResult,
    WarpBrownianCoagulationState,
    _available_fields,
    _dtype_itemsize,
    _memory_range,
    _overlaps,
    _validate_ownership,
)


def _aerosol() -> Aerosol:
    """Create an unbuilt aerosol to prove carriers do not inspect fields."""
    return object.__new__(Aerosol)


def test_cpu_carriers_retain_identity_and_are_frozen() -> None:
    """Test CPU carriers preserve uninspected caller-owned resources."""
    config = BrownianCoagulationConfig()
    aerosol = _aerosol()
    state = CPUCoagulationState(config, aerosol)
    result = CPUCoagulationResult(state, aerosol)

    assert state.config is config
    assert state.backend_payload is aerosol
    assert result.state is state
    assert result.backend_payload is aerosol
    assert state != CPUCoagulationState(config, aerosol)
    with pytest.raises(FrozenInstanceError):
        state.aerosol = aerosol  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, object()])
def test_cpu_state_rejects_invalid_inputs_in_order(value: object) -> None:
    """Test the exact config boundary precedes aerosol validation."""
    with pytest.raises(
        TypeError, match="^config must be a BrownianCoagulationConfig.$"
    ):
        CPUCoagulationState(value, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="^aerosol must be an Aerosol.$"):
        CPUCoagulationState(BrownianCoagulationConfig(), value)  # type: ignore[arg-type]


def test_cpu_result_requires_original_aerosol() -> None:
    """Test CPU result validates state, aerosol kind, then identity."""
    aerosol = _aerosol()
    state = CPUCoagulationState(BrownianCoagulationConfig(), aerosol)
    with pytest.raises(
        TypeError, match="^state must be a CPUCoagulationState.$"
    ):
        CPUCoagulationResult(object(), aerosol)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="^aerosol must be state.aerosol.$"):
        CPUCoagulationResult(state, _aerosol())


def test_cpu_carriers_require_exact_config_and_validate_result_kind() -> None:
    """Test CPU carrier boundaries retain exact config and aerosol checks."""

    class ConfigSubclass(BrownianCoagulationConfig):
        """Exercise the exact config-type boundary."""

    aerosol = _aerosol()
    state = CPUCoagulationState(BrownianCoagulationConfig(), aerosol)
    with pytest.raises(
        TypeError, match="^config must be a BrownianCoagulationConfig.$"
    ):
        CPUCoagulationState(ConfigSubclass(), aerosol)
    with pytest.raises(TypeError, match="^aerosol must be an Aerosol.$"):
        CPUCoagulationResult(state, object())  # type: ignore[arg-type]


def test_cpu_result_requires_exact_state_before_other_validation() -> None:
    """Test CPU result rejects state subclasses before inspecting aerosol."""

    class StateSubclass(CPUCoagulationState):
        """Exercise the exact state-type boundary."""

    aerosol = _aerosol()
    subclass_state = StateSubclass(BrownianCoagulationConfig(), aerosol)
    with pytest.raises(
        TypeError, match="^state must be a CPUCoagulationState.$"
    ):
        CPUCoagulationResult(subclass_state, object())  # type: ignore[arg-type]


def test_cpu_carriers_do_not_import_optional_warp_backend() -> None:
    """Test CPU-only construction loads neither Warp nor particula.gpu."""
    root = Path(__file__).parents[3]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(root), os.environ.get("PYTHONPATH")))
        )
    }
    script = """
import builtins
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "warp" or name.startswith("warp.") or name == "particula.gpu" or name.startswith("particula.gpu."):
        raise AssertionError(f"Unexpected optional backend import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from particula.aerosol import Aerosol
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig, CPUCoagulationResult, CPUCoagulationState,
)

aerosol = object.__new__(Aerosol)
state = CPUCoagulationState(BrownianCoagulationConfig(), aerosol)
assert CPUCoagulationResult(state, aerosol).backend_payload is aerosol
assert not any(
    name == "warp" or name.startswith("warp.")
    or name == "particula.gpu" or name.startswith("particula.gpu.")
    for name in sys.modules
)
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_execution_exports_stay_narrow_and_concrete_carriers_stay_private() -> (
    None
):
    """Test the public execution package stays carrier-free."""
    assert execution.__all__ == [
        "Backend",
        "Device",
        "Process",
        "Capability",
        "CapabilityRequirements",
        "CapabilityDeclaration",
        "CapabilityMatrix",
        "ExecutionRequest",
        "ExecutionAdapter",
        "ExecutionContext",
    ]
    for name in (
        "BrownianCoagulationConfig",
        "CPUCoagulationState",
        "CPUCoagulationResult",
        "WarpBrownianCoagulationState",
        "WarpBrownianCoagulationResult",
    ):
        assert not hasattr(execution, name)
        assert not hasattr(adapters, name)


def test_metadata_helpers_detect_only_supported_contiguous_ranges() -> None:
    """Test metadata-only helpers cover usable, empty, and invalid storage."""

    class WarpTypes:
        """Minimal Warp dtype namespace for helper-only tests."""

        float64 = object()
        int32 = object()
        uint32 = object()

    class Metadata:
        """Minimal contiguous float64 metadata."""

        dtype = WarpTypes.float64
        shape = (2, 3)
        strides = (24, 8)
        ptr = 8
        capacity = 48

    class EmptyMetadata:
        """Metadata for storage with no usable range."""

        dtype = WarpTypes.float64
        shape = (0,)
        strides = (8,)
        ptr = 8
        capacity = 48

    wp = WarpTypes()
    metadata = Metadata()
    assert _dtype_itemsize(wp.float64, wp) == 8
    assert _dtype_itemsize(wp.int32, wp) == 4
    assert _dtype_itemsize(object(), wp) is None
    assert _memory_range(metadata, wp) == (8, 56)
    assert _memory_range(EmptyMetadata(), wp) is None
    assert _overlaps((8, 16), (15, 24))
    assert not _overlaps((8, 16), (16, 24))
    assert not _overlaps(None, (8, 16))


def test_memory_range_defers_invalid_or_noncontiguous_metadata() -> None:
    """Test malformed metadata is deferred instead of inspected or coerced."""

    class WarpTypes:
        """Minimal Warp dtype namespace for helper-only tests."""

        float64 = object()
        int32 = object()
        uint32 = object()

    class Metadata:
        """Instance-configurable metadata stand-in."""

        def __init__(self, **attributes: object) -> None:
            self.dtype = WarpTypes.float64
            self.shape = (2,)
            self.strides = (8,)
            self.ptr = 8
            self.capacity = 16
            for name, value in attributes.items():
                setattr(self, name, value)

    wp = WarpTypes()
    invalid_cases = (
        Metadata(shape=[2]),
        Metadata(strides=(16,)),
        Metadata(ptr=0),
        Metadata(ptr=9),
        Metadata(capacity=8),
        Metadata(capacity=True),
        Metadata(dtype=object()),
    )
    assert all(_memory_range(value, wp) is None for value in invalid_cases)


def test_ownership_helpers_reject_identity_but_defer_unknown_metadata() -> None:
    """Test alias detection is strict for identity and permissive otherwise."""

    class WarpTypes:
        """Minimal Warp namespace for unknown-metadata ownership checks."""

        float64 = object()
        int32 = object()
        uint32 = object()

    class Resource:
        """Resource exposing one protected field."""

        field = object()

    wp = WarpTypes()
    protected = _available_fields(Resource(), ("field", "absent"))
    assert len(protected) == 1
    _validate_ownership(wp, protected, object(), object(), object())
    with pytest.raises(
        ValueError, match="^caller-owned Warp resources must not alias.$"
    ):
        _validate_ownership(wp, protected, protected[0], None, object())


def test_ownership_helpers_reject_detectable_range_overlaps() -> None:
    """Test writable sidecars cannot overlap protected or sibling storage."""

    class WarpTypes:
        """Minimal Warp dtype namespace for overlap tests."""

        float64 = object()
        int32 = object()
        uint32 = object()

    class Metadata:
        """Contiguous storage metadata with configurable byte range."""

        dtype = WarpTypes.float64
        shape = (2,)
        strides = (8,)
        capacity = 16

        def __init__(self, ptr: int) -> None:
            self.ptr = ptr

    wp = WarpTypes()
    protected = (Metadata(8),)
    with pytest.raises(
        ValueError, match="^caller-owned Warp resources must not alias.$"
    ):
        _validate_ownership(wp, protected, Metadata(16), None, Metadata(32))
    with pytest.raises(
        ValueError, match="^caller-owned Warp resources must not alias.$"
    ):
        _validate_ownership(wp, (), Metadata(8), Metadata(16), Metadata(32))


def _warp_particles() -> Any:
    """Create a metadata-valid resident particle struct on the Warp CPU."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import WarpParticleData

    particles = WarpParticleData()
    particles.masses = wp.ones((1, 2, 1), dtype=wp.float64, device="cpu")
    particles.concentration = wp.ones((1, 2), dtype=wp.float64, device="cpu")
    particles.charge = wp.zeros((1, 2), dtype=wp.float64, device="cpu")
    particles.density = wp.ones(1, dtype=wp.float64, device="cpu")
    particles.volume = wp.ones(1, dtype=wp.float64, device="cpu")
    return particles


def _environment() -> Any:
    """Create a metadata-valid resident environment struct on the Warp CPU."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import WarpEnvironmentData

    environment = WarpEnvironmentData()
    environment.temperature = wp.ones(1, dtype=wp.float64, device="cpu")
    environment.pressure = wp.ones(1, dtype=wp.float64, device="cpu")
    environment.saturation_ratio = wp.ones(
        (1, 1), dtype=wp.float64, device="cpu"
    )
    return environment


@pytest.mark.warp
def test_warp_state_retains_request_and_rng_intent_by_identity() -> None:
    """Test P2 preserves all opaque controls without executing Warp work."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    environment = _environment()
    rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    collision_pairs = wp.zeros((1, 2), dtype=wp.int32, device="cpu")
    n_collisions = wp.zeros(1, dtype=wp.int32, device="cpu")
    state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        None,
        None,
        object(),
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
        rng_seed=41,
        initialize_rng=True,
        environment=environment,
    )

    assert state.backend_payload is particles
    assert state.environment is environment
    assert state.rng_states is rng_states
    assert state.collision_pairs is collision_pairs
    assert state.n_collisions is n_collisions
    assert state.rng_seed == 41
    assert state.initialize_rng is True


@pytest.mark.warp
def test_warp_state_accepts_direct_form_and_rejects_aliases() -> None:
    """Test direct-form protection catches only detectable ownership aliases."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    temperature = wp.ones(1, dtype=wp.float64, device="cpu")
    pressure = wp.ones(1, dtype=wp.float64, device="cpu")
    rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        temperature,
        pressure,
        1.0,
        rng_states=rng_states,
    )
    assert state.temperature is temperature
    with pytest.raises(
        ValueError, match="^caller-owned Warp resources must not alias.$"
    ):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            particles,
            temperature,
            pressure,
            1.0,
            rng_states=particles.concentration,
        )


@pytest.mark.warp
def test_warp_state_validation_order_and_result_identity() -> None:
    """Test form/RNG precedence and future result identity constraints."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    environment = _environment()
    rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    with pytest.raises(
        ValueError,
        match="^provide either environment or both temperature and pressure.$",
    ):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(), particles, None, None, 1.0
        )
    with pytest.raises(ValueError, match="^rng_states must be supplied.$"):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            particles,
            None,
            None,
            1.0,
            environment=environment,
        )
    state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        None,
        None,
        1.0,
        rng_states=rng_states,
        environment=environment,
    )
    result = WarpBrownianCoagulationResult(state, particles, None, None)
    assert result.backend_payload is particles
    with pytest.raises(
        ValueError, match="^particles must be state.particles.$"
    ):
        WarpBrownianCoagulationResult(state, object(), None, None)


@pytest.mark.warp
def test_warp_result_requires_supplied_diagnostic_identities() -> None:
    """Test result diagnostics match only when the request supplied them."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    environment = _environment()
    rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    collision_pairs = wp.zeros((1, 2), dtype=wp.int32, device="cpu")
    n_collisions = wp.zeros(1, dtype=wp.int32, device="cpu")
    state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        None,
        None,
        1.0,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
        environment=environment,
    )
    result = WarpBrownianCoagulationResult(
        state, particles, collision_pairs, n_collisions
    )
    assert result.collision_pairs is collision_pairs
    with pytest.raises(
        ValueError, match="^collision_pairs must be state.collision_pairs.$"
    ):
        WarpBrownianCoagulationResult(state, particles, object(), n_collisions)
    with pytest.raises(
        ValueError, match="^n_collisions must be state.n_collisions.$"
    ):
        WarpBrownianCoagulationResult(
            state, particles, collision_pairs, object()
        )


@pytest.mark.warp
def test_memory_range_defers_malformed_metadata() -> None:
    """Test unknown or malformed metadata remains native validation work."""
    wp = pytest.importorskip("warp")

    class Metadata:
        """Minimal metadata stand-in."""

        dtype = wp.float64
        shape = [1]
        strides = (8,)
        ptr = 8
        capacity = 8

    assert _memory_range(Metadata(), wp) is None
