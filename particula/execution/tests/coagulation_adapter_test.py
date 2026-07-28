"""Unit tests for concrete-only Brownian coagulation carriers and adapters.

These tests verify selected-call forwarding and caller-owned resident-resource
boundaries. They do not compare CPU and Warp stochastic trajectories.
"""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import particula.execution as execution
import particula.execution.adapters as adapters
import particula.execution.adapters.coagulation as coagulation_adapter
from particula.aerosol import Aerosol
from particula.dynamics import Coagulation
from particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy import (
    BrownianCoagulationStrategy,
)
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    CPUCoagulationExecutionAdapter,
    CPUCoagulationExecutionState,
    CPUCoagulationResult,
    CPUCoagulationState,
    WarpBrownianCoagulationExecutionAdapter,
    WarpBrownianCoagulationExecutionState,
    WarpBrownianCoagulationResult,
    WarpBrownianCoagulationState,
    _available_fields,
    _dtype_itemsize,
    _memory_range,
    _overlaps,
    _validate_ownership,
)


@dataclass(frozen=True)
class _RequestedCoagulationMechanism:
    """Represent a test-only request for a non-Brownian mechanism."""

    mechanisms: tuple[str, ...]
    distribution_type: str = "particle_resolved"


class _BrownianMarkerSubclass(BrownianCoagulationConfig):
    """Exercise the exact Brownian marker boundary."""


_UNSUPPORTED_MARKERS = (
    _RequestedCoagulationMechanism(("charged_hard_sphere",)),
    _RequestedCoagulationMechanism(("sedimentation_sp2016",)),
    _RequestedCoagulationMechanism(("turbulent_shear_st1956",)),
    _RequestedCoagulationMechanism(("brownian", "charged_hard_sphere")),
    _RequestedCoagulationMechanism(("unknown",)),
    _RequestedCoagulationMechanism(("brownian",), "discrete"),
    _BrownianMarkerSubclass(),
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


@pytest.mark.parametrize("marker", _UNSUPPORTED_MARKERS)
def test_cpu_p2_rejects_non_brownian_requests_before_runnable(
    marker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CPU P2 rejects unsupported markers before runnable execution."""
    monkeypatch.setattr(
        Coagulation,
        "execute",
        lambda *_: pytest.fail("invalid marker must not invoke the runnable"),
    )

    with pytest.raises(
        TypeError, match="^config must be a BrownianCoagulationConfig.$"
    ):
        CPUCoagulationState(marker, _aerosol())  # type: ignore[arg-type]


@pytest.mark.parametrize("marker", _UNSUPPORTED_MARKERS)
def test_warp_p2_rejects_non_brownian_requests_before_resolver(
    marker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Warp P2 rejects unsupported markers before lazy resolution."""
    monkeypatch.setattr(
        coagulation_adapter,
        "_get_coagulation_step_gpu",
        lambda: pytest.fail("invalid marker must not resolve the kernel"),
    )

    with pytest.raises(
        TypeError, match="^config must be a BrownianCoagulationConfig.$"
    ):
        WarpBrownianCoagulationState(
            cast(BrownianCoagulationConfig, marker),
            object(),
            object(),
            None,
            object(),
            rng_states=None,
            environment=object(),
        )


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


def test_warp_p2_marker_validation_precedes_optional_warp_import() -> None:
    """Test invalid markers fail before Warp import and valid markers need Warp."""
    root = Path(__file__).parents[3]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(root), os.environ.get("PYTHONPATH")))
        )
    }
    script = """
import builtins

from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    WarpBrownianCoagulationState,
)

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "warp" or name.startswith("warp."):
        raise ModuleNotFoundError("blocked Warp import", name="warp")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
try:
    WarpBrownianCoagulationState(object(), object(), None, None, 1.0)
except TypeError as error:
    assert str(error) == "config must be a BrownianCoagulationConfig."
else:
    raise AssertionError("invalid marker did not reject")

try:
    WarpBrownianCoagulationState(
        BrownianCoagulationConfig(), object(), None, None, 1.0
    )
except RuntimeError as error:
    assert str(error) == (
        "Warp is required to construct WarpBrownianCoagulationState."
    )
else:
    raise AssertionError("valid marker did not attempt Warp import")
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
        float32 = object()
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
    assert _dtype_itemsize(wp.float32, wp) == 4
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
        float32 = object()
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
        float32 = object()
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
        float32 = object()
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


def test_ownership_helpers_reject_float32_direct_input_overlap() -> None:
    """Test float32 direct-input ranges reject overlapping sidecars."""

    class WarpTypes:
        """Minimal Warp dtype namespace for float32 ownership checks."""

        float64 = object()
        float32 = object()
        int32 = object()
        uint32 = object()

    class Metadata:
        """Contiguous float32 storage metadata with configurable pointer."""

        dtype = WarpTypes.float32
        shape = (2,)
        strides = (4,)
        capacity = 8

        def __init__(self, ptr: int) -> None:
            self.ptr = ptr

    wp = WarpTypes()
    direct_temperature = Metadata(8)
    non_overlapping_rng = Metadata(16)
    _validate_ownership(
        wp, (direct_temperature,), None, None, non_overlapping_rng
    )
    with pytest.raises(
        ValueError, match="^caller-owned Warp resources must not alias.$"
    ):
        _validate_ownership(wp, (direct_temperature,), None, None, Metadata(12))


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


def _resource_values(*resources: Any) -> list[np.ndarray]:
    """Copy caller-owned Warp arrays for a write-free assertion."""
    return [resource.numpy().copy() for resource in resources]


@pytest.mark.warp
def test_warp_state_construction_is_write_free_and_never_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test P2 construction neither changes resources nor enters Warp runtime work."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    environment = _environment()
    collision_pairs = wp.zeros((1, 2), dtype=wp.int32, device="cpu")
    n_collisions = wp.zeros(1, dtype=wp.int32, device="cpu")
    rng_states = wp.array([19], dtype=wp.uint32, device="cpu")
    resources = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
        collision_pairs,
        n_collisions,
        rng_states,
    )
    before = _resource_values(*resources)

    def fail_if_called(*_: object, **__: object) -> None:
        """Fail when state construction attempts runtime work."""
        raise AssertionError("state-only construction must not enter Warp work")

    monkeypatch.setattr(wp, "launch", fail_if_called)
    monkeypatch.setattr(wp, "copy", fail_if_called)
    monkeypatch.setattr(wp, "synchronize", fail_if_called)

    state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        None,
        None,
        1.0,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
        rng_seed=41,
        initialize_rng=True,
        environment=environment,
    )

    assert state.particles is particles
    for expected, resource in zip(before, resources, strict=True):
        np.testing.assert_array_equal(resource.numpy(), expected)


@pytest.mark.warp
def test_warp_state_preserves_p2_validation_order() -> None:
    """Test config, resource kind, form, time, RNG, then ownership order."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    environment = _environment()
    rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    resources = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
        rng_states,
    )
    before = _resource_values(*resources)
    invalid_config: Any = object()

    with pytest.raises(
        TypeError, match="^config must be a BrownianCoagulationConfig.$"
    ):
        WarpBrownianCoagulationState(invalid_config, object(), None, None, 1.0)
    with pytest.raises(
        TypeError, match="^particles must be a WarpParticleData.$"
    ):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(), object(), None, None, 1.0
        )
    with pytest.raises(
        TypeError, match="^environment must be a WarpEnvironmentData.$"
    ):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            particles,
            None,
            None,
            1.0,
            environment=object(),
        )
    with pytest.raises(
        ValueError,
        match="^provide either environment or both temperature and pressure.$",
    ):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(), particles, None, None, 1.0
        )
    for invalid_time, error in ((object(), TypeError), (-1.0, ValueError)):
        with pytest.raises(error, match="^time_step"):
            WarpBrownianCoagulationState(
                BrownianCoagulationConfig(),
                particles,
                None,
                None,
                invalid_time,
                rng_states=particles.concentration,
                environment=environment,
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
    with pytest.raises(
        ValueError, match="^caller-owned Warp resources must not alias.$"
    ):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            particles,
            None,
            None,
            1.0,
            rng_states=particles.concentration,
            environment=environment,
        )
    for expected, resource in zip(before, resources, strict=True):
        np.testing.assert_array_equal(resource.numpy(), expected)


@pytest.mark.warp
def test_warp_state_defers_native_schema_validation() -> None:
    """Test malformed native arrays remain opaque when they do not alias."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    temperature = wp.ones((1, 1), dtype=wp.float32, device="cpu")
    pressure = wp.ones((2,), dtype=wp.float32, device="cpu")
    opaque_rng_state = object()

    state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        temperature,
        pressure,
        1.0,
        rng_states=opaque_rng_state,
    )

    assert state.temperature is temperature
    assert state.pressure is pressure
    assert state.rng_states is opaque_rng_state


@pytest.mark.warp
def test_warp_state_rejects_required_sidecar_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test identity and metadata-detectable P2 sidecar aliases reject."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    environment = _environment()
    collision_pairs = wp.zeros((1, 2), dtype=wp.int32, device="cpu")
    _n_collisions = wp.zeros(1, dtype=wp.int32, device="cpu")
    rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    error = "^caller-owned Warp resources must not alias.$"

    for sidecar in (environment.temperature, particles.masses):
        with pytest.raises(ValueError, match=error):
            WarpBrownianCoagulationState(
                BrownianCoagulationConfig(),
                particles,
                None,
                None,
                1.0,
                rng_states=sidecar,
                environment=environment,
            )
    with pytest.raises(ValueError, match=error):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            particles,
            None,
            None,
            1.0,
            collision_pairs=collision_pairs,
            n_collisions=collision_pairs,
            rng_states=rng_states,
            environment=environment,
        )

    ranges = {environment.pressure: (8, 16), rng_states: (12, 20)}
    monkeypatch.setattr(
        coagulation_adapter,
        "_memory_range",
        lambda resource, _: ranges.get(resource),
    )
    with pytest.raises(ValueError, match=error):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            particles,
            None,
            None,
            1.0,
            rng_states=rng_states,
            environment=environment,
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    "sidecar_name", ("collision_pairs", "n_collisions", "rng_states")
)
def test_warp_environment_form_protects_explicit_volume_from_sidecars(
    sidecar_name: str,
) -> None:
    """Test an environment-form explicit volume cannot be a writable sidecar."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    environment = _environment()
    volume = wp.ones(1, dtype=wp.float64, device="cpu")
    resources = (particles.masses, environment.temperature, volume)
    before = _resource_values(*resources)
    sidecars: dict[str, object] = {
        "collision_pairs": None,
        "n_collisions": None,
        "rng_states": wp.zeros(1, dtype=wp.uint32, device="cpu"),
    }
    sidecars[sidecar_name] = volume

    with pytest.raises(
        ValueError, match="^caller-owned Warp resources must not alias.$"
    ):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            particles,
            None,
            None,
            1.0,
            volume=volume,
            collision_pairs=sidecars["collision_pairs"],
            n_collisions=sidecars["n_collisions"],
            rng_states=sidecars["rng_states"],
            environment=environment,
        )
    for expected, resource in zip(before, resources, strict=True):
        np.testing.assert_array_equal(resource.numpy(), expected)


@pytest.mark.warp
@pytest.mark.parametrize("initialize_rng", (0, 1, "true", None))
def test_warp_state_requires_exact_boolean_initialize_rng(
    initialize_rng: object,
) -> None:
    """Test truthy and falsy non-booleans cannot reset persistent RNG state."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    environment = _environment()
    rng_states = wp.array([19], dtype=wp.uint32, device="cpu")
    before = _resource_values(particles.masses, rng_states)

    with pytest.raises(TypeError, match="^initialize_rng must be a boolean.$"):
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            particles,
            None,
            None,
            1.0,
            rng_states=rng_states,
            initialize_rng=initialize_rng,
            environment=environment,
        )
    for expected, resource in zip(
        before, (particles.masses, rng_states), strict=True
    ):
        np.testing.assert_array_equal(resource.numpy(), expected)


@pytest.mark.warp
@pytest.mark.parametrize("initialize_rng", (False, True))
def test_warp_state_accepts_boolean_initialize_rng(
    initialize_rng: bool,
) -> None:
    """Test both exact boolean persistent-RNG controls remain valid."""
    wp = pytest.importorskip("warp")
    state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        _warp_particles(),
        None,
        None,
        1.0,
        rng_states=wp.zeros(1, dtype=wp.uint32, device="cpu"),
        initialize_rng=initialize_rng,
        environment=_environment(),
    )
    assert state.initialize_rng is initialize_rng


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
        1.0,
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


def _cpu_execution_state(
    time_step: object = 2.5,
    sub_steps: object = 3,
) -> tuple[CPUCoagulationExecutionState, Aerosol]:
    """Create an unconfigured exact runnable P3 state for spy-driven tests."""
    aerosol = _aerosol()
    state = CPUCoagulationState(BrownianCoagulationConfig(), aerosol)
    runnable = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    return CPUCoagulationExecutionState(
        state, time_step, sub_steps, runnable
    ), aerosol


def test_cpu_execution_state_retains_identity_and_exact_types() -> None:
    """Test CPU P3 construction is control-free and rejects subclasses."""
    state, aerosol = _cpu_execution_state(object(), object())
    assert state.backend_payload is aerosol
    assert state != _cpu_execution_state(object(), object())[0]
    with pytest.raises(FrozenInstanceError):
        state.time_step = 1.0  # type: ignore[misc]

    class P2Subclass(CPUCoagulationState):
        """Exercise the exact P2 boundary."""

    class RunnableSubclass(Coagulation):
        """Exercise the exact runnable boundary."""

    p2_subclass = P2Subclass(BrownianCoagulationConfig(), aerosol)
    runnable = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    with pytest.raises(TypeError, match="exact CPUCoagulationState"):
        CPUCoagulationExecutionState(p2_subclass, object(), object(), runnable)
    with pytest.raises(TypeError, match="exact Coagulation"):
        CPUCoagulationExecutionState(
            CPUCoagulationState(BrownianCoagulationConfig(), aerosol),
            object(),
            object(),
            object.__new__(RunnableSubclass),
        )


def test_cpu_execution_state_does_not_invoke_runnable_or_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CPU P3 construction leaves its opaque fields uninspected."""
    aerosol = _aerosol()
    runnable = object.__new__(Coagulation)
    controls = (object(), object())
    monkeypatch.setattr(
        Coagulation,
        "execute",
        lambda *_: pytest.fail("P3 construction must not invoke runnable"),
    )

    state = CPUCoagulationExecutionState(
        CPUCoagulationState(BrownianCoagulationConfig(), aerosol),
        *controls,
        runnable,
    )

    assert state.time_step is controls[0]
    assert state.sub_steps is controls[1]


def test_cpu_adapter_rejects_wrong_state_before_delegation() -> None:
    """Test CPU dispatch rejects a non-P3 state before any delegation."""
    with pytest.raises(TypeError, match="CPUCoagulationExecutionState"):
        CPUCoagulationExecutionAdapter().execute(object())  # type: ignore[arg-type]


def test_cpu_adapter_dispatches_once_and_wraps_typed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CPU dispatch preserves original controls and identity results."""
    state, aerosol = _cpu_execution_state()
    calls: list[tuple[object, ...]] = []

    def execute(self: Coagulation, *args: object) -> Aerosol:
        """Record the exact delegated call."""
        assert self is state.runnable
        calls.append(args)
        return aerosol

    monkeypatch.setattr(Coagulation, "execute", execute)
    result = CPUCoagulationExecutionAdapter().execute(state)
    assert calls == [(aerosol, state.time_step, state.sub_steps)]
    assert result.state is state
    assert result.mutation.scopes == frozenset({execution.MutationScope.STATE})
    assert isinstance(result.backend_result, execution.BackendResult)
    value = result.backend_result.value
    assert isinstance(value, CPUCoagulationResult)
    assert value.state is state.state
    assert value.aerosol is aerosol


@pytest.mark.parametrize(
    "strategy",
    (
        BrownianCoagulationStrategy("discrete"),
        object(),
    ),
)
def test_cpu_adapter_rejects_unsupported_runnable_before_delegation(
    monkeypatch: pytest.MonkeyPatch,
    strategy: object,
) -> None:
    """Test CPU dispatch accepts only Brownian particle-resolved runnables."""
    aerosol = _aerosol()
    request = CPUCoagulationExecutionState(
        CPUCoagulationState(BrownianCoagulationConfig(), aerosol),
        1.0,
        1,
        Coagulation(cast(Any, strategy)),
    )
    monkeypatch.setattr(
        Coagulation,
        "execute",
        lambda *_: pytest.fail("unsupported runnable must not execute"),
    )

    with pytest.raises(
        ValueError,
        match="^runnable must use Brownian particle_resolved coagulation.$",
    ):
        CPUCoagulationExecutionAdapter().execute(request)


@pytest.mark.parametrize(
    ("time_step", "sub_steps", "error"),
    [
        (True, 1, TypeError),
        (object(), 1, TypeError),
        (-1.0, 1, ValueError),
        (float("inf"), 1, ValueError),
        (float("nan"), 1, ValueError),
        (1.0, True, ValueError),
        (1.0, 1.5, ValueError),
        (1.0, 0, ValueError),
        (1.0, -1, ValueError),
    ],
)
def test_cpu_adapter_rejects_invalid_controls_before_delegation(
    monkeypatch: pytest.MonkeyPatch,
    time_step: object,
    sub_steps: object,
    error: type[Exception],
) -> None:
    """Test local control errors are call-free."""
    state, _ = _cpu_execution_state(time_step, sub_steps)
    monkeypatch.setattr(
        Coagulation,
        "execute",
        lambda *_: pytest.fail("invalid controls must not delegate"),
    )
    with pytest.raises(error):
        CPUCoagulationExecutionAdapter().execute(state)


def test_cpu_adapter_propagates_and_rejects_delegate_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CPU delegate errors and replacement results receive no recovery."""
    state, _ = _cpu_execution_state()
    calls = 0

    def replacement(*_: object) -> Aerosol:
        """Return an invalid replacement aerosol exactly once."""
        nonlocal calls
        calls += 1
        return _aerosol()

    monkeypatch.setattr(Coagulation, "execute", replacement)
    with pytest.raises(ValueError, match="aerosol must be state.aerosol"):
        CPUCoagulationExecutionAdapter().execute(state)
    assert calls == 1
    failure = RuntimeError("delegate failure")
    monkeypatch.setattr(
        Coagulation, "execute", lambda *_: (_ for _ in ()).throw(failure)
    )
    with pytest.raises(RuntimeError) as error:
        CPUCoagulationExecutionAdapter().execute(state)
    assert error.value is failure


@pytest.mark.warp
def test_warp_execution_state_is_exact_and_resource_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Warp P3 construction neither resolves a kernel nor inspects state."""
    wp = pytest.importorskip("warp")
    p2_state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        _warp_particles(),
        None,
        None,
        1.0,
        rng_states=wp.zeros(1, dtype=wp.uint32, device="cpu"),
        environment=_environment(),
    )
    monkeypatch.setattr(
        coagulation_adapter,
        "_get_coagulation_step_gpu",
        lambda: pytest.fail("P3 construction must not resolve the kernel"),
    )
    state = WarpBrownianCoagulationExecutionState(p2_state)
    assert state.backend_payload is p2_state.particles
    with pytest.raises(FrozenInstanceError):
        state.state = p2_state  # type: ignore[misc]

    class P2Subclass(WarpBrownianCoagulationState):
        """Exercise exact P2 state rejection."""

    subclass = P2Subclass(
        BrownianCoagulationConfig(),
        _warp_particles(),
        None,
        None,
        1.0,
        rng_states=wp.zeros(1, dtype=wp.uint32, device="cpu"),
        environment=_environment(),
    )
    with pytest.raises(TypeError, match="exact WarpBrownianCoagulationState"):
        WarpBrownianCoagulationExecutionState(subclass)


@pytest.mark.warp
@pytest.mark.parametrize("use_environment", [False, True])
def test_warp_adapter_forwards_p2_resources_once(
    monkeypatch: pytest.MonkeyPatch,
    use_environment: bool,
) -> None:
    """Test Warp dispatch forwards direct and environment P2 forms unchanged."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    temperature = (
        None if use_environment else wp.ones(1, dtype=wp.float64, device="cpu")
    )
    pressure = (
        None if use_environment else wp.ones(1, dtype=wp.float64, device="cpu")
    )
    environment = _environment() if use_environment else None
    collision_pairs = wp.zeros((1, 2), dtype=wp.int32, device="cpu")
    n_collisions = wp.zeros(1, dtype=wp.int32, device="cpu")
    rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    p2_state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        temperature,
        pressure,
        1.0,
        volume=object(),
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
        rng_seed=41,
        initialize_rng=True,
        environment=environment,
    )
    state = WarpBrownianCoagulationExecutionState(p2_state)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    resolver_calls = 0

    def kernel(
        *args: object, **kwargs: object
    ) -> tuple[object, object, object]:
        """Record one native call and return supplied identities."""
        calls.append((args, kwargs))
        return particles, collision_pairs, n_collisions

    def resolve() -> Any:
        """Record the sole native resolver call."""
        nonlocal resolver_calls
        resolver_calls += 1
        return kernel

    monkeypatch.setattr(
        coagulation_adapter, "_get_coagulation_step_gpu", resolve
    )
    result = WarpBrownianCoagulationExecutionAdapter().execute(state)
    assert resolver_calls == 1
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (
        particles,
        temperature,
        pressure,
        p2_state.time_step,
        p2_state.volume,
    )
    assert kwargs == {
        "rng_seed": 41,
        "collision_pairs": collision_pairs,
        "n_collisions": n_collisions,
        "rng_states": rng_states,
        "initialize_rng": True,
        "environment": environment,
    }
    assert result.state is state
    assert result.mutation.scopes == frozenset({execution.MutationScope.STATE})
    assert result.backend_result is not None
    assert isinstance(
        result.backend_result.value, WarpBrownianCoagulationResult
    )
    warp_result = result.backend_result.value
    assert isinstance(warp_result, WarpBrownianCoagulationResult)
    assert warp_result.particles is particles


@pytest.mark.warp
def test_warp_adapter_uses_no_conversion_or_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test selected Warp dispatch remains resident and avoids CPU fallback."""
    wp = pytest.importorskip("warp")
    from particula.gpu import conversion

    particles = _warp_particles()
    collision_pairs = wp.zeros((1, 2), dtype=wp.int32, device="cpu")
    n_collisions = wp.zeros(1, dtype=wp.int32, device="cpu")
    rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    p2_state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        None,
        None,
        1.0,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
        environment=_environment(),
    )

    def fail_if_called(*_: object, **__: object) -> None:
        """Fail when a forbidden conversion, sync, or fallback is used."""
        pytest.fail("Warp adapter must not convert, synchronize, or fall back")

    for name in (
        "from_warp_particle_data",
        "from_warp_gas_data",
        "from_warp_environment_data",
        "to_warp_particle_data",
        "to_warp_gas_data",
        "to_warp_environment_data",
    ):
        monkeypatch.setattr(conversion, name, fail_if_called)
    monkeypatch.setattr(wp, "synchronize", fail_if_called)
    monkeypatch.setattr(Coagulation, "execute", fail_if_called)
    monkeypatch.setattr(
        coagulation_adapter,
        "_get_coagulation_step_gpu",
        lambda: lambda *_args, **_kwargs: (
            particles,
            collision_pairs,
            n_collisions,
        ),
    )

    result = WarpBrownianCoagulationExecutionAdapter().execute(
        WarpBrownianCoagulationExecutionState(p2_state)
    )

    assert result.backend_result is not None
    warp_result = result.backend_result.value
    assert isinstance(warp_result, WarpBrownianCoagulationResult)
    assert warp_result.particles is particles


@pytest.mark.warp
def test_warp_adapter_rejects_before_resolution_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Warp wrong-state and resolver failures receive no fallback."""
    monkeypatch.setattr(
        coagulation_adapter,
        "_get_coagulation_step_gpu",
        lambda: pytest.fail("wrong state must not resolve"),
    )
    with pytest.raises(
        TypeError, match="WarpBrownianCoagulationExecutionState"
    ):
        WarpBrownianCoagulationExecutionAdapter().execute(object())  # type: ignore[arg-type]
    failure = RuntimeError("resolver failure")
    monkeypatch.setattr(
        coagulation_adapter,
        "_get_coagulation_step_gpu",
        lambda: (_ for _ in ()).throw(failure),
    )
    wp = pytest.importorskip("warp")
    p2_state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        _warp_particles(),
        None,
        None,
        1.0,
        rng_states=wp.zeros(1, dtype=wp.uint32, device="cpu"),
        environment=_environment(),
    )
    with pytest.raises(RuntimeError) as error:
        WarpBrownianCoagulationExecutionAdapter().execute(
            WarpBrownianCoagulationExecutionState(p2_state)
        )
    assert error.value is failure


@pytest.mark.warp
def test_warp_adapter_rejects_replaced_results_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test result identity failures do not trigger kernel recovery or retry."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    collision_pairs = wp.zeros((1, 2), dtype=wp.int32, device="cpu")
    n_collisions = wp.zeros(1, dtype=wp.int32, device="cpu")
    p2_state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        None,
        None,
        1.0,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=wp.zeros(1, dtype=wp.uint32, device="cpu"),
        environment=_environment(),
    )
    calls = 0
    resolver_calls = 0

    def kernel(*_: object, **__: object) -> tuple[object, object, object]:
        """Return a replacement particle object after one selected call."""
        nonlocal calls
        calls += 1
        return object(), collision_pairs, n_collisions

    def resolve() -> Any:
        """Record native resolution before returning the invalid result."""
        nonlocal resolver_calls
        resolver_calls += 1
        return kernel

    monkeypatch.setattr(
        coagulation_adapter, "_get_coagulation_step_gpu", resolve
    )
    with pytest.raises(ValueError, match="particles must be state.particles"):
        WarpBrownianCoagulationExecutionAdapter().execute(
            WarpBrownianCoagulationExecutionState(p2_state)
        )
    assert calls == 1
    assert resolver_calls == 1


@pytest.mark.warp
def test_warp_adapter_rejects_replaced_diagnostics_and_reuses_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test identity failures and repeated calls preserve RNG reuse intent."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    collision_pairs = wp.zeros((1, 2), dtype=wp.int32, device="cpu")
    n_collisions = wp.zeros(1, dtype=wp.int32, device="cpu")
    rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    p2_state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        None,
        None,
        1.0,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
        rng_seed=41,
        initialize_rng=False,
        environment=_environment(),
    )
    calls: list[dict[str, object]] = []

    def kernel(*_: object, **kwargs: object) -> tuple[object, object, object]:
        """Record persistent RNG handoff and return selected identities."""
        calls.append(kwargs)
        return particles, collision_pairs, n_collisions

    monkeypatch.setattr(
        coagulation_adapter,
        "_get_coagulation_step_gpu",
        lambda: kernel,
    )
    adapter = WarpBrownianCoagulationExecutionAdapter()
    state = WarpBrownianCoagulationExecutionState(p2_state)
    adapter.execute(state)
    adapter.execute(state)
    assert len(calls) == 2
    assert all(call["rng_states"] is rng_states for call in calls)
    assert all(call["initialize_rng"] is False for call in calls)

    monkeypatch.setattr(
        coagulation_adapter,
        "_get_coagulation_step_gpu",
        lambda: lambda *_args, **_kwargs: (
            particles,
            object(),
            n_collisions,
        ),
    )
    with pytest.raises(ValueError, match="collision_pairs must be state"):
        adapter.execute(state)


@pytest.mark.warp
@pytest.mark.parametrize(
    ("label", "failure"),
    [
        ("particle_environment_volume", ValueError("native particle failure")),
        ("output_capacity_dtype_device", ValueError("native output failure")),
        ("rng_schema", ValueError("native RNG failure")),
    ],
)
def test_warp_adapter_propagates_native_sentinels_once(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    failure: Exception,
) -> None:
    """Test selected native exceptions propagate without adapter recovery."""
    wp = pytest.importorskip("warp")
    p2_state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        _warp_particles(),
        None,
        None,
        1.0,
        rng_states=wp.zeros(1, dtype=wp.uint32, device="cpu"),
        environment=_environment(),
    )
    calls = 0
    resolver_calls = 0

    def kernel(*_: object, **__: object) -> tuple[object, object, object]:
        """Raise a sentinel error after exactly one native invocation."""
        nonlocal calls
        calls += 1
        raise failure

    def resolve() -> Any:
        """Record native resolution before returning the sentinel kernel."""
        nonlocal resolver_calls
        resolver_calls += 1
        return kernel

    monkeypatch.setattr(
        coagulation_adapter, "_get_coagulation_step_gpu", resolve
    )
    with pytest.raises(type(failure)) as error:
        WarpBrownianCoagulationExecutionAdapter().execute(
            WarpBrownianCoagulationExecutionState(p2_state)
        )
    assert error.value is failure
    assert label in {
        "particle_environment_volume",
        "output_capacity_dtype_device",
        "rng_schema",
    }
    assert resolver_calls == 1
    assert calls == 1


@pytest.mark.warp
def test_warp_adapter_preserves_writer_mutation_after_native_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a writer failure leaves its caller-owned sidecar mutation intact."""
    wp = pytest.importorskip("warp")
    particles = _warp_particles()
    rng_states = wp.array([19], dtype=wp.uint32, device="cpu")
    p2_state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        particles,
        None,
        None,
        1.0,
        rng_states=rng_states,
        environment=_environment(),
    )
    failure = RuntimeError("writer failure")
    calls = 0
    resolver_calls = 0

    def writer(*_: object, **__: object) -> tuple[object, object, object]:
        """Mutate the supplied sidecar before raising the native sentinel."""
        nonlocal calls
        calls += 1
        wp.copy(
            rng_states,
            wp.array([101], dtype=wp.uint32, device="cpu"),
        )
        raise failure

    def resolve() -> Any:
        """Record native resolution before returning the writer sentinel."""
        nonlocal resolver_calls
        resolver_calls += 1
        return writer

    monkeypatch.setattr(
        coagulation_adapter, "_get_coagulation_step_gpu", resolve
    )
    with pytest.raises(RuntimeError) as error:
        WarpBrownianCoagulationExecutionAdapter().execute(
            WarpBrownianCoagulationExecutionState(p2_state)
        )

    assert error.value is failure
    assert resolver_calls == 1
    assert calls == 1
    np.testing.assert_array_equal(rng_states.numpy(), np.array([101]))
