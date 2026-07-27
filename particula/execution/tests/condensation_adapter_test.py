"""Tests for concrete, non-executing condensation state carriers."""

import builtins
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from particula.aerosol import Aerosol
from particula.execution import (
    CONDENSATION_PROCESS,
    Backend,
    CapabilityDeclaration,
    CapabilityMatrix,
    CondensationActivityMode,
    CondensationConfiguration,
    CondensationExecutionMode,
    CondensationSurfaceMode,
    Device,
    ExecutionContext,
    ExecutionRequest,
    MutationScope,
    get_condensation_requirements,
)
from particula.execution.adapters.condensation import (
    CondensationExecutionConfig,
    CPUCondensationExecutionAdapter,
    CPUCondensationExecutionState,
    CPUCondensationState,
    WarpCondensationExecutionAdapter,
    WarpCondensationExecutionState,
    _memory_range,
    _overlaps,
    _validate_array,
    _validate_output_ownership,
)


def _configuration() -> CondensationConfiguration:
    """Create valid semantic configuration metadata."""
    return CondensationConfiguration(
        CondensationExecutionMode.EQUAL_STEP,
        False,
        CondensationActivityMode.IDEAL,
        CondensationSurfaceMode.STATIC,
    )


def _aerosol() -> Aerosol:
    """Create a deliberately unbuilt aerosol for ownership-only P2 checks."""
    return object.__new__(Aerosol)


def test_cpu_carriers_retain_resources_by_identity_and_freeze() -> None:
    """Test CPU carriers retain caller resources without inspecting them."""
    configuration = _configuration()
    config = CondensationExecutionConfig(configuration)
    aerosol = _aerosol()
    state = CPUCondensationState(config, aerosol)

    assert config.configuration is configuration
    assert state.config is config
    assert state.aerosol is aerosol
    assert state.backend_payload is aerosol
    assert config != CondensationExecutionConfig(configuration)
    assert state != CPUCondensationState(config, aerosol)
    with pytest.raises(FrozenInstanceError):
        state.aerosol = aerosol  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, object()])
def test_config_rejects_nonexact_configuration(value: object) -> None:
    """Test config rejects missing or unrelated configuration values."""
    with pytest.raises(
        TypeError,
        match="^configuration must be an exact CondensationConfiguration.$",
    ):
        CondensationExecutionConfig(value)  # type: ignore[arg-type]


def test_config_rejects_configuration_subclass() -> None:
    """Test the P2 config boundary requires the exact configuration class."""

    class DerivedConfiguration(CondensationConfiguration):
        """Unsupported configuration subtype."""

    source = _configuration()
    derived = DerivedConfiguration(
        source.execution_mode,
        source.latent_heat,
        source.activity_mode,
        source.surface_mode,
    )
    with pytest.raises(
        TypeError,
        match="^configuration must be an exact CondensationConfiguration.$",
    ):
        CondensationExecutionConfig(derived)


@pytest.mark.parametrize("aerosol", [None, object()])
def test_cpu_state_rejects_non_aerosol_after_config(aerosol: object) -> None:
    """Test CPU state validates aerosol after its exact config carrier."""
    with pytest.raises(TypeError, match="^aerosol must be an Aerosol.$"):
        CPUCondensationState(
            CondensationExecutionConfig(_configuration()),
            cast(Aerosol, aerosol),
        )


def test_cpu_state_config_validation_precedes_aerosol_validation() -> None:
    """Test both-invalid input reports the configuration defect first."""
    with pytest.raises(
        TypeError,
        match="^config must be an exact CondensationExecutionConfig.$",
    ):
        CPUCondensationState(object(), object())  # type: ignore[arg-type]


def test_cpu_adapter_import_and_construction_do_not_load_optional_backend() -> (
    None
):
    """Test the concrete CPU carrier remains available without Warp imports."""
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
from particula.execution import (
    CondensationActivityMode, CondensationConfiguration,
    CondensationExecutionMode, CondensationSurfaceMode,
)
from particula.execution.adapters.condensation import (
    CPUCondensationState, CondensationExecutionConfig,
)

configuration = CondensationConfiguration(
    CondensationExecutionMode.EQUAL_STEP, False,
    CondensationActivityMode.IDEAL, CondensationSurfaceMode.STATIC,
)
aerosol = object.__new__(Aerosol)
state = CPUCondensationState(CondensationExecutionConfig(configuration), aerosol)
assert state.backend_payload is aerosol
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


class _Array:
    """Minimal metadata-only Warp-array stand-in for helper tests."""

    def __init__(
        self,
        ptr: int,
        shape: tuple[int, ...],
        strides: tuple[int, ...] | None,
        dtype: object = "float64",
        device: object = "cpu",
    ) -> None:
        self.ptr = ptr
        self.shape = shape
        self.strides = strides
        self.dtype = dtype
        self.device = device


def test_metadata_helpers_validate_schema_ranges_and_output_ownership() -> None:
    """Test private metadata helpers reject invalid and aliased writable data."""
    valid = _Array(100, (2, 1), (8, 8))
    _validate_array("value", valid, "float64", (2, 1), "cpu")
    assert _memory_range("value", valid, 8) == (100, 116)
    assert _memory_range("empty", _Array(0, (0,), (8,)), 8) is None
    assert _overlaps((100, 116), (108, 124))
    assert not _overlaps((100, 116), None)

    with pytest.raises(ValueError, match="^value must use dtype float64.$"):
        _validate_array(
            "value", _Array(0, (2, 1), (8, 8), "float32"), "float64", (2, 1)
        )
    with pytest.raises(ValueError, match=r"^value must have shape \(1, 2\)\.$"):
        _validate_array("value", valid, "float64", (1, 2))
    with pytest.raises(
        ValueError, match="^value device must match particles.masses device.$"
    ):
        _validate_array("value", valid, "float64", (2, 1), "cuda")
    with pytest.raises(ValueError, match="^value must be a Warp array.$"):
        _validate_array("value", object(), "float64", (2, 1))
    with pytest.raises(
        ValueError,
        match="^mass_transfer must be contiguous for ownership checks.$",
    ):
        _memory_range("mass_transfer", _Array(0, (2,), (16,)), 8)
    with pytest.raises(
        ValueError, match="^mass_transfer must not overlap primary state.$"
    ):
        _validate_output_ownership(valid, None, (valid,))
    with pytest.raises(
        ValueError, match="^mass_transfer must not overlap energy_transfer.$"
    ):
        _validate_output_ownership(
            _Array(200, (1,), (8,)), _Array(200, (1,), (8,)), ()
        )


def _warp_state_inputs() -> tuple[Any, Any, Any]:
    """Create a metadata-valid one-box resident Warp state lazily."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    particles = WarpParticleData()
    particles.masses = wp.ones((1, 2, 1), dtype=wp.float64, device="cpu")
    particles.concentration = wp.ones((1, 2), dtype=wp.float64, device="cpu")
    particles.charge = wp.zeros((1, 2), dtype=wp.float64, device="cpu")
    particles.density = wp.ones(1, dtype=wp.float64, device="cpu")
    particles.volume = wp.ones(1, dtype=wp.float64, device="cpu")
    gas = WarpGasData()
    gas.molar_mass = wp.ones(1, dtype=wp.float64, device="cpu")
    gas.concentration = wp.ones((1, 1), dtype=wp.float64, device="cpu")
    gas.vapor_pressure = wp.ones((1, 1), dtype=wp.float64, device="cpu")
    gas.partitioning = wp.ones((1, 1), dtype=wp.int32, device="cpu")
    environment = WarpEnvironmentData()
    environment.temperature = wp.ones(1, dtype=wp.float64, device="cpu")
    environment.pressure = wp.ones(1, dtype=wp.float64, device="cpu")
    environment.saturation_ratio = wp.ones(
        (1, 1), dtype=wp.float64, device="cpu"
    )
    return particles, gas, environment


def _snapshot_warp_resources(*resources: Any) -> list[tuple[int, np.ndarray]]:
    """Snapshot caller-owned Warp arrays while retaining their identities."""
    return [
        (id(resource), np.array(resource.numpy(), copy=True))
        for resource in resources
    ]


def _assert_warp_resources_unchanged(
    snapshot: list[tuple[int, np.ndarray]], *resources: Any
) -> None:
    """Assert caller-owned Warp arrays retain identity and values."""
    assert [id(resource) for resource in resources] == [
        identity for identity, _ in snapshot
    ]
    for (_, expected), resource in zip(snapshot, resources, strict=True):
        np.testing.assert_array_equal(resource.numpy(), expected)


def _primary_warp_resources(
    particles: Any, gas: Any, environment: Any
) -> tuple[Any, ...]:
    """Return all mutable primary arrays from a P2 Warp state."""
    return (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.concentration,
        gas.vapor_pressure,
        gas.partitioning,
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
    )


@pytest.mark.warp
def test_warp_state_retains_primary_and_opaque_resources_by_identity() -> None:
    """Test valid Warp metadata construction has no ownership conversion."""
    from particula.execution.adapters.condensation import WarpCondensationState

    particles, gas, environment = _warp_state_inputs()
    thermodynamics = object()
    activity = object()
    scratch = object()
    latent_heat = object()
    thermal_work = object()
    state = WarpCondensationState(
        CondensationExecutionConfig(_configuration()),
        particles,
        gas,
        environment,
        thermodynamics,
        activity,
        scratch,
        None,
        latent_heat,
        None,
        thermal_work,
    )

    payload = state.backend_payload
    assert payload[0] is particles
    assert payload[1] is gas
    assert payload[2] is environment
    assert state.thermodynamics is thermodynamics
    assert state.activity_surface is activity
    assert state.scratch_buffers is scratch
    assert state.latent_heat is latent_heat
    assert state.thermal_work is thermal_work


@pytest.mark.warp
def test_warp_state_does_not_launch_transfer_or_synchronize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test metadata construction never invokes Warp execution operations."""
    wp = pytest.importorskip("warp")
    from particula.execution.adapters.condensation import WarpCondensationState

    particles, gas, environment = _warp_state_inputs()

    def fail(*args: object, **kwargs: object) -> None:
        """Fail if construction attempts a Warp operation."""
        del args, kwargs
        raise AssertionError("Carrier construction must not execute Warp.")

    monkeypatch.setattr(wp, "synchronize", fail)
    monkeypatch.setattr(wp, "copy", fail)

    state = WarpCondensationState(
        CondensationExecutionConfig(_configuration()),
        particles,
        gas,
        environment,
        object(),
    )

    assert state.backend_payload == (particles, gas, environment)


@pytest.mark.warp
def test_warp_state_requires_thermodynamics_before_output_metadata() -> None:
    """Test opaque required thermodynamics precedes writable output checks."""
    from particula.execution.adapters.condensation import WarpCondensationState

    particles, gas, environment = _warp_state_inputs()
    with pytest.raises(ValueError, match="^thermodynamics must not be None.$"):
        WarpCondensationState(
            CondensationExecutionConfig(_configuration()),
            particles,
            gas,
            environment,
            None,
            mass_transfer=object(),
        )


@pytest.mark.warp
def test_warp_state_validates_config_and_primary_types_before_metadata() -> (
    None
):
    """Test Warp preflight reports earlier carrier and container defects first."""
    from particula.execution.adapters.condensation import WarpCondensationState

    with pytest.raises(
        TypeError,
        match="^config must be an exact CondensationExecutionConfig.$",
    ):
        WarpCondensationState(object(), object(), object(), object(), object())  # type: ignore[arg-type]

    with pytest.raises(
        TypeError, match="^particles must be a WarpParticleData.$"
    ):
        WarpCondensationState(
            CondensationExecutionConfig(_configuration()),
            object(),
            object(),
            object(),
            object(),
        )


@pytest.mark.warp
def test_warp_state_rejects_missing_mass_metadata_before_shape_access() -> None:
    """Test malformed mass metadata reports the documented preflight error."""
    from particula.execution.adapters.condensation import WarpCondensationState
    from particula.gpu.warp_types import WarpParticleData

    _, gas, environment = _warp_state_inputs()
    with pytest.raises(
        ValueError, match="^particles.masses must be a Warp array.$"
    ):
        WarpCondensationState(
            CondensationExecutionConfig(_configuration()),
            WarpParticleData(),
            gas,
            environment,
            object(),
        )


def test_warp_state_reports_only_missing_top_level_warp_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test internal Warp-type import failures propagate without relabeling."""
    from particula.execution.adapters.condensation import WarpCondensationState

    original_import = cast(Any, builtins.__import__)

    def missing_warp(name: str, *args: object, **kwargs: object) -> object:
        """Simulate an absent optional top-level Warp package."""
        if name == "warp":
            raise ModuleNotFoundError("No module named 'warp'", name="warp")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_warp)
    with pytest.raises(
        RuntimeError,
        match="^WarpCondensationState requires the optional Warp runtime.$",
    ):
        WarpCondensationState(
            CondensationExecutionConfig(_configuration()),
            object(),
            object(),
            object(),
            object(),
        )

    def broken_warp_types(name: str, *args: object, **kwargs: object) -> object:
        """Simulate a dependency failure internal to the Warp type module."""
        if name == "particula.gpu.warp_types":
            raise ImportError("broken Warp type dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_warp_types)
    with pytest.raises(ImportError, match="^broken Warp type dependency$"):
        WarpCondensationState(
            CondensationExecutionConfig(_configuration()),
            object(),
            object(),
            object(),
            object(),
        )


@pytest.mark.warp
def test_warp_state_validates_writable_outputs_and_ownership() -> None:
    """Test output schema and aliases reject before any future adapter launch."""
    wp = pytest.importorskip("warp")
    from particula.execution.adapters.condensation import WarpCondensationState

    particles, gas, environment = _warp_state_inputs()
    config = CondensationExecutionConfig(_configuration())
    mass_transfer = wp.zeros((1, 2, 1), dtype=wp.float64, device="cpu")
    energy_transfer = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    state = WarpCondensationState(
        config,
        particles,
        gas,
        environment,
        object(),
        mass_transfer=mass_transfer,
        energy_transfer=energy_transfer,
    )
    assert state.mass_transfer is mass_transfer
    assert state.energy_transfer is energy_transfer

    with pytest.raises(
        ValueError,
        match=r"^mass_transfer must have shape \(1, 2, 1\).$",
    ):
        WarpCondensationState(
            config,
            particles,
            gas,
            environment,
            object(),
            mass_transfer=wp.zeros((1, 1, 1), dtype=wp.float64, device="cpu"),
        )
    with pytest.raises(
        ValueError,
        match="^mass_transfer must not overlap primary state.$",
    ):
        WarpCondensationState(
            config,
            particles,
            gas,
            environment,
            object(),
            mass_transfer=particles.masses,
        )
    with pytest.raises(
        ValueError,
        match="^energy_transfer must not overlap primary state.$",
    ):
        WarpCondensationState(
            config,
            particles,
            gas,
            environment,
            object(),
            energy_transfer=gas.concentration,
        )


@pytest.mark.warp
def test_warp_state_accepts_output_adjacent_to_int32_partitioning() -> None:
    """Test an output beside int32 primary storage is not an ownership alias."""
    wp = pytest.importorskip("warp")
    from particula.execution.adapters.condensation import WarpCondensationState

    particles, gas, environment = _warp_state_inputs()
    backing = wp.zeros(3, dtype=wp.float64, device="cpu")
    gas.partitioning = wp.array(
        ptr=backing.ptr,
        capacity=backing.capacity,
        dtype=wp.int32,
        shape=(1, 1),
        strides=(4, 4),
        device="cpu",
        copy=False,
    )
    mass_transfer = wp.array(
        ptr=backing.ptr + 4,
        capacity=backing.capacity - 4,
        dtype=wp.float64,
        shape=(1, 2, 1),
        strides=(16, 8, 8),
        device="cpu",
        copy=False,
    )

    state = WarpCondensationState(
        CondensationExecutionConfig(_configuration()),
        particles,
        gas,
        environment,
        object(),
        mass_transfer=mass_transfer,
    )

    assert cast(Any, state.gas).partitioning is gas.partitioning
    assert state.mass_transfer is mass_transfer


def test_cpu_execution_state_and_adapter_dispatch_once_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test selected CPU dispatch forwards uncoerced controls exactly once."""
    from particula.dynamics import MassCondensation

    aerosol = _aerosol()
    p2_state = CPUCondensationState(
        CondensationExecutionConfig(_configuration()), aerosol
    )
    runnable = object.__new__(MassCondensation)
    time_step = np.float64(2.0)
    sub_steps = np.int64(3)
    state = CPUCondensationExecutionState(
        p2_state, time_step, sub_steps, runnable
    )
    calls: list[tuple[object, object, object]] = []

    def execute(
        self: MassCondensation,
        received_aerosol: Aerosol,
        received_time_step: object,
        received_sub_steps: object,
    ) -> Aerosol:
        """Record the exact forwarded request and return its aerosol."""
        assert self is runnable
        calls.append((received_aerosol, received_time_step, received_sub_steps))
        return received_aerosol

    monkeypatch.setattr(MassCondensation, "execute", execute)
    result = CPUCondensationExecutionAdapter().execute(state)

    assert state.backend_payload is aerosol
    assert calls == [(aerosol, time_step, sub_steps)]
    assert result.state is state
    assert result.backend_result is not None
    assert result.backend_result.value is aerosol
    assert result.metadata == ()
    assert result.mutation.scopes == frozenset({MutationScope.STATE})


@pytest.mark.parametrize("sub_steps", [True, 0, -1, 1.5, object()])
def test_cpu_execution_adapter_rejects_invalid_sub_steps_before_delegate(
    sub_steps: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invalid sub-step controls cannot invoke the selected runnable."""
    from particula.dynamics import MassCondensation

    runnable = object.__new__(MassCondensation)
    calls = 0

    def execute(*args: object, **kwargs: object) -> Aerosol:
        """Fail if invalid controls reach the delegated runnable."""
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("invalid controls must not reach the runnable")

    monkeypatch.setattr(MassCondensation, "execute", execute)
    state = CPUCondensationExecutionState(
        CPUCondensationState(
            CondensationExecutionConfig(_configuration()), _aerosol()
        ),
        1.0,
        sub_steps,
        runnable,
    )

    with pytest.raises(
        ValueError, match="^sub_steps must be a positive integer.$"
    ):
        CPUCondensationExecutionAdapter().execute(state)

    assert calls == 0


def test_cpu_execution_adapter_rejects_semantic_latent_heat_before_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the isothermal boundary rejects latent heat before delegation."""
    from particula.dynamics import MassCondensation

    runnable = object.__new__(MassCondensation)
    calls = 0

    def execute(*args: object, **kwargs: object) -> Aerosol:
        """Fail if rejected semantic state reaches the runnable."""
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("latent heat must be rejected before delegation")

    monkeypatch.setattr(MassCondensation, "execute", execute)
    configuration = CondensationConfiguration(
        CondensationExecutionMode.EQUAL_STEP,
        True,
        CondensationActivityMode.IDEAL,
        CondensationSurfaceMode.STATIC,
    )
    state = CPUCondensationExecutionState(
        CPUCondensationState(
            CondensationExecutionConfig(configuration), _aerosol()
        ),
        1.0,
        1,
        runnable,
    )

    with pytest.raises(
        ValueError,
        match="^isothermal condensation execution requires latent_heat=False.$",
    ):
        CPUCondensationExecutionAdapter().execute(state)

    assert calls == 0


def test_cpu_execution_adapter_propagates_delegate_errors_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a runnable error escapes after exactly one selected invocation."""
    from particula.dynamics import MassCondensation

    runnable = object.__new__(MassCondensation)
    error = RuntimeError("delegate failure")
    calls = 0

    def execute(*args: object, **kwargs: object) -> Aerosol:
        """Raise the prepared error after recording the native call."""
        nonlocal calls
        del args, kwargs
        calls += 1
        raise error

    monkeypatch.setattr(MassCondensation, "execute", execute)
    state = CPUCondensationExecutionState(
        CPUCondensationState(
            CondensationExecutionConfig(_configuration()), _aerosol()
        ),
        1.0,
        1,
        runnable,
    )

    with pytest.raises(RuntimeError) as caught:
        CPUCondensationExecutionAdapter().execute(state)

    assert caught.value is error
    assert calls == 1


def test_execution_states_retain_p2_carriers_and_require_exact_types() -> None:
    """Test P3 carriers are frozen identity carriers with exact boundaries."""
    from particula.dynamics import MassCondensation

    p2_state = CPUCondensationState(
        CondensationExecutionConfig(_configuration()), _aerosol()
    )
    state = CPUCondensationExecutionState(
        p2_state, 1.0, 1, object.__new__(MassCondensation)
    )

    assert state.state is p2_state
    assert state != CPUCondensationExecutionState(
        p2_state, 1.0, 1, object.__new__(MassCondensation)
    )
    with pytest.raises(FrozenInstanceError):
        state.time_step = 2.0  # type: ignore[misc]
    with pytest.raises(
        TypeError, match="^state must be an exact CPUCondensationState.$"
    ):
        CPUCondensationExecutionState(
            cast(CPUCondensationState, object()),
            1.0,
            1,
            object.__new__(MassCondensation),
        )
    with pytest.raises(
        TypeError, match="^runnable must be an exact MassCondensation.$"
    ):
        CPUCondensationExecutionState(p2_state, 1.0, 1, object())  # type: ignore[arg-type]


def test_execution_states_reject_p2_and_runnable_subclasses() -> None:
    """Test P3 construction excludes P2 and runnable subclasses."""
    from particula.dynamics import MassCondensation

    class DerivedState(CPUCondensationState):
        """Unsupported P2 CPU state subtype."""

    class DerivedRunnable(MassCondensation):
        """Unsupported runnable subtype."""

    config = CondensationExecutionConfig(_configuration())
    derived_state = DerivedState(config, _aerosol())
    with pytest.raises(
        TypeError, match="^state must be an exact CPUCondensationState.$"
    ):
        CPUCondensationExecutionState(
            derived_state, 1.0, 1, object.__new__(MassCondensation)
        )
    with pytest.raises(
        TypeError, match="^runnable must be an exact MassCondensation.$"
    ):
        CPUCondensationExecutionState(
            CPUCondensationState(config, _aerosol()),
            1.0,
            1,
            object.__new__(DerivedRunnable),
        )


def test_cpu_execution_adapter_rejects_replacement_aerosol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a replacement result is rejected after the sole delegate call."""
    from particula.dynamics import MassCondensation

    runnable = object.__new__(MassCondensation)
    calls = 0

    def execute(*args: object, **kwargs: object) -> Aerosol:
        """Return a replacement aerosol after recording the native call."""
        nonlocal calls
        del args, kwargs
        calls += 1
        return _aerosol()

    monkeypatch.setattr(MassCondensation, "execute", execute)
    state = CPUCondensationExecutionState(
        CPUCondensationState(
            CondensationExecutionConfig(_configuration()), _aerosol()
        ),
        1.0,
        1,
        runnable,
    )

    with pytest.raises(
        ValueError, match="^CPU runnable must return the original aerosol.$"
    ):
        CPUCondensationExecutionAdapter().execute(state)

    assert calls == 1


def test_cpu_execution_adapter_and_state_do_not_import_optional_backend() -> (
    None
):
    """Test concrete CPU dispatch remains independent of optional Warp imports."""
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
from particula.dynamics import MassCondensation
from particula.execution import (
    CondensationActivityMode, CondensationConfiguration,
    CondensationExecutionMode, CondensationSurfaceMode,
)
from particula.execution.adapters.condensation import (
    CPUCondensationExecutionAdapter, CPUCondensationExecutionState,
    CPUCondensationState, CondensationExecutionConfig,
)

configuration = CondensationConfiguration(
    CondensationExecutionMode.EQUAL_STEP, False,
    CondensationActivityMode.IDEAL, CondensationSurfaceMode.STATIC,
)
aerosol = object.__new__(Aerosol)
runnable = object.__new__(MassCondensation)
MassCondensation.execute = lambda self, source, time_step, sub_steps: source
state = CPUCondensationExecutionState(
    CPUCondensationState(CondensationExecutionConfig(configuration), aerosol),
    1.0, 1, runnable,
)
assert CPUCondensationExecutionAdapter().execute(state).backend_result.value is aerosol
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


def test_condensation_adapters_register_and_resolve_context_locally() -> None:
    """Test selected concrete adapters remain context-local registration values."""
    configuration = _configuration()
    requirements = get_condensation_requirements(configuration)
    cpu_device = Device(Backend.CPU, "cpu")
    warp_device = Device(Backend.WARP, "cpu")
    context = ExecutionContext(
        CapabilityMatrix(
            frozenset(
                {
                    CapabilityDeclaration(
                        cpu_device, CONDENSATION_PROCESS, requirements
                    ),
                    CapabilityDeclaration(
                        warp_device, CONDENSATION_PROCESS, requirements
                    ),
                }
            )
        )
    )
    cpu_adapter = CPUCondensationExecutionAdapter()
    warp_adapter = WarpCondensationExecutionAdapter()
    context.register_adapter(CONDENSATION_PROCESS, Backend.CPU, cpu_adapter)
    context.register_adapter(CONDENSATION_PROCESS, Backend.WARP, warp_adapter)

    assert (
        context.resolve(
            ExecutionRequest(
                Backend.CPU,
                cpu_device,
                CONDENSATION_PROCESS,
                requirements,
            )
        )
        is cpu_adapter
    )
    assert (
        context.resolve(
            ExecutionRequest(
                Backend.WARP,
                warp_device,
                CONDENSATION_PROCESS,
                requirements,
            )
        )
        is warp_adapter
    )


@pytest.mark.parametrize(
    "time_step", [True, -1.0, float("inf"), float("nan"), object()]
)
def test_cpu_execution_adapter_rejects_controls_before_delegate(
    time_step: object,
) -> None:
    """Test invalid CPU time controls cannot reach the selected runnable."""
    from particula.dynamics import MassCondensation

    state = CPUCondensationExecutionState(
        CPUCondensationState(
            CondensationExecutionConfig(_configuration()), _aerosol()
        ),
        time_step,
        1,
        object.__new__(MassCondensation),
    )

    with pytest.raises((TypeError, ValueError)):
        CPUCondensationExecutionAdapter().execute(state)


def test_cpu_execution_adapter_rejects_unrelated_state() -> None:
    """Test CPU dispatch rejects an unselected execution-state carrier."""
    with pytest.raises(
        TypeError, match="^state must be a CPUCondensationExecutionState.$"
    ):
        CPUCondensationExecutionAdapter().execute(cast(Any, object()))


@pytest.mark.warp
def test_warp_execution_adapter_lazily_dispatches_native_tuple_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test selected Warp dispatch neither converts nor reconstructs results."""
    import particula.execution.adapters.condensation as condensation

    particles, gas, environment = _warp_state_inputs()
    transfer = pytest.importorskip("warp").zeros(
        (1, 2, 1), dtype=pytest.importorskip("warp").float64, device="cpu"
    )
    p2_state = condensation.WarpCondensationState(
        CondensationExecutionConfig(_configuration()),
        particles,
        gas,
        environment,
        object(),
        mass_transfer=transfer,
        thermal_work=object(),
    )
    state = WarpCondensationExecutionState(p2_state, 1.0)
    expected = (particles, transfer)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def kernel(*args: object, **kwargs: object) -> tuple[object, object]:
        """Record the direct native call without launching Warp work."""
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        condensation, "_get_condensation_step_gpu", lambda: kernel
    )
    wp = pytest.importorskip("warp")
    from particula.gpu import conversion

    def fail_boundary_work(*args: object, **kwargs: object) -> None:
        """Fail if dispatch crosses the explicit caller-owned boundary."""
        del args, kwargs
        raise AssertionError("Warp dispatch must not transfer or synchronize.")

    monkeypatch.setattr(wp, "synchronize", fail_boundary_work)
    monkeypatch.setattr(wp, "copy", fail_boundary_work)
    for name in (
        "to_warp_particle_data",
        "to_warp_gas_data",
        "to_warp_environment_data",
        "from_warp_particle_data",
        "from_warp_gas_data",
        "from_warp_environment_data",
    ):
        monkeypatch.setattr(conversion, name, fail_boundary_work)
    result = WarpCondensationExecutionAdapter().execute(state)

    assert len(calls) == 1
    assert calls[0][0] == (particles, gas, None, None, 1.0)
    assert calls[0][1]["mass_transfer"] is transfer
    assert calls[0][1]["environment"] is environment
    assert calls[0][1]["thermodynamics"] is p2_state.thermodynamics
    assert calls[0][1]["activity_surface"] is p2_state.activity_surface
    assert calls[0][1]["scratch_buffers"] is p2_state.scratch_buffers
    assert calls[0][1]["thermal_work"] is p2_state.thermal_work
    assert calls[0][1]["latent_heat"] is None
    assert calls[0][1]["energy_transfer"] is None
    assert result.state is state
    assert result.backend_result is not None
    assert result.backend_result.value is expected
    assert result.metadata == ()
    assert result.mutation.scopes == frozenset({MutationScope.STATE})


@pytest.mark.warp
def test_warp_execution_state_is_frozen_identity_carrier() -> None:
    """Test the P3 Warp state retains the P2 carrier without rebinding."""
    import particula.execution.adapters.condensation as condensation

    particles, gas, environment = _warp_state_inputs()
    p2_state = condensation.WarpCondensationState(
        CondensationExecutionConfig(_configuration()),
        particles,
        gas,
        environment,
        object(),
    )
    state = WarpCondensationExecutionState(p2_state, 1.0)

    assert state.state is p2_state
    assert state.backend_payload == (particles, gas, environment)
    assert state != WarpCondensationExecutionState(p2_state, 1.0)
    with pytest.raises(FrozenInstanceError):
        state.time_step = 2.0  # type: ignore[misc]
    with pytest.raises(
        TypeError, match="^state must be an exact WarpCondensationState.$"
    ):
        WarpCondensationExecutionState(object(), 1.0)  # type: ignore[arg-type]


@pytest.mark.warp
@pytest.mark.parametrize(
    ("sidecar_name", "message"),
    [
        (
            "latent_heat",
            "isothermal condensation execution requires latent_heat=None.",
        ),
        (
            "energy_transfer",
            "isothermal condensation execution requires energy_transfer=None.",
        ),
    ],
)
def test_warp_execution_rejects_thermal_sidecars_before_kernel_resolution(
    sidecar_name: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test isothermal rejection leaves the direct-kernel seam untouched."""
    import particula.execution.adapters.condensation as condensation

    particles, gas, environment = _warp_state_inputs()
    if sidecar_name == "energy_transfer":
        wp = pytest.importorskip("warp")
        sidecar = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    else:
        sidecar = object()
    kwargs = {sidecar_name: sidecar}
    p2_state = condensation.WarpCondensationState(
        CondensationExecutionConfig(_configuration()),
        particles,
        gas,
        environment,
        object(),
        **kwargs,
    )
    state = WarpCondensationExecutionState(p2_state, 1.0)
    resources = (*_primary_warp_resources(particles, gas, environment),)
    if sidecar_name == "energy_transfer":
        resources += (sidecar,)
    snapshot = _snapshot_warp_resources(*resources)
    monkeypatch.setattr(
        condensation,
        "_get_condensation_step_gpu",
        lambda: pytest.fail("preflight must not resolve the kernel"),
    )

    with pytest.raises(ValueError, match=f"^{message}$"):
        WarpCondensationExecutionAdapter().execute(state)

    _assert_warp_resources_unchanged(snapshot, *resources)


@pytest.mark.warp
def test_warp_execution_propagates_native_exception_without_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a direct-kernel error escapes by identity after its one call."""
    import particula.execution.adapters.condensation as condensation

    particles, gas, environment = _warp_state_inputs()
    p2_state = condensation.WarpCondensationState(
        CondensationExecutionConfig(_configuration()),
        particles,
        gas,
        environment,
        object(),
    )
    error = RuntimeError("native failure")
    calls = 0

    def fail(*args: object, **kwargs: object) -> object:
        """Raise the prepared native error after recording the call."""
        nonlocal calls
        del args, kwargs
        calls += 1
        raise error

    monkeypatch.setattr(
        condensation, "_get_condensation_step_gpu", lambda: fail
    )
    with pytest.raises(RuntimeError) as caught:
        WarpCondensationExecutionAdapter().execute(
            WarpCondensationExecutionState(p2_state, 1.0)
        )

    assert caught.value is error
    assert calls == 1


@pytest.mark.warp
@pytest.mark.parametrize(
    "time_step", [True, -1.0, float("inf"), float("nan"), object()]
)
def test_warp_execution_rejects_invalid_controls_before_kernel_resolution(
    time_step: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invalid Warp controls do not resolve or call the direct kernel."""
    import particula.execution.adapters.condensation as condensation

    particles, gas, environment = _warp_state_inputs()
    p2_state = condensation.WarpCondensationState(
        CondensationExecutionConfig(_configuration()),
        particles,
        gas,
        environment,
        object(),
    )
    resources = _primary_warp_resources(particles, gas, environment)
    snapshot = _snapshot_warp_resources(*resources)
    resolutions = 0

    def resolve() -> object:
        """Record an unexpected lazy-kernel resolution."""
        nonlocal resolutions
        resolutions += 1
        return object()

    monkeypatch.setattr(condensation, "_get_condensation_step_gpu", resolve)

    with pytest.raises((TypeError, ValueError)):
        WarpCondensationExecutionAdapter().execute(
            WarpCondensationExecutionState(p2_state, time_step)
        )

    assert resolutions == 0
    _assert_warp_resources_unchanged(snapshot, *resources)


@pytest.mark.warp
def test_warp_execution_rejects_unselected_state_and_profile_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test state and profile preflight precede the optional kernel import."""
    import particula.execution.adapters.condensation as condensation

    resolutions = 0

    def resolve() -> object:
        """Record an unexpected lazy-kernel resolution."""
        nonlocal resolutions
        resolutions += 1
        return object()

    monkeypatch.setattr(condensation, "_get_condensation_step_gpu", resolve)
    with pytest.raises(
        TypeError, match="^state must be a WarpCondensationExecutionState.$"
    ):
        WarpCondensationExecutionAdapter().execute(cast(Any, object()))

    particles, gas, environment = _warp_state_inputs()
    resources = _primary_warp_resources(particles, gas, environment)
    snapshot = _snapshot_warp_resources(*resources)
    unsupported = CondensationConfiguration(
        CondensationExecutionMode.EQUAL_STEP,
        False,
        CondensationActivityMode.NONREPRESENTABLE,
        CondensationSurfaceMode.STATIC,
    )
    state = WarpCondensationExecutionState(
        condensation.WarpCondensationState(
            CondensationExecutionConfig(unsupported),
            particles,
            gas,
            environment,
            object(),
        ),
        1.0,
    )
    with pytest.raises(
        ValueError, match="^Unsupported capability declaration:"
    ):
        WarpCondensationExecutionAdapter().execute(state)

    latent_heat = CondensationConfiguration(
        CondensationExecutionMode.EQUAL_STEP,
        True,
        CondensationActivityMode.IDEAL,
        CondensationSurfaceMode.STATIC,
    )
    semantic_state = WarpCondensationExecutionState(
        condensation.WarpCondensationState(
            CondensationExecutionConfig(latent_heat),
            particles,
            gas,
            environment,
            object(),
        ),
        1.0,
    )
    with pytest.raises(
        ValueError,
        match="^isothermal condensation execution requires latent_heat=False.$",
    ):
        WarpCondensationExecutionAdapter().execute(semantic_state)

    assert resolutions == 0
    _assert_warp_resources_unchanged(snapshot, *resources)


@pytest.mark.warp
def test_get_condensation_step_gpu_resolves_the_direct_kernel() -> None:
    """Test the private lazy resolver returns the supported direct entry point."""
    from particula.execution.adapters.condensation import (
        _get_condensation_step_gpu,
    )
    from particula.gpu.kernels import condensation_step_gpu

    assert _get_condensation_step_gpu() is condensation_step_gpu
