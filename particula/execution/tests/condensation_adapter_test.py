"""Tests for concrete, non-executing condensation state carriers."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from particula.aerosol import Aerosol
from particula.execution import (
    CondensationActivityMode,
    CondensationConfiguration,
    CondensationExecutionMode,
    CondensationSurfaceMode,
)
from particula.execution.adapters.condensation import (
    CondensationExecutionConfig,
    CPUCondensationState,
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
