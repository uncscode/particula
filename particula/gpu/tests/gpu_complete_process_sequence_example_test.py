"""Regression coverage for the explicit direct-Warp process-sequence example."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import a fresh example module without requiring Warp."""
    examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
    monkeypatch.syspath_prepend(str(examples))
    sys.modules.pop("gpu_complete_process_sequence", None)
    module = importlib.import_module("gpu_complete_process_sequence")
    yield module
    sys.modules.pop("gpu_complete_process_sequence", None)


def test_build_cpu_state_has_documented_sparse_float64_schema(
    example_module: Any,
) -> None:
    """The CPU fixture has two active and two exactly-free fixed slots."""
    particles, gas, environment, names = example_module._build_cpu_state()

    assert particles.masses.shape == (1, 4, 2)
    assert particles.masses.dtype == np.float64
    assert particles.concentration.shape == (1, 4)
    assert particles.charge.shape == (1, 4)
    assert np.array_equal(particles.concentration, [[2.0, 0.0, 5.0, 0.0]])
    assert np.array_equal(particles.masses[0, [1, 3]], [[0.0, 0.0], [0.0, 0.0]])
    assert gas.concentration.shape == (1, 2)
    assert gas.partitioning.dtype == np.bool_
    assert np.all(gas.partitioning)
    assert names == ["water", "organic"]
    assert environment.temperature.shape == (1,)
    assert environment.saturation_ratio.shape == (1, 2)


def test_forced_disabled_path_does_not_reach_enabled_loader(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The force flag returns deterministic metadata before any GPU loader."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    monkeypatch.setattr(
        example_module,
        "_load_enabled_runtime",
        lambda: pytest.fail("enabled loader must not run"),
    )

    result = example_module.run_example()

    assert result.particle_data is None
    assert result.gas_data is None
    assert result.environment_data is None
    assert result.mass_transfer is None
    assert (
        result.output[-1] == "Warp is unavailable or disabled; no kernel ran."
    )


def test_forced_disabled_script_is_a_successful_no_kernel_subprocess() -> None:
    """The standalone disabled route succeeds without an optional Warp import."""
    example_path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "Examples"
        / "gpu_complete_process_sequence.py"
    )

    process = subprocess.run(  # noqa: S603
        [sys.executable, str(example_path)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
        timeout=10,
    )

    assert process.stdout.splitlines()[-1] == (
        "Warp is unavailable or disabled; no kernel ran."
    )


def test_warp_enabled_handles_import_failure_and_available_runtime(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warp detection distinguishes unavailable and importable installations."""
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)

    def unavailable(name: str) -> Any:
        assert name == "warp"
        raise ImportError("Warp is not installed")

    monkeypatch.setattr(example_module.importlib, "import_module", unavailable)
    assert not example_module._warp_enabled()

    monkeypatch.setattr(
        example_module.importlib,
        "import_module",
        lambda name: SimpleNamespace() if name == "warp" else None,
    )
    assert example_module._warp_enabled()


def test_runtime_unavailable_returns_metadata_without_device_transfers(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable GPU package exits before any explicit transfer occurs."""
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(example_module, "_load_enabled_runtime", lambda: None)

    result = example_module.run_example()

    assert result.particle_data is None
    assert result.gas_data is None
    assert result.environment_data is None
    assert (
        result.output[-1] == "Warp is unavailable or disabled; no kernel ran."
    )


def test_enabled_runtime_loading_failure_propagates_without_success_output(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A loader failure is visible and main emits no misleading success text."""
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module,
        "_load_enabled_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("runtime unavailable")),
    )

    with pytest.raises(RuntimeError, match="runtime unavailable"):
        example_module.main()

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "converter_name",
    [
        "to_warp_particle_data",
        "to_warp_gas_data",
        "to_warp_environment_data",
    ],
)
def test_setup_conversion_failure_does_not_continue_to_sidecars_or_steps(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    converter_name: str,
) -> None:
    """Each explicit setup transfer propagates before allocation or a kernel."""
    calls: list[str] = []

    def fail_conversion(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        calls.append("conversion")
        raise ValueError("invalid direct input")

    def unexpected_success(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        calls.append("successful conversion")
        return SimpleNamespace()

    gpu = SimpleNamespace(
        to_warp_particle_data=unexpected_success,
        to_warp_gas_data=unexpected_success,
        to_warp_environment_data=unexpected_success,
    )
    setattr(gpu, converter_name, fail_conversion)
    runtime = SimpleNamespace(gpu=gpu)
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module, "_load_enabled_runtime", lambda: runtime
    )
    monkeypatch.setattr(
        example_module,
        "_allocate_sidecars",
        lambda *args, **kwargs: pytest.fail("sidecars must not be allocated"),
    )

    with pytest.raises(ValueError, match="invalid direct input"):
        example_module.run_example()

    expected_successes = {
        "to_warp_particle_data": [],
        "to_warp_gas_data": ["successful conversion"],
        "to_warp_environment_data": [
            "successful conversion",
            "successful conversion",
        ],
    }
    assert calls == [*expected_successes[converter_name], "conversion"]


def test_load_enabled_runtime_defers_and_collects_required_boundaries(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy loader returns no runtime when disabled and binds all steps."""
    unavailable_gpu = SimpleNamespace(WARP_AVAILABLE=False)
    monkeypatch.setattr(
        example_module.importlib,
        "import_module",
        lambda name: unavailable_gpu if name == "particula.gpu" else object(),
    )
    assert example_module._load_enabled_runtime() is None

    kernels = SimpleNamespace(
        condensation_step_gpu=object(),
        coagulation_step_gpu=object(),
        dilution_step_gpu=object(),
        wall_loss_step_gpu=object(),
        nucleation_step_gpu=object(),
    )
    records = {
        "particula.gpu.kernels.condensation": SimpleNamespace(
            CondensationScratchBuffers=object(),
        ),
        "particula.gpu.kernels.thermodynamics": SimpleNamespace(
            ThermodynamicsConfig=object(),
        ),
        "particula.gpu.kernels.coagulation": SimpleNamespace(
            CoagulationMechanismConfig=object(),
        ),
        "particula.gpu.kernels.wall_loss": SimpleNamespace(
            NeutralWallLossConfig=object(),
        ),
        "particula.gpu.kernels.exhaustion": SimpleNamespace(
            ResamplingBuffers=object(),
        ),
        "particula.gpu.kernels.nucleation": SimpleNamespace(
            NucleationConfig=object(),
            NucleationScratchBuffers=object(),
            NucleationFinalizedDemandBuffers=object(),
            NucleationDiagnosticBuffers=object(),
            NucleationExhaustionBuffers=object(),
            NucleationExhaustionControls=object(),
        ),
    }
    available_gpu = SimpleNamespace(WARP_AVAILABLE=True)
    modules = {
        "warp": object(),
        "particula.gpu": available_gpu,
        "particula.gpu.kernels": kernels,
        **records,
    }
    monkeypatch.setattr(
        example_module.importlib,
        "import_module",
        lambda name: modules[name],
    )

    runtime = example_module._load_enabled_runtime()

    assert runtime is not None
    assert runtime.gpu is available_gpu
    assert runtime.condensation_step_gpu is kernels.condensation_step_gpu
    assert runtime.coagulation_step_gpu is kernels.coagulation_step_gpu
    assert runtime.dilution_step_gpu is kernels.dilution_step_gpu
    assert runtime.wall_loss_step_gpu is kernels.wall_loss_step_gpu
    assert runtime.nucleation_step_gpu is kernels.nucleation_step_gpu
    assert (
        runtime.NucleationExhaustionControls
        is records[
            "particula.gpu.kernels.nucleation"
        ].NucleationExhaustionControls
    )


class _FakeArray:
    """Minimal metadata-preserving stand-in for a caller-owned Warp array."""

    def __init__(
        self, shape: tuple[int, ...], dtype: object, device: str
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device


class _FakeWarp:
    """Provide allocations and synchronization observation for example tests."""

    float64 = "float64"
    int32 = "int32"
    uint32 = "uint32"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def zeros(self, shape: Any, *, dtype: object, device: str) -> _FakeArray:
        return _FakeArray(_shape(shape), dtype, device)

    def ones(self, shape: Any, *, dtype: object, device: str) -> _FakeArray:
        return _FakeArray(_shape(shape), dtype, device)

    def full(
        self, shape: Any, value: object, *, dtype: object, device: str
    ) -> _FakeArray:
        del value
        return _FakeArray(_shape(shape), dtype, device)

    def array(
        self, values: np.ndarray, *, dtype: object, device: str
    ) -> _FakeArray:
        return _FakeArray(values.shape, dtype, device)

    def synchronize(self) -> None:
        self.events.append("synchronize")


def _shape(shape: Any) -> tuple[int, ...]:
    """Normalize Warp's scalar-or-tuple allocation shape convention."""
    return (shape,) if isinstance(shape, int) else tuple(shape)


class _Record:
    """Accept concrete record construction while retaining named fields."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.__dict__.update(kwargs)


def _fake_step_result(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    particles: Any,
    gas: Any,
    environment: Any,
) -> Any:
    """Assert one fake direct-step contract and return its documented owners."""
    assert args[0] is particles
    if name == "condensation":
        assert args[1] is gas and kwargs["environment"] is environment
        return particles, kwargs["mass_transfer"]
    if name == "coagulation":
        assert kwargs["environment"] is environment
        return particles, kwargs["collision_pairs"], kwargs["n_collisions"]
    if name == "dilution":
        assert args[1] is gas
        return particles, gas
    if name == "wall_loss":
        assert kwargs["environment"] is environment
        return particles
    assert args[1] is gas and kwargs["environment"] is environment
    return particles, gas


def test_enabled_path_converts_once_orders_steps_and_restores_once(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The five direct calls retain resident owners until one final restore."""
    events: list[str] = []
    particles = SimpleNamespace()
    gas = SimpleNamespace()
    environment = SimpleNamespace()

    def particle_convert(value: Any, *, device: str) -> Any:
        events.append("convert_particles")
        assert device == "cpu"
        return particles

    def gas_convert(value: Any, *, device: str) -> Any:
        events.append("convert_gas")
        return gas

    def environment_convert(value: Any, *, device: str) -> Any:
        events.append("convert_environment")
        return environment

    def particle_restore(value: Any, *, sync: bool) -> Any:
        assert value is particles and not sync
        events.append("restore_particles")
        return "particles"

    def gas_restore(value: Any, *, name: list[str], sync: bool) -> Any:
        assert value is gas and name == ["water", "organic"] and not sync
        events.append("restore_gas")
        return "gas"

    def environment_restore(value: Any, *, sync: bool) -> Any:
        assert value is environment and not sync
        events.append("restore_environment")
        return "environment"

    fake_gpu = SimpleNamespace(
        to_warp_particle_data=particle_convert,
        to_warp_gas_data=gas_convert,
        to_warp_environment_data=environment_convert,
        from_warp_particle_data=particle_restore,
        from_warp_gas_data=gas_restore,
        from_warp_environment_data=environment_restore,
    )

    def step(name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            events.append(name)
            return _fake_step_result(
                name, args, kwargs, particles, gas, environment
            )

        return call

    runtime = SimpleNamespace(
        wp=_FakeWarp(events),
        gpu=fake_gpu,
        condensation_step_gpu=step("condensation"),
        coagulation_step_gpu=step("coagulation"),
        dilution_step_gpu=step("dilution"),
        wall_loss_step_gpu=step("wall_loss"),
        nucleation_step_gpu=step("nucleation"),
        CondensationScratchBuffers=_Record,
        ThermodynamicsConfig=_Record,
        CoagulationMechanismConfig=_Record,
        NeutralWallLossConfig=_Record,
        ResamplingBuffers=_Record,
        NucleationConfig=_Record,
        NucleationScratchBuffers=_Record,
        NucleationFinalizedDemandBuffers=_Record,
        NucleationDiagnosticBuffers=_Record,
        NucleationExhaustionBuffers=_Record,
        NucleationExhaustionControls=_Record,
    )
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module, "_load_enabled_runtime", lambda: runtime
    )

    result = example_module.run_example()

    assert events == [
        "convert_particles",
        "convert_gas",
        "convert_environment",
        "condensation",
        "coagulation",
        "dilution",
        "wall_loss",
        "nucleation",
        "synchronize",
        "restore_particles",
        "restore_gas",
        "restore_environment",
    ]
    assert result.particle_data == "particles"
    assert result.gas_data == "gas"
    assert result.environment_data == "environment"
    assert result.mass_transfer.shape == (1, 4, 2)
    assert result.collision_pairs.shape == (1, 4, 2)
    assert result.coagulation_rng.shape == (1,)
    assert result.wall_rng.shape == (1,)


def test_direct_failure_propagates_without_sync_or_restore(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct-boundary validation error remains visible and has no fallback."""
    events: list[str] = []
    particles = SimpleNamespace()
    gas = SimpleNamespace()
    environment = SimpleNamespace()

    def fail_condensation(*args: Any, **kwargs: Any) -> Any:
        """Raise the documented invalid direct-input error at the first step."""
        assert args[:2] == (particles, gas)
        assert kwargs["environment"] is environment
        events.append("condensation")
        raise ValueError("time_step must be finite and nonnegative")

    def unexpected_restore(*args: Any, **kwargs: Any) -> Any:
        """Fail if an error path attempts an intermediate CPU checkpoint."""
        del args, kwargs
        pytest.fail("direct failure must not restore CPU state")

    runtime = SimpleNamespace(
        gpu=SimpleNamespace(
            to_warp_particle_data=lambda *args, **kwargs: particles,
            to_warp_gas_data=lambda *args, **kwargs: gas,
            to_warp_environment_data=lambda *args, **kwargs: environment,
            from_warp_particle_data=unexpected_restore,
            from_warp_gas_data=unexpected_restore,
            from_warp_environment_data=unexpected_restore,
        ),
        condensation_step_gpu=fail_condensation,
        NeutralWallLossConfig=_Record,
        NucleationConfig=_Record,
        CoagulationMechanismConfig=_Record,
    )
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module,
        "_load_enabled_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr(
        example_module,
        "_allocate_sidecars",
        lambda *args, **kwargs: SimpleNamespace(
            mass_transfer=object(),
            thermodynamics=object(),
            condensation_scratch=object(),
        ),
    )

    with pytest.raises(ValueError, match="time_step must be finite"):
        example_module.run_example()

    assert events == ["condensation"]


def test_main_prints_only_example_output(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The script entry function prints each completed-example output line."""
    monkeypatch.setattr(
        example_module,
        "run_example",
        lambda: example_module.ExampleRun(output=["first", "second"]),
    )

    example_module.main()

    assert capsys.readouterr().out.splitlines() == ["first", "second"]


@pytest.mark.warp
def test_real_warp_cpu_path_restores_named_cpu_containers(
    example_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The published path runs on Warp CPU when the optional runtime exists."""
    gpu = pytest.importorskip("particula.gpu")
    if not gpu.WARP_AVAILABLE:
        pytest.skip("Warp is not available")
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)

    result = example_module.run_example()

    assert result.particle_data is not None
    assert result.gas_data is not None
    assert result.environment_data is not None
    assert result.particle_data.masses.shape == (1, 4, 2)
    assert result.gas_data.name == ["water", "organic"]
    assert result.environment_data.temperature.shape == (1,)
