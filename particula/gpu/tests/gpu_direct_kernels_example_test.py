"""Regression tests for the explicit direct-condensation quick-start."""

from __future__ import annotations

import importlib
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from particula.gpu import WARP_AVAILABLE

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "Examples"
    / "gpu_direct_kernels_quick_start.py"
)
EXAMPLES_ROOT = EXAMPLE_PATH.parent
CPU_ONLY_OUTPUT = [
    "Canonical path: docs/Examples/gpu_direct_kernels_quick_start.py",
    "ParticleData constructed: masses=(1, 2, 1), concentration=(1, 2), charge=(1, 2), density=(1,), volume=(1,)",
    "GasData constructed: concentration=(1, 1), molar_mass=(1,), partitioning=(1,), names=['Water']",
    "Warp is unavailable or disabled; no kernel ran.",
]
WARP_OUTPUT = [
    "Canonical path: docs/Examples/gpu_direct_kernels_quick_start.py",
    "ParticleData constructed: masses=(1, 2, 1), concentration=(1, 2), charge=(1, 2), density=(1,), volume=(1,)",
    "GasData constructed: concentration=(1, 1), molar_mass=(1,), partitioning=(1,), names=['Water']",
    "Explicit helpers: CPU→Warp conversion -> direct condensation -> CPU checkpoints",
    "Direct condensation complete: device=cpu, calls=2, total_transfer_shape=(1, 2, 1)",
    "Final checkpoints restored: particle_masses=(1, 2, 1), gas_concentration=(1, 1), names=['Water']",
    "Two-item kernel return; energy remains a caller-owned sidecar.",
    "Fixed-shape fp64 scratch, latent heat, and energy sidecars reused.",
]
EXAMPLE_TIMEOUT_SECONDS = 10
KERNEL_MODULES = (
    "particula.gpu.kernels",
    "particula.gpu.kernels.condensation",
    "particula.gpu.kernels.thermodynamics",
)


@pytest.fixture
def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load the published top-level example module."""
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
    sys.modules.pop("gpu_direct_kernels_quick_start", None)
    return importlib.import_module("gpu_direct_kernels_quick_start")


def _run_example(*, force_no_warp: bool) -> subprocess.CompletedProcess[str]:
    """Execute the published example path and capture output."""
    env = os.environ.copy()
    if force_no_warp:
        env["PARTICULA_EXAMPLE_FORCE_NO_WARP"] = "1"
    else:
        env.pop("PARTICULA_EXAMPLE_FORCE_NO_WARP", None)
    return subprocess.run(  # noqa: S603
        [sys.executable, str(EXAMPLE_PATH)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=EXAMPLE_TIMEOUT_SECONDS,
    )


def test_cpu_builders_preserve_documented_dtype_and_species_order(
    example_module: types.ModuleType,
) -> None:
    """Test CPU fixture schemas, dtypes, and Water mass-column ordering."""
    particle_data = example_module._build_particle_data()
    gas_data = example_module._build_gas_data()
    vapor_pressure = example_module._build_vapor_pressure()

    assert particle_data.masses.shape == (1, 2, 1)
    assert particle_data.concentration.shape == (1, 2)
    assert particle_data.masses.dtype == np.float64
    assert particle_data.concentration.dtype == np.float64
    assert particle_data.charge.dtype == np.float64
    assert particle_data.density.dtype == np.float64
    assert particle_data.volume.dtype == np.float64
    assert gas_data.name == ["Water"]
    assert gas_data.molar_mass.shape == (1,)
    assert gas_data.concentration.shape == (1, 1)
    assert gas_data.molar_mass.dtype == np.float64
    assert gas_data.concentration.dtype == np.float64
    assert gas_data.partitioning.dtype == np.bool_
    assert vapor_pressure.shape == (1, 1)
    assert vapor_pressure.dtype == np.float64


def test_load_gpu_runtime_imports_only_direct_condensation_contract(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test lazy loading requests public step and concrete sidecar modules."""
    imported: list[str] = []
    fake_wp = object()
    fake_step = object()
    fake_scratch = object()
    fake_thermodynamics = object()
    modules = {
        "warp": fake_wp,
        "particula.gpu.kernels": types.SimpleNamespace(
            condensation_step_gpu=fake_step
        ),
        "particula.gpu.kernels.condensation": types.SimpleNamespace(
            CondensationScratchBuffers=fake_scratch
        ),
        "particula.gpu.kernels.thermodynamics": types.SimpleNamespace(
            ThermodynamicsConfig=fake_thermodynamics
        ),
    }

    def fake_import(name: str) -> object:
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(example_module.importlib, "import_module", fake_import)

    assert example_module._load_gpu_runtime() == (
        fake_wp,
        fake_step,
        fake_scratch,
        fake_thermodynamics,
    )
    assert imported == ["warp", *KERNEL_MODULES]


def test_forced_no_warp_import_run_main_and_subprocess_defer_kernels(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test every forced no-Warp route avoids direct and concrete modules."""
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    sys.modules.pop("gpu_direct_kernels_quick_start", None)
    for module_name in KERNEL_MODULES:
        sys.modules.pop(module_name, None)

    module = importlib.import_module("gpu_direct_kernels_quick_start")
    result = module.run_example()
    module.main()

    assert result.output == CPU_ONLY_OUTPUT
    assert capsys.readouterr().out.splitlines() == CPU_ONLY_OUTPUT
    assert all(module_name not in sys.modules for module_name in KERNEL_MODULES)
    process = _run_example(force_no_warp=True)
    assert process.stdout.splitlines() == CPU_ONLY_OUTPUT


def test_main_entrypoint_prints_result_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test runpy execution prints the ``ExampleRun.output`` lines."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == CPU_ONLY_OUTPUT


class _FakeArray:
    """Small Warp-array fake retaining metadata and test-visible values."""

    def __init__(self, values: Any, dtype: object, device: str) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = dtype
        self.device = device

    def numpy(self) -> np.ndarray:
        """Return fake storage as a NumPy array."""
        return self.values


class _FakeWP:
    """Construct metadata-bearing fake Warp arrays."""

    float64 = object()
    int32 = object()

    @classmethod
    def zeros(
        cls, shape: tuple[int, ...], dtype: object, device: str
    ) -> _FakeArray:
        """Allocate zero fake storage using dtype metadata."""
        return _FakeArray(np.zeros(shape), dtype, device)

    @classmethod
    def array(cls, values: Any, dtype: object, device: str) -> _FakeArray:
        """Construct fake storage from values using dtype metadata."""
        return _FakeArray(values, dtype, device)


class _FakeScratch:
    """Concrete scratch-sidecar fake retaining supplied field identities."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeThermodynamics:
    """Concrete thermodynamic-sidecar fake retaining supplied fields."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def test_enabled_path_reuses_complete_caller_owned_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test direct calls use identical fp64 sidecars and restore last state."""
    calls: list[dict[str, Any]] = []
    events: list[str] = []
    conversion_calls: dict[str, Any] = {}
    restored_names: list[list[str]] = []
    particle_data = example_module._build_particle_data()
    gas_data = example_module._build_gas_data()
    gpu_particles = object()
    gpu_gas = object()

    def fake_step(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        events.append("step")
        calls.append(kwargs)
        return args[0], kwargs["scratch_buffers"].total_mass_transfer

    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        lambda: (_FakeWP, fake_step, _FakeScratch, _FakeThermodynamics),
    )
    monkeypatch.setattr(
        example_module,
        "to_warp_particle_data",
        lambda *args, **kwargs: (
            conversion_calls.setdefault("particles", (args, kwargs)),
            events.append("to_particles"),
            gpu_particles,
        )[2],
    )
    monkeypatch.setattr(
        example_module,
        "to_warp_gas_data",
        lambda *args, **kwargs: (
            conversion_calls.setdefault("gas", (args, kwargs)),
            events.append("to_gas"),
            gpu_gas,
        )[2],
    )
    monkeypatch.setattr(
        example_module,
        "from_warp_particle_data",
        lambda value: (events.append("from_particles"), particle_data)[1],
    )
    monkeypatch.setattr(
        example_module,
        "from_warp_gas_data",
        lambda value, name: (
            restored_names.append(name),
            events.append("from_gas"),
            gas_data,
        )[2],
    )

    result = example_module.run_example(device="cpu")

    assert result.output == WARP_OUTPUT
    assert events == [
        "to_particles",
        "to_gas",
        "step",
        "step",
        "from_particles",
        "from_gas",
    ]
    assert conversion_calls["particles"][1]["device"] == "cpu"
    assert conversion_calls["gas"][1]["device"] == "cpu"
    np.testing.assert_allclose(
        conversion_calls["gas"][1]["vapor_pressure"],
        np.array([[2330.0]], dtype=np.float64),
    )
    assert restored_names == [["Water"]]
    assert len(calls) == 2
    assert calls[0]["scratch_buffers"] is calls[1]["scratch_buffers"]
    assert calls[0]["latent_heat"] is calls[1]["latent_heat"]
    assert calls[0]["energy_transfer"] is calls[1]["energy_transfer"]
    assert (
        result.total_mass_transfer is result.scratch_buffers.total_mass_transfer
    )
    assert (
        result.total_mass_transfer
        is calls[1]["scratch_buffers"].total_mass_transfer
    )
    for call in calls:
        assert set(call) == {
            "temperature",
            "pressure",
            "time_step",
            "thermodynamics",
            "scratch_buffers",
            "latent_heat",
            "energy_transfer",
        }
        assert call["temperature"] == 298.15
        assert call["pressure"] == 101325.0
        assert call["time_step"] == 0.1
    scratch = result.scratch_buffers
    expected = {
        "work_mass_transfer": (1, 2, 1),
        "total_mass_transfer": (1, 2, 1),
        "dynamic_viscosity": (1,),
        "mean_free_path": (1,),
        "positive_mass_transfer_demand": (1, 1),
        "negative_mass_transfer_release": (1, 1),
        "positive_mass_transfer_scale": (1, 1),
    }
    for field, shape in expected.items():
        value = getattr(scratch, field)
        assert value.shape == shape
        assert value.dtype is _FakeWP.float64
        assert value.device == "cpu"
    assert result.latent_heat.shape == (1,)
    assert result.energy_transfer.shape == (1, 1)
    assert result.latent_heat.dtype is _FakeWP.float64
    assert result.energy_transfer.dtype is _FakeWP.float64
    thermodynamics = calls[0]["thermodynamics"]
    assert thermodynamics.modes.shape == (1,)
    assert thermodynamics.modes.dtype is _FakeWP.int32
    assert thermodynamics.modes.device == "cpu"
    assert thermodynamics.parameters.shape == (1, 4)
    assert thermodynamics.parameters.dtype is _FakeWP.float64
    assert thermodynamics.parameters.device == "cpu"
    assert thermodynamics.parameters.numpy()[0, 0] > 0.0
    assert thermodynamics.molar_mass_reference.shape == (1,)
    assert thermodynamics.molar_mass_reference.dtype is _FakeWP.float64
    assert thermodynamics.molar_mass_reference.device == "cpu"


@pytest.mark.parametrize("failure_call", [1, 2])
def test_kernel_failure_propagates_without_restore_or_success_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    example_module: types.ModuleType,
    failure_call: int,
) -> None:
    """Test failed steps are observable and do not claim a checkpoint."""
    invocation = 0

    def failing_step(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        nonlocal invocation
        invocation += 1
        if invocation == failure_call:
            raise RuntimeError("condensation failed")
        return args[0], kwargs["scratch_buffers"].total_mass_transfer

    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        lambda: (_FakeWP, failing_step, _FakeScratch, _FakeThermodynamics),
    )
    monkeypatch.setattr(
        example_module, "to_warp_particle_data", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        example_module, "to_warp_gas_data", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        example_module,
        "from_warp_particle_data",
        lambda *a, **k: pytest.fail("restore must not run"),
    )
    monkeypatch.setattr(
        example_module,
        "from_warp_gas_data",
        lambda *a, **k: pytest.fail("restore must not run"),
    )

    with pytest.raises(RuntimeError, match="condensation failed"):
        example_module.main()
    assert capsys.readouterr().out == ""


@pytest.mark.warp
@pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp is not available")
def test_real_warp_cpu_path_reuses_sidecars_and_couples_gas(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test real Warp CPU execution restores physical final checkpoints."""
    original_loader = example_module._load_gpu_runtime
    calls: list[dict[str, Any]] = []

    def loader_with_spy() -> tuple[Any, Any, Any, Any]:
        wp, step, scratch, thermodynamics = original_loader()

        def spy(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            calls.append(kwargs)
            return step(*args, **kwargs)

        return wp, spy, scratch, thermodynamics

    monkeypatch.setattr(example_module, "_load_gpu_runtime", loader_with_spy)
    result = example_module.run_example(device="cpu")

    assert result.particle_data is not None
    assert result.gas_data is not None
    assert result.particle_data.masses.shape == (1, 2, 1)
    assert result.gas_data.name == ["Water"]
    assert len(calls) == 2
    assert calls[0]["scratch_buffers"] is calls[1]["scratch_buffers"]
    assert calls[0]["latent_heat"] is calls[1]["latent_heat"]
    assert calls[0]["energy_transfer"] is calls[1]["energy_transfer"]
    assert (
        result.total_mass_transfer is result.scratch_buffers.total_mass_transfer
    )
    transfer = result.total_mass_transfer.numpy()
    energy = result.energy_transfer.numpy()
    latent_heat = result.latent_heat.numpy()
    assert np.any(transfer != 0.0)
    assert np.any(energy != 0.0)
    np.testing.assert_allclose(
        energy,
        np.sum(transfer, axis=1) * latent_heat[None, :],
        rtol=1e-12,
        atol=1e-18,
    )
    initial_particles = example_module._build_particle_data()
    initial_gas = example_module._build_gas_data()
    particle_change = np.sum(
        (result.particle_data.masses - initial_particles.masses)
        * result.particle_data.concentration[:, :, None],
        axis=1,
    )
    gas_change = result.gas_data.concentration - initial_gas.concentration
    assert np.any(gas_change != 0.0)
    np.testing.assert_allclose(
        particle_change + gas_change,
        0.0,
        rtol=1e-12,
        atol=1e-18,
    )
