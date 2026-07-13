"""Smoke tests for the direct GPU kernel quick-start example."""

from __future__ import annotations

import importlib
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

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
    "GasData constructed: concentration=(1, 1), molar_mass=(1,), partitioning=(1,)",
    "Warp is optional; direct particula.gpu.kernels imports stay deferred until WARP_AVAILABLE passes, so this CPU-default quick-start finished without Warp.",
]
WARP_OUTPUT = [
    "Canonical path: docs/Examples/gpu_direct_kernels_quick_start.py",
    "ParticleData constructed: masses=(1, 2, 1), concentration=(1, 2), charge=(1, 2), density=(1,), volume=(1,)",
    "GasData constructed: concentration=(1, 1), molar_mass=(1,), partitioning=(1,)",
    "Explicit helpers: to_warp_particle_data/to_warp_gas_data -> particula.gpu.kernels -> from_warp_particle_data/from_warp_gas_data",
    "Condensation kernel complete: device=cpu, temperature=298.15 K, pressure=101325.0 Pa, mass_transfer_shape=(1, 2, 1)",
    "Gas round trip: restored_concentration=(1, 1), restored_names=['Water']",
    "Coagulation kernel complete: device=cpu, rng_states=caller-owned, initialize_rng=True, rng_seed=41, collision_counts_shape=(1,)",
    "Particle round trip: restored_masses=(1, 2, 1), restored_concentration=(1, 2)",
    "Direct GPU kernel quick-start complete on the Warp cpu path.",
]

EXAMPLE_TIMEOUT_SECONDS = 10


def _load_module(module_name: str, module_path: Path) -> types.ModuleType:
    """Load a Python module directly from a file path."""
    spec = __import__("importlib.util").util.spec_from_file_location(
        module_name,
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = __import__("importlib.util").util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
    command = [sys.executable, str(EXAMPLE_PATH)]
    try:
        return subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=EXAMPLE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        pytest.fail(
            "Example subprocess timed out after "
            f"{EXAMPLE_TIMEOUT_SECONDS} seconds: {command!r}\n"
            f"stdout:\n{stdout!r}\n"
            f"stderr:\n{stderr!r}"
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            "Example subprocess failed: "
            f"{command!r}\n"
            f"stdout:\n{exc.stdout!r}\n"
            f"stderr:\n{exc.stderr!r}"
        )


def test_build_particle_data_returns_documented_shapes(
    example_module: types.ModuleType,
) -> None:
    """Test particle example data matches the documented single-box schema."""
    particle_data = example_module._build_particle_data()

    assert particle_data.masses.shape == (1, 2, 1)
    assert particle_data.concentration.shape == (1, 2)
    assert particle_data.charge.shape == (1, 2)
    assert particle_data.density.shape == (1,)
    assert particle_data.volume.shape == (1,)
    np.testing.assert_allclose(particle_data.volume, np.array([1.0e-6]))


def test_build_gas_data_returns_documented_shapes_and_names(
    example_module: types.ModuleType,
) -> None:
    """Test gas example data matches the documented single-box schema."""
    gas_data = example_module._build_gas_data()

    assert gas_data.name == ["Water"]
    assert gas_data.molar_mass.shape == (1,)
    assert gas_data.concentration.shape == (1, 1)
    assert gas_data.partitioning.shape == (1,)
    np.testing.assert_array_equal(gas_data.partitioning, np.array([True]))


def test_build_vapor_pressure_returns_documented_shape_and_value(
    example_module: types.ModuleType,
) -> None:
    """Test condensation vapor pressure input matches the example contract."""
    vapor_pressure = example_module._build_vapor_pressure()

    assert vapor_pressure.shape == (1, 1)
    np.testing.assert_allclose(vapor_pressure, np.array([[2330.0]]))


def test_warp_enabled_respects_force_no_warp_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test the Warp guard honors both availability and forced CPU mode."""
    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)

    assert example_module._warp_enabled() is True

    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")

    assert example_module._warp_enabled() is False


def test_load_gpu_runtime_imports_supported_kernel_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test lazy runtime loading imports Warp and kernel entry points only."""
    imported: list[str] = []
    fake_wp = object()
    fake_kernels = types.SimpleNamespace(
        condensation_step_gpu=object(),
        coagulation_step_gpu=object(),
    )

    def _fake_import_module(name: str) -> object:
        imported.append(name)
        if name == "warp":
            return fake_wp
        if name == "particula.gpu.kernels":
            return fake_kernels
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(
        example_module.importlib,
        "import_module",
        _fake_import_module,
    )

    loaded = example_module._load_gpu_runtime()

    assert loaded == (
        fake_wp,
        fake_kernels.condensation_step_gpu,
        fake_kernels.coagulation_step_gpu,
    )
    assert imported == ["warp", "particula.gpu.kernels"]


def test_run_example_reports_cpu_only_message_when_warp_disabled(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test run_example returns the documented CPU-only success message."""
    monkeypatch.setattr(example_module, "WARP_AVAILABLE", False)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    sys.modules.pop("particula.gpu.kernels", None)

    output = example_module.run_example()

    assert output == CPU_ONLY_OUTPUT
    assert "particula.gpu.kernels" not in sys.modules


def test_module_import_and_no_warp_run_defer_kernel_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test import and no-Warp execution avoid loading kernel modules."""
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    sys.modules.pop("particula.gpu.kernels", None)
    sys.modules.pop("gpu_direct_kernels_quick_start", None)

    module = importlib.import_module("gpu_direct_kernels_quick_start")

    assert "particula.gpu.kernels" not in sys.modules
    assert module.run_example() == CPU_ONLY_OUTPUT
    assert "particula.gpu.kernels" not in sys.modules


def test_example_main_prints_example_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    example_module: types.ModuleType,
) -> None:
    """Test the published example prints the documented output."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")

    example_module.main()

    captured = capsys.readouterr()
    assert captured.out.splitlines() == CPU_ONLY_OUTPUT


def test_example_runs_as_main_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test the published example executes successfully as __main__."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")

    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.splitlines() == CPU_ONLY_OUTPUT


def test_example_non_warp_path_reports_cpu_success() -> None:
    """Test the published example path completes without Warp transfers."""
    result = _run_example(force_no_warp=True)

    assert result.stdout.splitlines() == CPU_ONLY_OUTPUT


def test_run_example_surfaces_subprocess_failure_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test subprocess failures include captured stdout and stderr."""

    def _raise_called_process_error(*args: object, **kwargs: object) -> object:
        raise subprocess.CalledProcessError(
            2,
            [sys.executable, str(EXAMPLE_PATH)],
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

    with pytest.raises(pytest.fail.Exception, match="partial stdout"):
        _run_example(force_no_warp=True)


def test_run_example_surfaces_subprocess_timeout_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test subprocess timeouts include captured stdout and stderr."""

    def _raise_timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(
            [sys.executable, str(EXAMPLE_PATH)],
            EXAMPLE_TIMEOUT_SECONDS,
            output="slow stdout",
            stderr="slow stderr",
        )

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(pytest.fail.Exception, match="slow stderr"):
        _run_example(force_no_warp=True)


def test_run_example_failure_does_not_print_success_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    example_module: types.ModuleType,
) -> None:
    """Test a kernel failure aborts before any success summary is printed."""

    class _FakeArray:
        def __init__(self, shape: tuple[int, ...], device: str = "cpu") -> None:
            self.shape = shape
            self.device = device

    class _FakeWP:
        uint32 = object()
        int32 = object()
        float64 = object()

        @staticmethod
        def zeros(
            shape: tuple[int, ...],
            dtype: object,
            device: str,
        ) -> _FakeArray:
            del dtype
            return _FakeArray(shape, device)

        @staticmethod
        def array(
            values: object,
            dtype: object,
            device: str,
        ) -> _FakeArray:
            del dtype
            return _FakeArray(np.asarray(values).shape, device)

    class _FakeParticleGPU:
        def __init__(self) -> None:
            self.masses = _FakeArray((1, 2, 1))

    class _FakeGasGPU:
        pass

    def _raise_condensation(*args: object, **kwargs: object) -> object:
        raise RuntimeError("condensation failed")

    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        lambda: (_FakeWP(), _raise_condensation, object()),
    )
    monkeypatch.setattr(
        example_module,
        "to_warp_particle_data",
        lambda *args, **kwargs: _FakeParticleGPU(),
    )
    monkeypatch.setattr(
        example_module,
        "to_warp_gas_data",
        lambda *args, **kwargs: _FakeGasGPU(),
    )

    with pytest.raises(RuntimeError, match="condensation failed"):
        example_module.main()

    captured = capsys.readouterr()
    assert captured.out == ""


def test_run_example_uses_scalar_condensation_inputs_and_caller_owned_rng(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test the lazy kernel path uses the documented direct-call contracts."""
    calls: dict[str, object] = {}
    particle_data = example_module._build_particle_data()
    gas_data = example_module._build_gas_data()

    class _FakeArray:
        def __init__(self, shape: tuple[int, ...], device: str = "cpu") -> None:
            self.shape = shape
            self.device = device

    class _FakeWP:
        uint32 = object()
        int32 = object()
        float64 = object()

        @staticmethod
        def zeros(
            shape: tuple[int, ...],
            dtype: object,
            device: str,
        ) -> _FakeArray:
            calls["rng_shape"] = shape
            calls["rng_dtype"] = dtype
            calls["rng_device"] = device
            return _FakeArray(shape, device)

        @staticmethod
        def array(
            values: object,
            dtype: object,
            device: str,
        ) -> _FakeArray:
            del dtype
            return _FakeArray(np.asarray(values).shape, device)

    class _FakeParticleGPU:
        def __init__(self) -> None:
            self.masses = _FakeArray(particle_data.masses.shape)

    class _FakeGasGPU:
        pass

    def _fake_condensation(
        *args: object, **kwargs: object
    ) -> tuple[object, _FakeArray]:
        calls["condensation_args"] = args
        calls["condensation_kwargs"] = kwargs
        return args[0], _FakeArray((1, 2, 1))

    def _fake_coagulation(
        *args: object, **kwargs: object
    ) -> tuple[object, _FakeArray, _FakeArray]:
        calls["coagulation_args"] = args
        calls["coagulation_kwargs"] = kwargs
        return args[0], _FakeArray((1, 4, 2)), _FakeArray((1,))

    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        lambda: (_FakeWP(), _fake_condensation, _fake_coagulation),
    )
    monkeypatch.setattr(
        example_module,
        "to_warp_particle_data",
        lambda *args, **kwargs: _FakeParticleGPU(),
    )
    monkeypatch.setattr(
        example_module,
        "to_warp_gas_data",
        lambda *args, **kwargs: _FakeGasGPU(),
    )
    monkeypatch.setattr(
        example_module,
        "from_warp_particle_data",
        lambda *args, **kwargs: particle_data,
    )
    monkeypatch.setattr(
        example_module,
        "from_warp_gas_data",
        lambda *args, **kwargs: gas_data,
    )

    output = example_module.run_example(device="cpu")

    assert output == WARP_OUTPUT
    assert calls["rng_shape"] == (1,)
    assert calls["rng_device"] == "cpu"
    condensation_kwargs = calls["condensation_kwargs"]
    coagulation_kwargs = calls["coagulation_kwargs"]
    assert isinstance(condensation_kwargs, dict)
    assert isinstance(coagulation_kwargs, dict)
    assert condensation_kwargs["temperature"] == 298.15
    assert condensation_kwargs["pressure"] == 101325.0
    assert condensation_kwargs["time_step"] == 0.1
    assert condensation_kwargs["thermodynamics"].modes.shape == (1,)
    assert coagulation_kwargs["temperature"] == 298.15
    assert coagulation_kwargs["pressure"] == 101325.0
    assert coagulation_kwargs["time_step"] == 0.1
    assert coagulation_kwargs["initialize_rng"] is True
    assert coagulation_kwargs["rng_seed"] == 41
    assert "environment" not in coagulation_kwargs
    assert coagulation_kwargs["rng_states"].shape == (1,)


@pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp is not available")
def test_run_example_warp_cpu_path_reports_kernel_and_transfer_output() -> None:
    """Test the real Warp CPU path exercises both direct kernels."""
    result = _run_example(force_no_warp=False)

    output_lines = result.stdout.splitlines()

    assert output_lines[-len(WARP_OUTPUT) :] == WARP_OUTPUT
