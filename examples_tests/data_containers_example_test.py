"""Smoke tests for the runnable data-container example."""

from __future__ import annotations

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
    Path(__file__).resolve().parents[1]
    / "docs"
    / "Examples"
    / "data_containers_and_gpu_foundations.py"
)
GUIDE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "Examples"
    / "Data_Containers"
    / "data_containers_and_gpu_foundations.py"
)
EXAMPLES_ROOT = EXAMPLE_PATH.parent


def _load_module(module_name: str, module_path: Path) -> types.ModuleType:
    """Load a Python module directly from a file path."""
    spec = __import__("importlib.util").util.spec_from_file_location(
        module_name,
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load the published top-level example module."""
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
    sys.modules.pop("data_containers_and_gpu_foundations", None)
    return _load_module(
        "data_containers_and_gpu_foundations_test",
        EXAMPLE_PATH,
    )


@pytest.fixture
def guide_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load the guide-local forwarding module."""
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
    sys.modules.pop("data_containers_and_gpu_foundations", None)
    return _load_module(
        "data_containers_and_gpu_foundations_guide_test",
        GUIDE_PATH,
    )


def _run_example(*, force_no_warp: bool) -> subprocess.CompletedProcess[str]:
    """Execute the published example path and capture output."""
    env = os.environ.copy()
    if force_no_warp:
        env["PARTICULA_EXAMPLE_FORCE_NO_WARP"] = "1"
    else:
        env.pop("PARTICULA_EXAMPLE_FORCE_NO_WARP", None)
    return subprocess.run(  # noqa: S603 - repo-local example path and interpreter
        [sys.executable, str(EXAMPLE_PATH)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_build_particle_data_returns_documented_shapes(
    example_module: types.ModuleType,
) -> None:
    """Test particle example data matches the documented single-box schema."""
    particle_data = example_module._build_particle_data()

    assert particle_data.masses.shape == (1, 2, 2)
    assert particle_data.concentration.shape == (1, 2)
    assert particle_data.charge.shape == (1, 2)
    assert particle_data.density.shape == (2,)
    assert particle_data.volume.shape == (1,)
    np.testing.assert_allclose(particle_data.volume, np.array([1.0e-6]))


def test_build_gas_data_returns_documented_shapes_and_names(
    example_module: types.ModuleType,
) -> None:
    """Test gas example data matches the documented single-box schema."""
    gas_data = example_module._build_gas_data()

    assert gas_data.name == ["Water", "H2SO4"]
    assert gas_data.molar_mass.shape == (2,)
    assert gas_data.concentration.shape == (1, 2)
    assert gas_data.partitioning.shape == (2,)
    np.testing.assert_array_equal(gas_data.partitioning, np.array([True, True]))


def test_warp_enabled_honors_force_no_warp_environment(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test the force-no-warp environment variable disables Warp transfers."""
    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")

    assert example_module._warp_enabled() is False


def test_run_example_skips_transfers_when_warp_disabled(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Disabled execution constructs CPU containers without transfers."""
    monkeypatch.setattr(example_module, "WARP_AVAILABLE", False)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)

    events = []
    for name in ("_build_particle_data", "_build_gas_data"):
        original = getattr(example_module, name)

        def build(name=name, original=original):
            events.append(name)
            return original()

        monkeypatch.setattr(example_module, name, build)
    monkeypatch.setattr(
        example_module,
        "to_warp_particle_data",
        lambda *a, **k: pytest.fail("disabled transfer"),
    )
    monkeypatch.setattr(
        example_module,
        "to_warp_gas_data",
        lambda *a, **k: pytest.fail("disabled transfer"),
    )
    example_module.run_example()
    assert events == ["_build_particle_data", "_build_gas_data"]


def test_run_example_stops_transfers_after_optional_warp_failure(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """An optional transfer failure stops later transfers without escaping."""
    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)

    events = []

    def _raise_runtime_error(*args: object, **kwargs: object) -> object:
        events.append("transfer-failed")
        raise RuntimeError("warp backend unavailable")

    monkeypatch.setattr(
        example_module,
        "to_warp_particle_data",
        _raise_runtime_error,
    )

    for name in ("from_warp_particle_data", "to_warp_gas_data"):
        monkeypatch.setattr(
            example_module,
            name,
            lambda *a, **k: pytest.fail("continued after failed transfer"),
        )
    example_module.run_example()
    assert events == ["transfer-failed"]


def test_example_main_prints_example_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    example_module: types.ModuleType,
) -> None:
    """Test the entry point forwards arbitrary returned output lines."""
    lines = ["container-sentinel", "transfer-sentinel"]
    monkeypatch.setattr(example_module, "run_example", lambda: lines)

    example_module.main()

    captured = capsys.readouterr()
    assert captured.out.splitlines() == lines


def test_guide_main_prints_example_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    guide_module: types.ModuleType,
) -> None:
    """Test the guide-local module delegates to the canonical example."""
    lines = ["guide-sentinel", "canonical-sentinel"]
    monkeypatch.setattr(
        guide_module._CANONICAL_MODULE, "run_example", lambda: lines
    )

    guide_module.main()

    captured = capsys.readouterr()
    assert captured.out.splitlines() == lines


def test_example_runs_as_main_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test the published example executes successfully as __main__."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")

    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.err == ""


def test_guide_runs_as_main_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test the guide-local forwarding module executes successfully."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")

    runpy.run_path(str(GUIDE_PATH), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.err == ""


def test_example_non_warp_path_reports_cpu_success() -> None:
    """Test the published example path completes without Warp transfers."""
    result = _run_example(force_no_warp=True)

    assert result.returncode == 0


def test_guide_module_re_exports_canonical_run_example(
    monkeypatch: pytest.MonkeyPatch,
    guide_module: types.ModuleType,
) -> None:
    """Test the guide module re-exports the canonical run_example helper."""
    assert (
        guide_module.run_example is guide_module._CANONICAL_MODULE.run_example
    )
    assert guide_module.main is guide_module._CANONICAL_MODULE.main


def test_guide_module_raises_import_error_when_canonical_example_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
    guide_module: types.ModuleType,
) -> None:
    """Test guide import fails clearly when the canonical example cannot load."""

    def _missing_spec(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        guide_module.importlib.util,
        "spec_from_file_location",
        _missing_spec,
    )

    with pytest.raises(ImportError, match="Unable to load canonical example"):
        guide_module._load_canonical_module()


@pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp is not available")
def test_example_warp_path_reports_round_trip_shapes_and_names() -> None:
    """Test the published example path exercises Warp CPU round trips."""
    result = _run_example(force_no_warp=False)

    assert result.returncode == 0


@pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp is not available")
def test_run_example_warp_path_reports_round_trip_shapes_and_names(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Observe real CPU round trips, preserving arrays and ordered gas names."""
    monkeypatch.setattr(example_module, "WARP_AVAILABLE", True)
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)

    events = []
    sources = {}
    devices = {}
    completed = []

    def observe(name, function):
        def call(value, **kwargs):
            events.append(name)
            family = "particle" if "particle" in name else "gas"
            if name.startswith("to_"):
                assert kwargs["device"] == "cpu"
                sources[family] = value
                if family == "gas":
                    np.testing.assert_array_equal(
                        kwargs["vapor_pressure"], [[2330.0, 120.0]]
                    )
                result = function(value, **kwargs)
                devices[family] = result
                completed.append(name)
                return result
            assert value is devices[family]
            result = function(value, **kwargs)
            fields = (
                ("masses", "concentration", "charge", "density", "volume")
                if family == "particle"
                else ("concentration", "molar_mass", "partitioning")
            )
            for field in fields:
                np.testing.assert_array_equal(
                    getattr(result, field), getattr(sources[family], field)
                )
            if family == "gas":
                assert kwargs["name"] is sources[family].name
                assert result.name == ["Water", "H2SO4"]
                assert not hasattr(result, "vapor_pressure")
            completed.append(name)
            return result

        return call

    names = [
        "to_warp_particle_data",
        "from_warp_particle_data",
        "to_warp_gas_data",
        "from_warp_gas_data",
    ]
    for name in names:
        monkeypatch.setattr(
            example_module, name, observe(name, getattr(example_module, name))
        )
    example_module.run_example()
    assert events == names
    assert completed == names
