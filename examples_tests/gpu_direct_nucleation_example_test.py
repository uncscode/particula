"""Regression coverage for the published direct-Warp nucleation example."""

from __future__ import annotations

import ast
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "docs/Examples/Nucleation/gpu_direct_nucleation.py"


def _require_warp() -> None:
    """Skip cleanly when the optional Warp prerequisite is unavailable."""
    gpu = pytest.importorskip("particula.gpu")
    if not gpu.WARP_AVAILABLE:
        pytest.skip("Warp is unavailable")


def _run_documented_command() -> subprocess.CompletedProcess[str]:
    """Run the documented example command with actionable timeout errors."""
    command = [sys.executable, "-Werror", str(EXAMPLE_PATH)]
    try:
        return subprocess.run(  # noqa: S603
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            "Direct Warp nucleation example timed out after 30 seconds: "
            f"{' '.join(command)}"
        ) from error


@pytest.mark.warp
def test_direct_example_runs_with_explicit_transfer_and_conservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the standalone fixture and check identity-derived final state."""
    _require_warp()
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    globals_ = namespace["run_example"].__globals__
    events: list[str] = []
    converted = {}

    def observe(name, function):
        def call(*args, **kwargs):
            events.append(name)
            if name.startswith("from_warp_"):
                assert kwargs["sync"] is False
                assert args[0] is converted[name.replace("from_", "to_")]
            if name == "nucleation_step_gpu":
                assert args[0] is converted["to_warp_particle_data"]
                assert args[1] is converted["to_warp_gas_data"]
                assert (
                    kwargs["environment"]
                    is converted["to_warp_environment_data"]
                )
            result = function(*args, **kwargs)
            if name.startswith("to_warp_"):
                assert kwargs["device"] == "cpu"
                converted[name] = result
            return result

        return call

    boundaries = [
        "to_warp_particle_data",
        "to_warp_gas_data",
        "to_warp_environment_data",
        "nucleation_step_gpu",
        "synchronize",
        "from_warp_particle_data",
        "from_warp_gas_data",
        "from_warp_environment_data",
    ]
    for name in boundaries:
        if name != "synchronize":
            monkeypatch.setitem(globals_, name, observe(name, globals_[name]))
    wp = globals_["wp"]
    monkeypatch.setitem(
        globals_,
        "wp",
        SimpleNamespace(
            zeros=wp.zeros,
            ones=wp.ones,
            float64=wp.float64,
            int32=wp.int32,
            synchronize=observe("synchronize", wp.synchronize),
        ),
    )
    particles, gas, environment = namespace["run_example"]()
    assert events == boundaries
    assert particles.masses.shape == (1, 2, 1)
    assert gas.concentration.shape == (1, 1)
    assert environment.temperature.shape == (1,)
    assert np.count_nonzero(particles.concentration) == 1
    inventory = np.sum(particles.masses * particles.concentration[:, :, None])
    npt.assert_allclose(
        inventory + gas.concentration.sum(), 1.0, rtol=1e-12, atol=1e-30
    )


def test_direct_example_imports_documented_api_boundaries() -> None:
    """Retain the public step and concrete-only configuration import contract."""
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "particula.gpu.kernels" in imports
    assert "particula.gpu.kernels.nucleation" in imports
    assert "particula.gpu.kernels.exhaustion" in imports
    assert "particula.dynamics" not in imports


def test_documented_command_timeout_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the documented command when its subprocess exceeds the limit."""
    command = [sys.executable, "-Werror", str(EXAMPLE_PATH)]

    def raise_timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    with pytest.raises(
        AssertionError, match="Direct Warp nucleation example timed out"
    ) as error:
        _run_documented_command()

    assert " ".join(command) in str(error.value)


@pytest.mark.warp
def test_documented_direct_example_command_runs() -> None:
    """Run the documented warning-clean command without cross-process identity."""
    _require_warp()
    completed = _run_documented_command()
    assert completed.returncode == 0, completed.stderr
