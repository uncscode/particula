"""Regression tests for the selected-adapter GPU coagulation example."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs/Examples/gpu_coagulation_direct.py"
)
EXAMPLES_ROOT = EXAMPLE_PATH.parent
LAZY_IMPORTS = (
    "particula.gpu",
    "warp",
    "particula.execution.adapters.coagulation",
)
RUNTIME_IMPORTS = LAZY_IMPORTS[1:]
DISABLED_OUTPUT = [
    "Canonical path: docs/Examples/gpu_coagulation_direct.py",
    "ParticleData constructed: masses=(1, 8, 1), concentration=(1, 8), charge=(1, 8), density=(1,), volume=(1,)",
    "Warp is unavailable or disabled; no kernel ran.",
]


@pytest.fixture
def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load the standalone example without retained module state."""
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
    monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
    return importlib.import_module("gpu_coagulation_direct")


def test_cpu_fixture_has_documented_active_and_inactive_slots(
    example_module: types.ModuleType,
) -> None:
    """Test the deterministic fixed-slot CPU fixture."""
    particle_data = example_module._build_particle_data()

    assert particle_data.masses.shape == (1, 8, 1)
    assert particle_data.concentration.shape == (1, 8)
    assert all(
        array.dtype == np.float64
        for array in (
            particle_data.masses,
            particle_data.concentration,
            particle_data.charge,
            particle_data.density,
            particle_data.volume,
        )
    )
    np.testing.assert_array_equal(
        np.flatnonzero(particle_data.concentration[0] > 0.0), np.arange(6)
    )
    np.testing.assert_array_equal(
        np.flatnonzero(particle_data.concentration[0] == 0.0), [6, 7]
    )


def test_forced_disabled_routes_never_import_gpu_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test forced no-Warp behavior avoids conversion and adapter dispatch."""
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
    with monkeypatch.context() as cleanup:
        for module_name in LAZY_IMPORTS:
            cleanup.delitem(sys.modules, module_name, raising=False)
        module = importlib.import_module("gpu_coagulation_direct")
        cleanup.setattr(
            module,
            "_load_gpu_helpers",
            lambda: pytest.fail("disabled path loaded helpers"),
        )
        cleanup.setattr(
            module,
            "_load_gpu_runtime",
            lambda: pytest.fail("disabled path loaded runtime"),
        )
        assert module.run_example().output == DISABLED_OUTPUT
        module.main()
        assert capsys.readouterr().out.splitlines() == DISABLED_OUTPUT
        assert all(name not in sys.modules for name in LAZY_IMPORTS)

    process = subprocess.run(  # noqa: S603
        [sys.executable, str(EXAMPLE_PATH)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
        timeout=10,
    )
    assert process.stdout.splitlines() == DISABLED_OUTPUT


def test_runtime_loader_uses_selected_adapter_imports(
    monkeypatch: pytest.MonkeyPatch, example_module: types.ModuleType
) -> None:
    """Test lazy loading resolves Warp and concrete adapter types in order."""
    imported: list[str] = []
    wp = object()
    adapter_module = types.SimpleNamespace(
        BrownianCoagulationConfig=object(),
        WarpBrownianCoagulationState=object(),
        WarpBrownianCoagulationExecutionState=object(),
        WarpBrownianCoagulationExecutionAdapter=object(),
    )

    def fake_import(name: str) -> object:
        imported.append(name)
        return {"warp": wp, RUNTIME_IMPORTS[1]: adapter_module}[name]

    monkeypatch.setattr(example_module.importlib, "import_module", fake_import)
    assert example_module._load_gpu_runtime() == (
        wp,
        adapter_module.BrownianCoagulationConfig,
        adapter_module.WarpBrownianCoagulationState,
        adapter_module.WarpBrownianCoagulationExecutionState,
        adapter_module.WarpBrownianCoagulationExecutionAdapter,
    )
    assert imported == list(RUNTIME_IMPORTS)


class _FakeArray:
    """Minimal Warp array fake retaining allocation metadata."""

    def __init__(
        self, shape: tuple[int, ...], dtype: object, device: str
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.values = np.zeros(
            shape, dtype=np.int32 if dtype is _FakeWP.int32 else np.uint32
        )

    def numpy(self) -> np.ndarray:
        """Return fake storage for assertions."""
        return self.values


class _FakeWP:
    """Fake Warp runtime with allocations and synchronization events."""

    int32 = object()
    uint32 = object()
    events: list[str] = []

    @classmethod
    def zeros(
        cls, shape: tuple[int, ...], dtype: object, device: str
    ) -> _FakeArray:
        """Allocate a fake same-device sidecar."""
        return _FakeArray(shape, dtype, device)

    @classmethod
    def synchronize(cls) -> None:
        """Record caller-controlled synchronization."""
        cls.events.append("synchronize")


class _FakeConfig:
    """Record the exact marker construction count."""

    calls = 0

    def __init__(self) -> None:
        type(self).calls += 1


class _FakeState:
    """Record selected state resources and arguments."""

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        type(self).calls.append((args, kwargs))
        self.args = args
        self.kwargs = kwargs


class _FakeExecutionState:
    """Wrap the state as the selected execution carrier."""

    def __init__(self, state: _FakeState) -> None:
        self.state = state


class _FakeAdapter:
    """Return the selected backend result through the execution envelope."""

    calls: list[_FakeExecutionState] = []

    def execute(self, execution_state: _FakeExecutionState) -> Any:
        """Record execution and retain supplied diagnostic identities."""
        type(self).calls.append(execution_state)
        state = execution_state.state
        return types.SimpleNamespace(
            backend_result=types.SimpleNamespace(
                value=types.SimpleNamespace(
                    collision_pairs=state.kwargs["collision_pairs"],
                    n_collisions=state.kwargs["n_collisions"],
                )
            )
        )


def _reset_fakes() -> None:
    """Clear fake runtime records between enabled-path tests."""
    _FakeWP.events = []
    _FakeConfig.calls = 0
    _FakeState.calls = []
    _FakeAdapter.calls = []


def test_enabled_path_uses_selected_adapter_and_explicit_lifecycle(
    monkeypatch: pytest.MonkeyPatch, example_module: types.ModuleType
) -> None:
    """Test state construction, identity forwarding, and lifecycle order."""
    _reset_fakes()
    events: list[str] = []
    gpu_particles = types.SimpleNamespace(volume=object())
    restored = example_module._build_particle_data()

    def convert(_: Any, device: str | None = None) -> Any:
        """Return the fake GPU particle container after recording conversion."""
        del device
        events.append("convert")
        return gpu_particles

    def restore(_: Any) -> Any:
        """Return the restored CPU checkpoint after recording restoration."""
        events.append("restore")
        return restored

    monkeypatch.setattr(
        example_module,
        "_load_gpu_helpers",
        lambda: (True, convert, restore),
    )
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        lambda: (
            _FakeWP,
            _FakeConfig,
            _FakeState,
            _FakeExecutionState,
            _FakeAdapter,
        ),
    )

    result = example_module.run_example()

    assert events == ["convert", "restore"]
    assert _FakeWP.events == ["synchronize"]
    assert _FakeConfig.calls == 2
    assert len(_FakeState.calls) == len(_FakeAdapter.calls) == 2
    for index, (args, kwargs) in enumerate(_FakeState.calls):
        assert isinstance(args[0], _FakeConfig)
        assert args[1:] == (gpu_particles, 298.15, 101325.0, 1.0)
        assert kwargs["volume"] is gpu_particles.volume
        assert kwargs["rng_seed"] == 41
        assert kwargs["initialize_rng"] is (index == 0)
    first_kwargs = _FakeState.calls[0][1]
    second_kwargs = _FakeState.calls[1][1]
    for name in ("collision_pairs", "n_collisions", "rng_states"):
        assert first_kwargs[name] is second_kwargs[name]
    assert result.collision_pairs is first_kwargs["collision_pairs"]
    assert result.n_collisions is first_kwargs["n_collisions"]
    assert result.rng_states is first_kwargs["rng_states"]
    assert result.collision_pairs.shape == (1, 4, 2)
    assert result.n_collisions.shape == result.rng_states.shape == (1,)
    assert "selected-adapter dispatch" in result.output[2]
    assert "Selected Brownian coagulation complete" in result.output[3]


@pytest.mark.parametrize(
    "failure_at",
    ["loader", "conversion", "first", "second", "synchronize", "restore"],
)
def test_failures_propagate_without_fallback_or_restore(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
    failure_at: str,
) -> None:
    """Test every enabled-path boundary propagates without fallback work."""
    _reset_fakes()
    restored = False

    class FailingAdapter(_FakeAdapter):
        """Fail a selected dispatch at the requested call number."""

        calls: list[_FakeExecutionState] = []

        def execute(self, execution_state: _FakeExecutionState) -> Any:
            if failure_at == ("first" if not self.calls else "second"):
                raise RuntimeError(failure_at)
            return super().execute(execution_state)

    def restore(value: Any) -> Any:
        nonlocal restored
        restored = True
        if failure_at == "restore":
            raise RuntimeError("restore")
        return example_module._build_particle_data()

    class FailingWP(_FakeWP):
        """Optionally fail caller synchronization."""

        @classmethod
        def synchronize(cls) -> None:
            if failure_at == "synchronize":
                raise RuntimeError("synchronize")
            super().synchronize()

    def conversion(data: Any, device: str | None = None) -> Any:
        """Return a fake GPU particle container or raise the requested error."""
        del data, device
        if failure_at == "conversion":
            raise RuntimeError("conversion")
        return types.SimpleNamespace(volume=object())

    def load_helpers() -> tuple[bool, Any, Any]:
        """Return the fake helper tuple used by the selected-adapter path."""
        return True, conversion, restore

    monkeypatch.setattr(example_module, "_load_gpu_helpers", load_helpers)
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        (
            lambda: (
                FailingWP,
                _FakeConfig,
                _FakeState,
                _FakeExecutionState,
                FailingAdapter,
            )
        )
        if failure_at != "loader"
        else lambda: (_ for _ in ()).throw(RuntimeError("loader")),
    )

    with pytest.raises(RuntimeError, match=failure_at):
        example_module.run_example()
    assert restored is (failure_at == "restore")


@pytest.mark.warp
def test_real_warp_selected_adapter_reuses_rng_and_preserves_identity(
    example_module: types.ModuleType,
) -> None:
    """Test the real selected route retains supplied sidecars by identity."""
    gpu = pytest.importorskip("particula.gpu")
    if not gpu.WARP_AVAILABLE:
        pytest.skip("Warp is not available")
    result = example_module.run_example(device="cpu")

    assert result.particle_data is not None
    assert result.collision_pairs is not None
    assert result.n_collisions is not None
    assert result.rng_states is not None
    assert result.collision_pairs.shape == (1, 4, 2)
    assert result.n_collisions.shape == result.rng_states.shape == (1,)
    assert 0 <= int(result.n_collisions.numpy()[0]) <= 4
