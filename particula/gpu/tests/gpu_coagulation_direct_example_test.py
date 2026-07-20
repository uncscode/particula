"""Regression tests for the direct GPU coagulation example."""

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
    / "docs"
    / "Examples"
    / "gpu_coagulation_direct.py"
)
EXAMPLES_ROOT = EXAMPLE_PATH.parent
LAZY_IMPORTS = (
    "particula.gpu",
    "warp",
    "particula.gpu.kernels",
    "particula.gpu.kernels.coagulation",
)
RUNTIME_IMPORTS = LAZY_IMPORTS[1:]
DISABLED_OUTPUT = [
    "Canonical path: docs/Examples/gpu_coagulation_direct.py",
    "ParticleData constructed: masses=(1, 8, 1), concentration=(1, 8), charge=(1, 8), density=(1,), volume=(1,)",
    "Warp is unavailable or disabled; no kernel ran.",
]


def _require_warp_gpu() -> Any:
    """Return GPU helpers only from a Warp-enabled test path."""
    gpu = pytest.importorskip("particula.gpu")
    if not gpu.WARP_AVAILABLE:
        pytest.skip("Warp is not available")
    return gpu


@pytest.fixture
def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load the standalone example with no retained module state."""
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
    monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
    return importlib.import_module("gpu_coagulation_direct")


def test_cpu_fixture_has_documented_active_and_inactive_slots(
    example_module: types.ModuleType,
) -> None:
    """Test the literal fixture schema and finite signed charge state."""
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
    assert np.all(particle_data.masses >= 0.0)
    assert np.all(np.isfinite(particle_data.charge))
    assert np.any(particle_data.charge < 0.0)
    assert np.any(particle_data.charge > 0.0)
    assert np.all(particle_data.density > 0.0)
    assert np.all(particle_data.volume > 0.0)
    np.testing.assert_array_equal(particle_data.density, [1000.0])
    np.testing.assert_array_equal(particle_data.volume, [1.0e-18])


def test_forced_disabled_routes_never_import_or_execute_gpu_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test disabled paths avoid availability, transfer, and runtime work."""
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
    with monkeypatch.context() as cleanup:
        for module_name in LAZY_IMPORTS:
            cleanup.delitem(sys.modules, module_name, raising=False)
        module = importlib.import_module("gpu_coagulation_direct")
        failures: list[str] = []

        def fail_if_called(operation: str) -> None:
            failures.append(operation)
            pytest.fail(f"disabled path invoked {operation}")

        cleanup.setattr(
            module,
            "_load_gpu_helpers",
            lambda: fail_if_called("GPU availability/helper loading"),
        )
        cleanup.setattr(
            module,
            "_load_gpu_runtime",
            lambda: fail_if_called("Warp/kernel runtime loading"),
        )
        result = module.run_example()
        module.main()
        assert result.output == DISABLED_OUTPUT
        assert result.particle_data is None
        assert result.collision_pairs is None
        assert result.n_collisions is None
        assert result.rng_states is None
        assert capsys.readouterr().out.splitlines() == DISABLED_OUTPUT
        assert failures == []
        assert all(
            module_name not in sys.modules for module_name in LAZY_IMPORTS
        )

    process = subprocess.run(  # noqa: S603
        [sys.executable, str(EXAMPLE_PATH)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
        timeout=10,
    )
    assert process.stdout.splitlines() == DISABLED_OUTPUT


def test_unavailable_warp_does_not_reach_loader_or_conversion(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test the unavailable path has neither conversion nor kernel fallback."""
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module,
        "_load_gpu_helpers",
        lambda: (
            False,
            lambda *args, **kwargs: pytest.fail("conversion ran"),
            lambda *args, **kwargs: pytest.fail("restoration ran"),
        ),
    )
    monkeypatch.setattr(
        example_module, "_load_gpu_runtime", lambda: pytest.fail("loader ran")
    )

    assert example_module.run_example().output == DISABLED_OUTPUT
    result = example_module.run_example()
    assert result.particle_data is None
    assert result.collision_pairs is None
    assert result.n_collisions is None
    assert result.rng_states is None


def test_loader_uses_exact_lazy_import_order(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test only the documented Warp modules are loaded, in order."""
    imported: list[str] = []
    wp, step, config = object(), object(), object()
    modules = {
        "warp": wp,
        "particula.gpu.kernels": types.SimpleNamespace(
            coagulation_step_gpu=step
        ),
        "particula.gpu.kernels.coagulation": types.SimpleNamespace(
            CoagulationMechanismConfig=config
        ),
    }

    def fake_import(name: str) -> object:
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(example_module.importlib, "import_module", fake_import)
    assert example_module._load_gpu_runtime() == (wp, step, config)
    assert imported == list(RUNTIME_IMPORTS)


def test_helper_loader_imports_only_gpu_transfer_helpers(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test availability and explicit transfer helpers load lazily together."""
    imported: list[str] = []
    to_warp, from_warp = object(), object()
    gpu = types.SimpleNamespace(
        WARP_AVAILABLE=True,
        to_warp_particle_data=to_warp,
        from_warp_particle_data=from_warp,
    )

    def fake_import(name: str) -> object:
        imported.append(name)
        assert name == "particula.gpu"
        return gpu

    monkeypatch.setattr(example_module.importlib, "import_module", fake_import)

    assert example_module._load_gpu_helpers() == (True, to_warp, from_warp)
    assert imported == ["particula.gpu"]


class _FakeArray:
    """Small Warp-array fake retaining values and allocation metadata."""

    def __init__(
        self, shape: tuple[int, ...], dtype: object, device: str
    ) -> None:
        numpy_dtype = np.int32 if dtype is _FakeWP.int32 else np.uint32
        self.values = np.zeros(shape, dtype=numpy_dtype)
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def numpy(self) -> np.ndarray:
        """Return fake storage for assertions."""
        return self.values


class _FakeWP:
    """Fake Warp allocator with distinct dtype sentinels."""

    int32 = object()
    uint32 = object()

    @classmethod
    def zeros(
        cls, shape: tuple[int, ...], dtype: object, device: str
    ) -> _FakeArray:
        """Allocate a metadata-preserving fake array."""
        return _FakeArray(shape, dtype, device)


class _FakeConfig:
    """Record host-only configuration arguments."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def test_enabled_path_uses_shared_sidecars_and_restores_after_two_calls(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test sidecar schema, call arguments, identities, and restoration order."""
    events: list[str] = []
    calls: list[tuple[Any, dict[str, Any]]] = []
    conversion_devices: list[str] = []
    gpu_particles = types.SimpleNamespace(volume=object())
    restored = example_module._build_particle_data()

    def fake_step(particles: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        events.append("step")
        calls.append((particles, kwargs))
        return particles, kwargs["collision_pairs"], kwargs["n_collisions"]

    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module,
        "_load_gpu_helpers",
        lambda: (
            True,
            lambda data, device: _record_conversion(
                events,
                conversion_devices,
                device,
                gpu_particles,
            ),
            lambda data: _record_restore(events, restored),
        ),
    )
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        lambda: (_FakeWP, fake_step, _FakeConfig),
    )

    result = example_module.run_example()

    assert events == ["convert", "step", "step", "restore"]
    assert conversion_devices == ["cpu"]
    assert result.particle_data is restored
    assert result.output == [
        "Canonical path: docs/Examples/gpu_coagulation_direct.py",
        "ParticleData constructed: masses=(1, 8, 1), concentration=(1, 8), charge=(1, 8), density=(1,), volume=(1,)",
        "Explicit helpers: CPU→Warp conversion -> direct coagulation -> CPU checkpoint",
        "Direct Brownian coagulation complete: device=cpu, calls=2, collision_pairs=(1, 1, 2), n_collisions=(1,)",
        "Final checkpoint restored: particle_masses=(1, 8, 1)",
        "Three-item direct return; collision and RNG sidecars remain caller-owned.",
        "Persistent RNG state is initialized once and reused by the second call.",
    ]
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] is gpu_particles
    for index, (_, call) in enumerate(calls):
        assert call["temperature"] == 298.15
        assert call["pressure"] == 101325.0
        assert call["time_step"] == 1.0
        assert call["volume"] is gpu_particles.volume
        assert call["max_collisions"] == 1
        assert call["rng_seed"] == 41
        assert call["initialize_rng"] is (index == 0)
        assert call["mechanism_config"].mechanisms == ("brownian",)
        assert call["mechanism_config"].distribution_type == "particle_resolved"
    assert calls[0][1]["collision_pairs"] is calls[1][1]["collision_pairs"]
    assert calls[0][1]["n_collisions"] is calls[1][1]["n_collisions"]
    assert calls[0][1]["rng_states"] is calls[1][1]["rng_states"]
    assert result.collision_pairs is calls[1][1]["collision_pairs"]
    assert result.n_collisions is calls[1][1]["n_collisions"]
    assert result.rng_states is calls[1][1]["rng_states"]
    assert result.collision_pairs.shape == (1, 1, 2)
    assert result.n_collisions.shape == (1,)
    assert result.rng_states.shape == (1,)
    assert result.collision_pairs.dtype is _FakeWP.int32
    assert result.n_collisions.dtype is _FakeWP.int32
    assert result.rng_states.dtype is _FakeWP.uint32
    assert result.collision_pairs.device == "cpu"
    assert result.n_collisions.device == "cpu"
    assert result.rng_states.device == "cpu"
    assert result.collision_pairs.numpy().dtype == np.int32
    assert result.n_collisions.numpy().dtype == np.int32
    assert result.rng_states.numpy().dtype == np.uint32
    np.testing.assert_array_equal(result.collision_pairs.numpy(), 0)
    np.testing.assert_array_equal(result.n_collisions.numpy(), 0)
    np.testing.assert_array_equal(result.rng_states.numpy(), 0)


def _record_conversion(
    events: list[str],
    conversion_devices: list[str],
    device: str,
    gpu_particles: Any,
) -> Any:
    """Record a conversion event and return the GPU particle fixture."""
    events.append("convert")
    conversion_devices.append(device)
    return gpu_particles


def _record_restore(events: list[str], restored: Any) -> Any:
    """Record a restore event and return the restored fixture."""
    events.append("restore")
    return restored


@pytest.mark.parametrize(
    "failure_at", ["loader", "conversion", "first", "second", "restore"]
)
def test_failures_propagate_without_success_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    example_module: types.ModuleType,
    failure_at: str,
) -> None:
    """Test every upstream boundary propagates without partial metadata."""
    calls = 0
    restored = False

    def fake_step(particles: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        nonlocal calls
        calls += 1
        if failure_at == ("first" if calls == 1 else "second"):
            raise RuntimeError(failure_at)
        return particles, kwargs["collision_pairs"], kwargs["n_collisions"]

    def fake_restore(value: Any) -> Any:
        nonlocal restored
        restored = True
        if failure_at == "restore":
            raise RuntimeError("restore")
        return example_module._build_particle_data()

    to_warp_particle_data = (
        (lambda data, device: types.SimpleNamespace(volume=object()))
        if failure_at != "conversion"
        else lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("conversion")
        )
    )
    monkeypatch.setattr(
        example_module,
        "_load_gpu_helpers",
        lambda: (True, to_warp_particle_data, fake_restore),
    )
    monkeypatch.setattr(
        example_module,
        "_load_gpu_runtime",
        (lambda: (_FakeWP, fake_step, _FakeConfig))
        if failure_at != "loader"
        else lambda: (_ for _ in ()).throw(RuntimeError("loader")),
    )
    with pytest.raises(RuntimeError, match=failure_at):
        example_module.main()
    assert capsys.readouterr().out == ""
    assert restored is (failure_at == "restore")
    if failure_at == "restore":
        assert calls == 2
    elif failure_at in {"loader", "conversion", "first", "second"}:
        assert not restored
        assert (
            calls
            == {"loader": 0, "conversion": 0, "first": 1, "second": 2}[
                failure_at
            ]
        )


@pytest.mark.warp
@pytest.mark.stochastic
def test_real_warp_cpu_path_preserves_invariants_and_persistent_rng(
    monkeypatch: pytest.MonkeyPatch,
    example_module: types.ModuleType,
) -> None:
    """Test real CPU Warp sidecar reuse and conservation-safe invariants."""
    _require_warp_gpu()
    original_loader = example_module._load_gpu_runtime
    states: list[np.ndarray] = []
    active_before_second: list[np.ndarray] = []
    sidecar_ids: list[tuple[int, int, int]] = []
    initial = example_module._build_particle_data()

    def loader_with_spy() -> tuple[Any, Any, Any]:
        wp, step, config = original_loader()

        def spy(particles: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
            sidecar_ids.append(
                (
                    id(kwargs["collision_pairs"]),
                    id(kwargs["n_collisions"]),
                    id(kwargs["rng_states"]),
                )
            )
            states.append(kwargs["rng_states"].numpy().copy())
            if len(states) == 3:
                active_before_second.append(
                    np.flatnonzero(particles.concentration.numpy()[0] > 0.0)
                )
            result = step(particles, **kwargs)
            states.append(kwargs["rng_states"].numpy().copy())
            return result

        return wp, spy, config

    monkeypatch.setattr(example_module, "_load_gpu_runtime", loader_with_spy)
    result = example_module.run_example(device="cpu")

    assert result.particle_data is not None
    assert result.collision_pairs is not None
    assert result.n_collisions is not None
    assert result.rng_states is not None
    assert sidecar_ids[0] == sidecar_ids[1]
    assert sidecar_ids[1] == (
        id(result.collision_pairs),
        id(result.n_collisions),
        id(result.rng_states),
    )
    np.testing.assert_array_equal(states[1], states[2])
    assert not np.array_equal(states[0], states[1])
    assert not np.array_equal(states[2], states[3])
    pairs = result.collision_pairs.numpy()[0]
    count = int(result.n_collisions.numpy()[0])
    assert 0 <= count <= 1
    assert np.unique(pairs[:count]).size == 2 * count
    for recipient, donor in pairs[:count]:
        assert recipient < donor
        assert recipient in active_before_second[0]
        assert donor in active_before_second[0]
    initial_inventory = np.sum(
        initial.masses * initial.concentration[:, :, None]
    )
    final_inventory = np.sum(
        result.particle_data.masses
        * result.particle_data.concentration[:, :, None]
    )
    np.testing.assert_allclose(
        final_inventory, initial_inventory, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        np.sum(result.particle_data.charge),
        np.sum(initial.charge),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        result.particle_data.masses[:, 6:], initial.masses[:, 6:]
    )
    np.testing.assert_array_equal(
        result.particle_data.concentration[:, 6:], initial.concentration[:, 6:]
    )
    np.testing.assert_array_equal(
        result.particle_data.charge[:, 6:], initial.charge[:, 6:]
    )


@pytest.mark.warp
@pytest.mark.parametrize("active_count", [0, 1])
def test_real_warp_zero_and_one_active_inputs_are_unchanged(
    example_module: types.ModuleType, active_count: int
) -> None:
    """Test degenerate valid fixtures return zero collisions without mutation."""
    gpu = _require_warp_gpu()
    wp, step, config_type = example_module._load_gpu_runtime()
    particle_data = example_module._build_particle_data()
    particle_data.concentration[0, active_count:] = 0.0
    before = particle_data.copy()
    particles = gpu.to_warp_particle_data(particle_data, device="cpu")
    pairs = wp.zeros((1, 1, 2), dtype=wp.int32, device="cpu")
    counts = wp.zeros((1,), dtype=wp.int32, device="cpu")
    states = wp.zeros((1,), dtype=wp.uint32, device="cpu")
    returned, returned_pairs, returned_counts = step(
        particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=particles.volume,
        max_collisions=1,
        rng_seed=41,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        mechanism_config=config_type(
            mechanisms=("brownian",), distribution_type="particle_resolved"
        ),
        initialize_rng=True,
    )
    assert returned is particles
    assert returned_pairs is pairs
    assert returned_counts is counts
    np.testing.assert_array_equal(counts.numpy(), [0])
    np.testing.assert_array_equal(particles.masses.numpy(), before.masses)
    np.testing.assert_array_equal(
        particles.concentration.numpy(), before.concentration
    )
    np.testing.assert_array_equal(particles.charge.numpy(), before.charge)
