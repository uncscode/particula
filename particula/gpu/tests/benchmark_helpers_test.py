"""Fast tests for benchmark-study helper behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest


def _load_benchmark_module(monkeypatch: pytest.MonkeyPatch):
    """Load the benchmark module without running its opt-in skip gates."""
    module_path = Path(__file__).with_name("benchmark_test.py")
    module_name = "particula_gpu_benchmark_test_fast_import"
    fake_cuda_module = types.ModuleType(
        "particula.gpu.tests.cuda_availability"
    )
    fake_cuda_module.cuda_available = lambda _wp: True

    monkeypatch.setitem(
        sys.modules,
        "particula.gpu.tests.cuda_availability",
        fake_cuda_module,
    )
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--benchmark"])

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark_module(monkeypatch: pytest.MonkeyPatch):
    """Load the opt-in benchmark module for fast helper tests."""
    return _load_benchmark_module(monkeypatch)


def test_benchmark_enabled_detects_opt_in_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark helper reports whether the opt-in flag is present."""
    monkeypatch.setattr(sys, "argv", ["pytest", "--benchmark"])
    module = _load_benchmark_module(monkeypatch)

    assert module._benchmark_enabled() is True


def test_save_results_writes_json_snapshot(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Benchmark snapshots are flushed to the configured output path."""
    output_path = tmp_path / "gpu_benchmark_results.json"
    monkeypatch.setattr(benchmark_module, "BENCHMARK_OUTPUT", output_path)
    monkeypatch.setattr(
        benchmark_module,
        "_benchmark_results",
        {"started_at": "2026-01-01T00:00:00+00:00", "benchmarks": {}},
    )

    benchmark_module._save_results()

    written = output_path.read_text()
    assert '"started_at": "2026-01-01T00:00:00+00:00"' in written
    assert '"updated_at":' in written


def test_skip_if_no_cuda_skips_when_cuda_unavailable(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark helpers skip cleanly when CUDA is unavailable."""
    monkeypatch.setattr(benchmark_module, "cuda_available", lambda _wp: False)

    with pytest.raises(pytest.skip.Exception, match="CUDA not available"):
        benchmark_module._skip_if_no_cuda()


def test_warp_profiled_without_profile_env_is_passthrough(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile wrapper yields directly when profiling is disabled."""
    monkeypatch.setenv("WARP_PROFILE", "0")
    calls: list[str] = []

    with benchmark_module._warp_profiled("tag"):
        calls.append("body")

    assert calls == ["body"]


def test_warp_profiled_uses_capture_hooks_when_available(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile wrapper prefers Warp capture hooks when present."""
    monkeypatch.setenv("WARP_PROFILE", "1")
    events: list[str] = []
    fake_wp = types.SimpleNamespace(
        capture_begin=lambda tag: events.append(f"begin:{tag}"),
        capture_end=lambda: events.append("end"),
    )
    monkeypatch.setattr(benchmark_module, "wp", fake_wp)

    with benchmark_module._warp_profiled("captured"):
        events.append("body")

    assert events == ["begin:captured", "body", "end"]


def test_warp_profiled_falls_back_to_profiler_object(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile wrapper uses the profiler fallback when capture hooks are absent."""
    monkeypatch.setenv("WARP_PROFILE", "1")
    events: list[str] = []

    class _FakeProfiler:
        def begin(self) -> None:
            events.append("begin")

        def end(self) -> None:
            events.append("end")

    monkeypatch.setattr(
        benchmark_module,
        "wp",
        types.SimpleNamespace(profiler=_FakeProfiler()),
    )

    with benchmark_module._warp_profiled("profiled"):
        events.append("body")

    assert events == ["begin", "body", "end"]


def test_timing_helpers_run_expected_iterations(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU and GPU timing helpers execute warmup and timed steps."""
    gpu_calls: list[str] = []
    cpu_calls: list[str] = []
    sync_calls: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "wp",
        types.SimpleNamespace(synchronize=lambda: sync_calls.append("sync")),
    )

    def gpu_step() -> None:
        gpu_calls.append("step")

    def cpu_step() -> None:
        cpu_calls.append("step")

    gpu_elapsed = benchmark_module._time_gpu_loop(gpu_step, steps=3, warmup=2)
    cpu_elapsed = benchmark_module._time_cpu_loop(cpu_step, steps=4, warmup=1)

    assert gpu_elapsed >= 0.0
    assert cpu_elapsed >= 0.0
    assert gpu_calls == ["step"] * 5
    assert cpu_calls == ["step"] * 5
    assert sync_calls == ["sync", "sync"]


def test_compute_speedup_returns_ratio_and_skips_invalid_data(
    benchmark_module,
) -> None:
    """Speedup helper returns a ratio and rejects non-positive timings."""
    assert benchmark_module._compute_speedup(6.0, 2.0) == 3.0

    with pytest.raises(pytest.skip.Exception, match="Invalid timing data"):
        benchmark_module._compute_speedup(0.0, 2.0)


def test_benchmark_data_builders_return_expected_shapes(benchmark_module) -> None:
    """Study benchmark builders return deterministic array shapes and dtypes."""
    particles = benchmark_module._make_particle_data(2, 3, 2, concentration_scale=7.0)
    gas = benchmark_module._make_gas_data(2, 2)
    vapor_pressure = benchmark_module._make_vapor_pressure(2, 2)

    assert particles.masses.shape == (2, 3, 2)
    assert particles.concentration.shape == (2, 3)
    assert np.all(particles.concentration == 7.0)
    assert particles.charge.shape == (2, 3)
    assert particles.density.shape == (2,)
    assert particles.volume.shape == (2,)
    assert gas.concentration.shape == (2, 2)
    assert gas.partitioning.dtype == bool
    assert gas.name == ["species_0", "species_1"]
    assert vapor_pressure.shape == (2, 2)
    npt.assert_allclose(vapor_pressure[1], np.array([850.0, 850.0]))


def test_cpu_mass_transfer_handles_zero_concentration_and_zero_volume(
    benchmark_module,
) -> None:
    """CPU mass-transfer helper stays deterministic on skipped particle paths."""
    particles = benchmark_module._make_particle_data(1, 2, 1)
    particles.concentration[0, 0] = 0.0
    particles.masses[0, 1, 0] = 0.0
    gas = benchmark_module._make_gas_data(1, 1)
    vapor_pressure = benchmark_module._make_vapor_pressure(1, 1)

    result = benchmark_module._cpu_mass_transfer(
        particles,
        gas,
        vapor_pressure,
        np.array([benchmark_module.DEFAULT_SURFACE_TENSION], dtype=np.float64),
        np.array([benchmark_module.DEFAULT_MASS_ACCOMMODATION], dtype=np.float64),
        np.array([benchmark_module.DEFAULT_DIFFUSION_COEFFICIENT], dtype=np.float64),
        benchmark_module.DEFAULT_TEMPERATURE,
        benchmark_module.DEFAULT_PRESSURE,
        benchmark_module.DEFAULT_TIME_STEP,
    )

    assert result.shape == particles.masses.shape
    npt.assert_array_equal(result, np.zeros_like(result))


def test_cpu_condensation_step_clamps_negative_updated_mass(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU condensation helper clamps negative updated masses to zero."""
    particles = benchmark_module._make_particle_data(1, 2, 1)
    initial = particles.masses.copy()
    gas = benchmark_module._make_gas_data(1, 1)
    transfer = np.array([[[-2.0e-18], [1.0e-19]]], dtype=np.float64)
    monkeypatch.setattr(
        benchmark_module,
        "_cpu_mass_transfer",
        lambda *args, **kwargs: transfer,
    )

    benchmark_module._cpu_condensation_step(
        particles,
        gas,
        benchmark_module._make_vapor_pressure(1, 1),
        np.array([benchmark_module.DEFAULT_SURFACE_TENSION], dtype=np.float64),
        np.array([benchmark_module.DEFAULT_MASS_ACCOMMODATION], dtype=np.float64),
        np.array([benchmark_module.DEFAULT_DIFFUSION_COEFFICIENT], dtype=np.float64),
        benchmark_module.DEFAULT_TEMPERATURE,
        benchmark_module.DEFAULT_PRESSURE,
        benchmark_module.DEFAULT_TIME_STEP,
        np.zeros_like(initial),
    )

    expected = np.maximum(0.0, initial + transfer)
    npt.assert_allclose(particles.masses, expected)


def test_build_kernel_radius_handles_empty_and_nonempty_inputs(
    benchmark_module,
) -> None:
    """Kernel-radius helper returns bounded interpolation grids."""
    empty_grid = benchmark_module._build_kernel_radius(np.zeros(3, dtype=np.float64))
    populated_grid = benchmark_module._build_kernel_radius(
        np.array([0.0, 2.0e-9, 5.0e-9], dtype=np.float64)
    )

    assert empty_grid.shape == (32,)
    assert populated_grid.shape == (64,)
    assert populated_grid[0] >= 1.0e-9
    assert populated_grid[-1] > populated_grid[0]


def test_cpu_coagulation_step_updates_colliding_particles(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU coagulation helper zeroes collided concentration and updates masses."""
    particles = benchmark_module._make_particle_data(1, 3, 2)
    original_masses = particles.masses.copy()
    original_density = particles.density.copy()
    monkeypatch.setattr(
        benchmark_module,
        "get_brownian_kernel_via_system_state",
        lambda **kwargs: np.ones((4, 4), dtype=np.float64),
    )
    monkeypatch.setattr(
        benchmark_module,
        "get_particle_resolved_coagulation_step",
        lambda *args, **kwargs: np.array([[0, 1]], dtype=np.int64),
    )
    monkeypatch.setattr(
        benchmark_module,
        "get_particle_resolved_update_step",
        lambda radii, *_args: (
            np.array([0.0, radii[1] * 1.5, radii[2]], dtype=np.float64),
            None,
            None,
        ),
    )

    benchmark_module._cpu_coagulation_step(
        particles,
        benchmark_module.DEFAULT_TEMPERATURE,
        benchmark_module.DEFAULT_PRESSURE,
        benchmark_module.DEFAULT_TIME_STEP,
        np.random.default_rng(42),
        np.linspace(1.0e-9, 1.0e-8, 4),
    )

    assert particles.concentration[0, 0] == 0.0
    assert np.all(particles.masses[0, 0] == 0.0)
    total_mass_before = np.sum(original_masses[0, 1])
    mass_fractions_before = original_masses[0, 1] / total_mass_before
    npt.assert_allclose(
        particles.masses[0, 1] / np.sum(particles.masses[0, 1]),
        mass_fractions_before,
    )
    assert np.sum(particles.masses[0, 1]) > total_mass_before
    npt.assert_allclose(
        particles.density,
        original_density,
    )


def test_print_timing_emits_speedup_summary(
    benchmark_module,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Timing printer reports GPU, CPU, and speedup values."""
    benchmark_module._print_timing("Case A", gpu_time=2.0, cpu_time=10.0)

    captured = capsys.readouterr()
    assert "Case A: GPU 2.0000s | CPU 10.0000s | speedup 5.00x" in captured.out


def test_project_mass_precision_candidate_reconstructs_expected_shapes(
    benchmark_module,
) -> None:
    """Supported study candidates reconstruct arrays with stable shapes."""
    case = benchmark_module._build_mass_precision_cases()[2]

    for candidate_id in (
        "fp32_absolute_mass",
        "mixed_precision_mass_plus_density",
        "fp32_total_mass_fp32_mass_fraction",
    ):
        reconstructed = benchmark_module._project_mass_precision_candidate(
            case.masses,
            candidate_id,
        )
        assert reconstructed.shape == case.masses.shape
        assert reconstructed.dtype == np.float64


def test_project_mass_precision_candidate_handles_zero_total_mass(
    benchmark_module,
) -> None:
    """Zero-total-mass inputs reconstruct zeros without instability."""
    zero_masses = np.zeros((2, 3, 2), dtype=np.float64)

    reconstructed = benchmark_module._project_mass_precision_candidate(
        zero_masses,
        "fp32_total_mass_fp32_mass_fraction",
    )

    npt.assert_array_equal(reconstructed, zero_masses)


def test_project_mass_precision_candidate_rejects_unsupported_candidate(
    benchmark_module,
) -> None:
    """Unsupported candidate ids fail with a stable error message."""
    case = benchmark_module._build_mass_precision_cases()[0]

    with pytest.raises(ValueError, match="Unsupported candidate id"):
        benchmark_module._project_mass_precision_candidate(
            case.masses,
            "unsupported",
        )


def test_mass_precision_projection_benchmark_records_bounded_result(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark entry point records bounded study metadata without CUDA."""
    monkeypatch.setattr(benchmark_module, "_skip_if_no_cuda", lambda: None)
    monkeypatch.setattr(benchmark_module, "_save_results", lambda: None)

    label, case_index, candidate_id = (
        benchmark_module._MASS_PRECISION_BENCHMARK_CONFIGS[0]
    )
    benchmark_module.test_mass_precision_projection_benchmark(
        label,
        case_index,
        candidate_id,
    )

    result = benchmark_module._benchmark_results[
        "benchmarks"
    ][f"mass_precision_projection_{label}_{candidate_id}"]
    assert result["case_name"] == label
    assert result["candidate_id"] == candidate_id
    assert result["repeats"] == 5_000
    assert result["elapsed_s"] >= 0.0
    assert result["mean_us"] >= 0.0
