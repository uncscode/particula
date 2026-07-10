"""Fast tests for benchmark-study helper behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
import warnings
from builtins import __import__ as builtins_import
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu.tests.mass_precision_study_support import (
    _build_mass_precision_cases,
    _project_candidate,
)


def _load_benchmark_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    benchmark_enabled: bool = True,
):
    """Load the benchmark module without running its opt-in skip gates."""
    module_path = Path(__file__).with_name("benchmark_test.py")
    module_name = "particula_gpu_benchmark_test_fast_import"
    argv = ["pytest"]
    if benchmark_enabled:
        argv.append("--benchmark")
    monkeypatch.setattr(sys, "argv", argv)

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


def test_benchmark_module_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark module keeps the explicit --benchmark import gate."""
    with pytest.raises(
        pytest.skip.Exception,
        match=r"GPU benchmarks skipped \(pass --benchmark to enable\)",
    ):
        _load_benchmark_module(monkeypatch, benchmark_enabled=False)


def test_benchmark_module_skips_before_warp_import_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-opt-in import skips before attempting to import Warp."""

    def _guarded_import(name, *args, **kwargs):
        if name == "warp":
            raise AssertionError(
                "warp import should not run before opt-in skip"
            )
        return builtins_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _guarded_import)

    with pytest.raises(
        pytest.skip.Exception,
        match=r"GPU benchmarks skipped \(pass --benchmark to enable\)",
    ):
        _load_benchmark_module(monkeypatch, benchmark_enabled=False)


def test_benchmark_enabled_detects_opt_in_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark helper reports whether the opt-in flag is present."""
    monkeypatch.setattr(sys, "argv", ["pytest", "--benchmark"])
    module = _load_benchmark_module(monkeypatch)

    assert module._benchmark_enabled() is True


def test_parse_positive_int_env_uses_default_without_override(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive-int env helper falls back to the provided default."""
    monkeypatch.delenv("BENCHMARK_MAX_BYTES", raising=False)

    assert (
        benchmark_module._parse_positive_int_env("BENCHMARK_MAX_BYTES", 9) == 9
    )


@pytest.mark.parametrize("raw_value", ["0", "-1", "not-an-int"])
def test_parse_positive_int_env_rejects_invalid_overrides(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """Positive-int env helper rejects invalid overrides clearly."""
    monkeypatch.setenv("BENCHMARK_MAX_BYTES", raw_value)

    with pytest.raises(ValueError, match="BENCHMARK_MAX_BYTES"):
        benchmark_module._parse_positive_int_env("BENCHMARK_MAX_BYTES", 9)


def test_sanitize_benchmark_output_name_keeps_filename_only(
    benchmark_module,
) -> None:
    """Benchmark output names are normalized to artifact-safe filenames."""
    assert (
        benchmark_module._sanitize_benchmark_output_name(
            " nested/results.json "
        )
        == "results.json"
    )


@pytest.mark.parametrize("raw_value", ["", "   ", ".", ".."])
def test_sanitize_benchmark_output_name_rejects_empty_candidates(
    benchmark_module,
    raw_value: str,
) -> None:
    """Benchmark output names must resolve to non-empty filenames."""
    with pytest.raises(ValueError, match="non-empty filename"):
        benchmark_module._sanitize_benchmark_output_name(raw_value)


def test_get_benchmark_output_path_stays_within_artifact_directory(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output path resolution keeps overrides rooted in the artifact folder."""
    monkeypatch.setenv("BENCHMARK_OUTPUT", "../custom.json")

    output_path = benchmark_module._get_benchmark_output_path()

    assert (
        output_path == benchmark_module.BENCHMARK_ARTIFACT_DIR / "custom.json"
    )


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


def test_save_results_creates_parent_directories(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Benchmark snapshots create missing parent directories first."""
    output_path = tmp_path / "nested" / "gpu_benchmark_results.json"
    monkeypatch.setattr(benchmark_module, "BENCHMARK_OUTPUT", output_path)
    monkeypatch.setattr(
        benchmark_module,
        "_benchmark_results",
        {"started_at": "2026-01-01T00:00:00+00:00", "benchmarks": {}},
    )

    benchmark_module._save_results()

    assert output_path.exists()


def test_save_results_surfaces_write_failures(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Benchmark snapshot failures raise a clear RuntimeError."""
    output_path = tmp_path / "gpu_benchmark_results.json"
    monkeypatch.setattr(benchmark_module, "BENCHMARK_OUTPUT", output_path)

    def _raise_write_error(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _raise_write_error)

    with pytest.raises(RuntimeError, match="Failed to write benchmark results"):
        benchmark_module._save_results()


def test_skip_if_no_cuda_skips_when_cuda_unavailable(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark helpers skip cleanly when CUDA is unavailable."""
    monkeypatch.setattr(benchmark_module, "cuda_available", lambda _wp: False)

    with pytest.raises(pytest.skip.Exception, match="Warp/CUDA not available"):
        benchmark_module._skip_if_no_cuda()


def test_skip_if_no_cuda_skips_when_warp_is_missing(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-only helper coverage remains importable without Warp."""
    monkeypatch.setattr(benchmark_module, "wp", None)

    with pytest.raises(pytest.skip.Exception, match="Warp/CUDA not available"):
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


def test_benchmark_budget_helpers_estimate_and_gate_large_cases(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget helpers estimate bytes and skip only when limits are exceeded."""
    condensation_budget = benchmark_module._estimate_condensation_budget(
        "cond", 2, 3, 4, True
    )
    coagulation_budget = benchmark_module._estimate_coagulation_budget(
        "coag", 2, 3, 4, False
    )

    assert condensation_budget.label == "cond"
    assert coagulation_budget.label == "coag"
    assert condensation_budget.total_bytes > condensation_budget.gpu_bytes > 0
    assert coagulation_budget.total_bytes > coagulation_budget.cpu_bytes > 0

    monkeypatch.setenv(
        "BENCHMARK_MAX_BYTES", str(condensation_budget.total_bytes - 1)
    )
    with pytest.raises(
        pytest.skip.Exception, match="exceeding BENCHMARK_MAX_BYTES"
    ):
        benchmark_module._validate_benchmark_budget(condensation_budget)

    monkeypatch.setenv(
        "BENCHMARK_MAX_BYTES", str(condensation_budget.total_bytes)
    )
    benchmark_module._validate_benchmark_budget(condensation_budget)


def test_benchmark_data_builders_return_expected_shapes(
    benchmark_module,
) -> None:
    """Study benchmark builders return deterministic array shapes and dtypes."""
    particles = benchmark_module._make_particle_data(
        2, 3, 2, concentration_scale=7.0
    )
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error")
        result = benchmark_module._cpu_mass_transfer(
            particles,
            gas,
            vapor_pressure,
            np.array(
                [benchmark_module.DEFAULT_SURFACE_TENSION], dtype=np.float64
            ),
            np.array(
                [benchmark_module.DEFAULT_MASS_ACCOMMODATION],
                dtype=np.float64,
            ),
            np.array(
                [benchmark_module.DEFAULT_DIFFUSION_COEFFICIENT],
                dtype=np.float64,
            ),
            benchmark_module.DEFAULT_TEMPERATURE,
            benchmark_module.DEFAULT_PRESSURE,
            benchmark_module.DEFAULT_TIME_STEP,
        )

    assert not caught
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
        np.array(
            [benchmark_module.DEFAULT_MASS_ACCOMMODATION], dtype=np.float64
        ),
        np.array(
            [benchmark_module.DEFAULT_DIFFUSION_COEFFICIENT], dtype=np.float64
        ),
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
    empty_grid = benchmark_module._build_kernel_radius(
        np.zeros(3, dtype=np.float64)
    )
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


def test_project_mass_precision_candidate_reconstructs_expected_shapes() -> (
    None
):
    """Supported study candidates reconstruct arrays with stable shapes."""
    case = _build_mass_precision_cases()[2]

    for candidate_id in (
        "fp32_absolute_mass",
        "mixed_precision_mass_plus_density",
        "fp32_total_mass_fp32_mass_fraction",
    ):
        reconstructed = _project_candidate(case, candidate_id)[
            "reconstructed_masses"
        ]
        assert reconstructed.shape == case.masses.shape
        assert reconstructed.dtype == np.float64


def test_project_mass_precision_candidate_handles_zero_total_mass() -> None:
    """Zero-total-mass inputs reconstruct zeros without instability."""
    case = _build_mass_precision_cases()[2]
    zero_mass_case = type(case)(
        case_name=case.case_name,
        size_band=case.size_band,
        radius_unit=case.radius_unit,
        density_unit=case.density_unit,
        volume_fraction_unit=case.volume_fraction_unit,
        target_radius_m=case.target_radius_m,
        density_kg_m3=case.density_kg_m3.copy(),
        volume_fractions=case.volume_fractions.copy(),
        masses=np.zeros((2, 3, 2), dtype=np.float64),
        concentration=case.concentration.copy(),
        charge=case.charge.copy(),
        volume=case.volume.copy(),
    )

    reconstructed = _project_candidate(
        zero_mass_case,
        "fp32_total_mass_fp32_mass_fraction",
    )["reconstructed_masses"]

    npt.assert_array_equal(reconstructed, zero_mass_case.masses)


def test_project_mass_precision_candidate_rejects_unsupported_candidate() -> (
    None
):
    """Unsupported candidate ids fail with a stable error message."""
    case = _build_mass_precision_cases()[0]

    with pytest.raises(ValueError, match="Unsupported candidate id"):
        _project_candidate(case, "unsupported")


def test_mass_precision_projection_benchmark_records_bounded_result(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark entry point records bounded study metadata without CUDA."""
    monkeypatch.setattr(benchmark_module, "_save_results", lambda: None)

    label, case_index, candidate_id = (
        benchmark_module._MASS_PRECISION_BENCHMARK_CONFIGS[0]
    )
    benchmark_module.test_mass_precision_projection_benchmark(
        label,
        case_index,
        candidate_id,
    )

    result = benchmark_module._benchmark_results["benchmarks"][
        f"mass_precision_candidate_payload_{label}_{candidate_id}"
    ]
    assert result["case_name"] == label
    assert result["candidate_id"] == candidate_id
    assert result["repeats"] == 5_000
    assert result["elapsed_s"] >= 0.0
    assert result["mean_us"] >= 0.0


def test_preflight_condensation_case_allocations_checks_largest_buffers(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Condensation preflight covers the largest arrays before allocation."""
    seen: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "_preflight_benchmark_array",
        lambda shape, *, label, dtype=np.float64, itemsize=None: seen.append(
            label
        ),
    )

    benchmark_module._preflight_condensation_case_allocations("1x1k", 2, 3, 4)

    assert seen == [
        "cond-1x1k particle masses",
        "cond-1x1k particle concentration",
        "cond-1x1k gas concentration",
        "cond-1x1k vapor pressure",
        "cond-1x1k gpu mass transfer buffer",
    ]


def test_preflight_coagulation_case_allocations_checks_largest_buffers(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coagulation preflight covers particle and collision buffers."""
    seen: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "_preflight_benchmark_array",
        lambda shape, *, label, dtype=np.float64, itemsize=None: seen.append(
            label
        ),
    )

    benchmark_module._preflight_coagulation_case_allocations("1x500", 2, 3, 4)

    assert seen == [
        "coag-1x500 particle masses",
        "coag-1x500 particle concentration",
        "coag-1x500 collision pairs",
        "coag-1x500 collision counts",
        "coag-1x500 RNG state",
    ]


def test_wp_func_benchmark_input_builder_returns_expected_shapes(
    benchmark_module,
) -> None:
    """wp.func benchmark inputs are deterministic and shape-stable."""
    result = benchmark_module._build_wp_func_benchmark_inputs(8, seed=7)

    assert set(result) == {
        "temperatures",
        "mobilities",
        "pressure_deltas",
        "mass_transport",
        "molar_masses",
        "total_volumes",
        "radii_i",
        "radii_j",
        "diff_i",
        "diff_j",
        "g_i",
        "g_j",
        "speed_i",
        "speed_j",
    }
    for values in result.values():
        assert values.shape == (8,)
        assert values.dtype == np.float64


def test_preflight_large_allocation_skips_when_limit_is_exceeded(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized opt-in allocations skip before touching device memory."""
    monkeypatch.setenv("BENCHMARK_MAX_ALLOC_BYTES", "16")

    with pytest.raises(pytest.skip.Exception, match="requires"):
        benchmark_module._preflight_large_allocation(
            (3, 3),
            label="test array",
            itemsize=8,
        )


def test_wp_zeros_with_guard_skips_allocation_failures(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allocation helper turns device allocation failures into skips."""
    fake_wp = types.SimpleNamespace(
        zeros=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("oom"))
    )
    monkeypatch.setattr(benchmark_module, "wp", fake_wp)

    with pytest.raises(pytest.skip.Exception, match="allocation"):
        benchmark_module._wp_zeros_with_guard(
            (2, 2),
            dtype=np.float64,
            device="cuda",
            label="test allocation",
            itemsize=8,
        )


def test_seed_coagulation_rng_states_once_launches_and_synchronizes(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RNG seeding helper launches exactly once and synchronizes afterward."""
    calls: list[tuple[int, object, str]] = []
    fake_buffer = object()

    def fake_initialize_coagulation_rng_states(**kwargs: Any) -> object:
        calls.append(
            (kwargs["rng_seed"], kwargs["rng_states"], kwargs["device"])
        )
        return types.SimpleNamespace(shape=(3,))

    monkeypatch.setattr(
        benchmark_module,
        "initialize_coagulation_rng_states",
        fake_initialize_coagulation_rng_states,
    )

    benchmark_module._seed_coagulation_rng_states_once(
        rng_seed=42,
        rng_states=fake_buffer,
        n_boxes=3,
        device="cuda",
    )

    assert calls == [(42, fake_buffer, "cuda")]


def test_benchmark_module_imports_without_warp_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opted-in import tolerates Warp being unavailable."""

    def _guarded_import(name, *args, **kwargs):
        if name == "warp":
            raise ImportError("warp missing")
        return builtins_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _guarded_import)
    module = _load_benchmark_module(monkeypatch, benchmark_enabled=True)

    assert module.wp is None


def test_allocation_helpers_compute_sizes_and_delegate_preflight(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allocation helpers compute bytes and forward item sizes."""
    assert benchmark_module._allocation_itemsize(16, np.float64) == 16
    assert benchmark_module._allocation_itemsize(None, np.float32) == 4
    assert benchmark_module._estimate_allocation_nbytes((2, 3, 4), 8) == 192

    seen: list[tuple[tuple[int, ...], str, int]] = []
    monkeypatch.setattr(
        benchmark_module,
        "_preflight_large_allocation",
        lambda shape, *, label, itemsize: seen.append((shape, label, itemsize)),
    )

    benchmark_module._preflight_benchmark_array(
        (5, 6),
        label="buffer",
        dtype=np.float32,
    )

    assert seen == [((5, 6), "buffer", 4)]


def test_byte_sizing_helpers_cover_numpy_and_warp_paths(
    benchmark_module,
) -> None:
    """Byte-size helpers report stable dense-array allocation sizes."""
    assert (
        benchmark_module._warp_dtype_nbytes(benchmark_module.WARP_FLOAT64) == 8
    )
    assert benchmark_module._warp_dtype_nbytes(benchmark_module.WARP_INT32) == 4
    assert benchmark_module._array_nbytes((2, 3, 4), 8) == 192
    assert benchmark_module._numpy_nbytes((2, 3), np.float32) == 24
    assert (
        benchmark_module._warp_nbytes((2, 3), benchmark_module.WARP_UINT32)
        == 24
    )


def test_warp_dtype_nbytes_rejects_unsupported_dtype(benchmark_module) -> None:
    """Unsupported Warp dtypes fail with a clear sizing error."""
    with pytest.raises(ValueError, match="Unsupported Warp dtype"):
        benchmark_module._warp_dtype_nbytes(object())


def test_warp_profiled_yields_when_no_capture_or_profiler_available(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile wrapper still executes the body without Warp hooks."""
    monkeypatch.setenv("WARP_PROFILE", "1")
    monkeypatch.setattr(benchmark_module, "wp", types.SimpleNamespace())
    calls: list[str] = []

    with benchmark_module._warp_profiled("plain"):
        calls.append("body")

    assert calls == ["body"]


def test_cpu_mass_transfer_reuses_output_buffer_for_active_particles(
    benchmark_module,
) -> None:
    """CPU mass transfer fills a caller-provided output buffer in place."""
    particles = benchmark_module._make_particle_data(1, 1, 1)
    gas = benchmark_module._make_gas_data(1, 1)
    vapor_pressure = benchmark_module._make_vapor_pressure(1, 1)
    out = np.full_like(particles.masses, -1.0)

    result = benchmark_module._cpu_mass_transfer(
        particles,
        gas,
        vapor_pressure,
        np.array([benchmark_module.DEFAULT_SURFACE_TENSION], dtype=np.float64),
        np.array(
            [benchmark_module.DEFAULT_MASS_ACCOMMODATION],
            dtype=np.float64,
        ),
        np.array(
            [benchmark_module.DEFAULT_DIFFUSION_COEFFICIENT],
            dtype=np.float64,
        ),
        benchmark_module.DEFAULT_TEMPERATURE,
        benchmark_module.DEFAULT_PRESSURE,
        benchmark_module.DEFAULT_TIME_STEP,
        out=out,
    )

    assert result is out
    assert result.shape == particles.masses.shape
    assert np.all(np.isfinite(result))
    assert not np.all(result == 0.0)


def test_cpu_coagulation_step_handles_empty_collision_pairs(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU coagulation leaves state unchanged when no collisions occur."""
    particles = benchmark_module._make_particle_data(1, 3, 2)
    original_masses = particles.masses.copy()
    original_concentration = particles.concentration.copy()
    monkeypatch.setattr(
        benchmark_module,
        "get_brownian_kernel_via_system_state",
        lambda **kwargs: np.ones((4, 4), dtype=np.float64),
    )
    monkeypatch.setattr(
        benchmark_module,
        "get_particle_resolved_coagulation_step",
        lambda *args, **kwargs: np.empty((0, 2), dtype=np.int64),
    )

    benchmark_module._cpu_coagulation_step(
        particles,
        benchmark_module.DEFAULT_TEMPERATURE,
        benchmark_module.DEFAULT_PRESSURE,
        benchmark_module.DEFAULT_TIME_STEP,
        np.random.default_rng(42),
        np.linspace(1.0e-9, 1.0e-8, 4),
    )

    npt.assert_allclose(particles.masses, original_masses)
    npt.assert_allclose(particles.concentration, original_concentration)


def test_benchmark_cpu_wp_funcs_returns_timing_keys(benchmark_module) -> None:
    """CPU wp.func benchmark helper reports each timing bucket."""
    result = benchmark_module._benchmark_cpu_wp_funcs(
        benchmark_module._build_wp_func_benchmark_inputs(16, seed=5),
        kernel_sample=8,
    )

    assert set(result) == {
        "diffusion_coefficient",
        "mass_transfer_rate",
        "particle_radius_from_volume",
        "brownian_diffusivity",
        "brownian_kernel_pair",
    }
    assert all(value >= 0.0 for value in result.values())


def test_benchmark_gpu_wp_funcs_uses_fake_warp_backend(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU wp.func benchmark helper can run against a fake Warp facade."""
    launch_calls: list[tuple[object, int, str]] = []

    class _FakeWarp:
        float64 = np.float64

        @staticmethod
        def array(values, dtype=None, device=None):
            return np.asarray(values, dtype=np.float64)

        @staticmethod
        def zeros(shape, dtype=None, device=None):
            return np.zeros(shape, dtype=np.float64)

        @staticmethod
        def launch(kernel, dim, inputs=None, outputs=None, device=None):
            launch_calls.append((kernel, dim, device))

        @staticmethod
        def synchronize():
            return None

    monkeypatch.setattr(benchmark_module, "wp", _FakeWarp())
    monkeypatch.setattr(
        benchmark_module,
        "_diffusion_coefficient_kernel",
        object(),
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_mass_transfer_rate_kernel",
        object(),
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_particle_radius_kernel",
        object(),
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_brownian_diffusivity_kernel",
        object(),
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_brownian_kernel_pair_kernel",
        object(),
        raising=False,
    )

    result = benchmark_module._benchmark_gpu_wp_funcs(
        benchmark_module._build_wp_func_benchmark_inputs(4, seed=11)
    )

    assert len(launch_calls) == 10
    assert set(result) == {
        "diffusion_coefficient",
        "mass_transfer_rate",
        "particle_radius_from_volume",
        "brownian_diffusivity",
        "brownian_kernel_pair",
    }
    assert all(value >= 0.0 for value in result.values())


def test_condensation_scaling_records_cpu_and_gpu_paths(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Condensation benchmark records both comparison and GPU-only metadata."""
    monkeypatch.setattr(benchmark_module, "_skip_if_no_cuda", lambda: None)
    monkeypatch.setattr(
        benchmark_module,
        "_preflight_condensation_case_allocations",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "to_warp_particle_data",
        lambda particles, device: particles,
    )
    monkeypatch.setattr(
        benchmark_module,
        "to_warp_gas_data",
        lambda gas, device, vapor_pressure: gas,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_wp_zeros_with_guard",
        lambda *args, **kwargs: np.zeros((1, 1, 1), dtype=np.float64),
    )
    monkeypatch.setattr(
        benchmark_module,
        "wp",
        types.SimpleNamespace(
            float64=np.float64,
            array=lambda values, **kwargs: np.asarray(values, dtype=np.float64),
        ),
    )
    gpu_calls: list[str] = []
    cpu_calls: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "condensation_step_gpu",
        lambda *args, **kwargs: gpu_calls.append("gpu"),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_cpu_condensation_step",
        lambda *args, **kwargs: cpu_calls.append("cpu"),
    )

    @contextmanager
    def _profile(_tag: str):
        yield

    monkeypatch.setattr(benchmark_module, "_warp_profiled", _profile)

    def _run_gpu(step_fn, steps, warmup):
        step_fn()
        return 0.5

    def _run_cpu(step_fn, steps, warmup):
        step_fn()
        return 1.5

    monkeypatch.setattr(benchmark_module, "_time_gpu_loop", _run_gpu)
    monkeypatch.setattr(benchmark_module, "_time_cpu_loop", _run_cpu)
    monkeypatch.setattr(benchmark_module, "_print_timing", lambda *args: None)
    monkeypatch.setattr(
        benchmark_module,
        "_compute_speedup",
        lambda cpu_time, gpu_time: cpu_time / gpu_time,
    )
    monkeypatch.setattr(benchmark_module, "_save_results", lambda: None)
    monkeypatch.setattr(
        benchmark_module,
        "_benchmark_results",
        {"started_at": "2026-01-01T00:00:00+00:00", "benchmarks": {}},
    )

    benchmark_module.test_condensation_scaling("case-a", 1, 1, 1, True)
    benchmark_module.test_condensation_scaling("case-b", 1, 1, 1, False)

    assert gpu_calls == ["gpu", "gpu"]
    assert cpu_calls == ["cpu"]
    cpu_entry = benchmark_module._benchmark_results["benchmarks"][
        "condensation_case-a"
    ]
    gpu_only_entry = benchmark_module._benchmark_results["benchmarks"][
        "condensation_case-b"
    ]
    assert cpu_entry["gpu_time_s"] == 0.5
    assert cpu_entry["cpu_time_s"] == 1.5
    assert cpu_entry["speedup"] == 3.0
    assert gpu_only_entry["gpu_time_s"] == 0.5
    assert "cpu_time_s" not in gpu_only_entry


def test_coagulation_scaling_records_cpu_and_gpu_paths(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coagulation benchmark records both comparison and GPU-only metadata."""
    monkeypatch.setattr(benchmark_module, "_skip_if_no_cuda", lambda: None)
    monkeypatch.setattr(
        benchmark_module,
        "_preflight_coagulation_case_allocations",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "to_warp_particle_data",
        lambda particles, device: particles,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_wp_zeros_with_guard",
        lambda shape, **kwargs: np.zeros(shape, dtype=np.int32),
    )
    monkeypatch.setattr(
        benchmark_module,
        "wp",
        types.SimpleNamespace(
            int32=np.int32,
            uint32=np.uint32,
        ),
    )
    gpu_seeds: list[int] = []
    seed_calls: list[tuple[int, object, int, str]] = []
    cpu_calls: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "_seed_coagulation_rng_states_once",
        lambda **kwargs: seed_calls.append(
            (
                kwargs["rng_seed"],
                kwargs["rng_states"],
                kwargs["n_boxes"],
                kwargs["device"],
            )
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "coagulation_step_gpu",
        lambda *args, **kwargs: gpu_seeds.append(kwargs["rng_seed"]),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_cpu_coagulation_step",
        lambda *args, **kwargs: cpu_calls.append("cpu"),
    )

    @contextmanager
    def _profile(_tag: str):
        yield

    monkeypatch.setattr(benchmark_module, "_warp_profiled", _profile)

    def _run_gpu(step_fn, steps, warmup):
        step_fn()
        return 0.25

    def _run_cpu(step_fn, steps, warmup):
        step_fn()
        return 1.0

    monkeypatch.setattr(benchmark_module, "_time_gpu_loop", _run_gpu)
    monkeypatch.setattr(benchmark_module, "_time_cpu_loop", _run_cpu)
    monkeypatch.setattr(benchmark_module, "_print_timing", lambda *args: None)
    monkeypatch.setattr(
        benchmark_module,
        "_compute_speedup",
        lambda cpu_time, gpu_time: cpu_time / gpu_time,
    )
    monkeypatch.setattr(benchmark_module, "_save_results", lambda: None)
    monkeypatch.setattr(
        benchmark_module,
        "_benchmark_results",
        {"started_at": "2026-01-01T00:00:00+00:00", "benchmarks": {}},
    )

    benchmark_module.test_coagulation_scaling("case-a", 1, 2, 1, True)
    benchmark_module.test_coagulation_scaling("case-b", 1, 2, 1, False)

    assert gpu_seeds == [42, 42]
    assert len(seed_calls) == 2
    assert [call[0] for call in seed_calls] == [42, 42]
    assert [call[2] for call in seed_calls] == [1, 1]
    assert [call[3] for call in seed_calls] == ["cuda", "cuda"]
    assert cpu_calls == ["cpu"]
    cpu_entry = benchmark_module._benchmark_results["benchmarks"][
        "coagulation_case-a"
    ]
    gpu_only_entry = benchmark_module._benchmark_results["benchmarks"][
        "coagulation_case-b"
    ]
    assert cpu_entry["gpu_time_s"] == 0.25
    assert cpu_entry["cpu_time_s"] == 1.0
    assert cpu_entry["speedup"] == 4.0
    assert gpu_only_entry["gpu_time_s"] == 0.25
    assert "cpu_time_s" not in gpu_only_entry


@pytest.mark.parametrize(
    "n_boxes,n_particles,n_species",
    [(1, 4, 1), (10, 500, 2)],
)
def test_make_coagulation_particle_data_builds_deterministic_mixed_scale_fixture(
    benchmark_module,
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> None:
    """Coagulation fixture helper builds deterministic mixed-scale data."""
    first = benchmark_module._make_coagulation_particle_data(
        n_boxes,
        n_particles,
        n_species,
    )
    second = benchmark_module._make_coagulation_particle_data(
        n_boxes,
        n_particles,
        n_species,
    )

    assert first.masses.shape == (n_boxes, n_particles, n_species)
    assert first.concentration.shape == (n_boxes, n_particles)
    assert first.charge.shape == (n_boxes, n_particles)
    assert first.density.shape == (n_species,)
    assert first.volume.shape == (n_boxes,)
    assert np.all(first.concentration > 0.0)

    radii = first.radii
    assert radii.shape == (n_boxes, n_particles)
    assert np.min(radii) < 1.0e-8
    assert np.max(radii) > 1.0e-6
    assert np.max(radii) / np.min(radii) > 1.0e3

    npt.assert_allclose(first.masses, second.masses)
    npt.assert_allclose(first.concentration, second.concentration)
    npt.assert_allclose(first.radii, second.radii)


def test_coagulation_scaling_uses_coagulation_only_helper(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coagulation uses its dedicated helper while condensation keeps generic setup."""
    monkeypatch.setattr(benchmark_module, "_skip_if_no_cuda", lambda: None)
    monkeypatch.setattr(
        benchmark_module,
        "_preflight_condensation_case_allocations",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_preflight_coagulation_case_allocations",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_validate_benchmark_budget",
        lambda *args, **kwargs: None,
    )

    generic_calls: list[tuple[int, int, int]] = []
    coag_calls: list[tuple[int, int, int]] = []

    generic_particles = benchmark_module._make_particle_data(1, 2, 1)
    coag_particles = benchmark_module._make_coagulation_particle_data(1, 2, 1)

    def _record_generic_call(
        n_boxes: int, n_particles: int, n_species: int
    ) -> object:
        generic_calls.append((n_boxes, n_particles, n_species))
        return generic_particles

    def _record_coag_call(
        n_boxes: int, n_particles: int, n_species: int
    ) -> object:
        coag_calls.append((n_boxes, n_particles, n_species))
        return coag_particles

    monkeypatch.setattr(
        benchmark_module,
        "_make_particle_data",
        _record_generic_call,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_make_coagulation_particle_data",
        _record_coag_call,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_make_gas_data",
        lambda n_boxes, n_species: object(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_make_vapor_pressure",
        lambda n_boxes, n_species: np.ones(
            (n_boxes, n_species), dtype=np.float64
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "to_warp_particle_data",
        lambda particles, device: particles,
    )
    monkeypatch.setattr(
        benchmark_module,
        "to_warp_gas_data",
        lambda gas, device, vapor_pressure: gas,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_wp_zeros_with_guard",
        lambda shape, **kwargs: np.zeros(shape, dtype=np.float64),
    )
    monkeypatch.setattr(
        benchmark_module,
        "wp",
        types.SimpleNamespace(
            float64=np.float64,
            int32=np.int32,
            uint32=np.uint32,
            array=lambda values, **kwargs: values,
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "condensation_step_gpu",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "coagulation_step_gpu",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_seed_coagulation_rng_states_once",
        lambda **kwargs: None,
    )

    @contextmanager
    def _profile(_tag: str):
        yield

    monkeypatch.setattr(benchmark_module, "_warp_profiled", _profile)
    monkeypatch.setattr(
        benchmark_module,
        "_time_gpu_loop",
        lambda step_fn, steps, warmup: (step_fn() or 0.25),
    )
    monkeypatch.setattr(benchmark_module, "_save_results", lambda: None)
    monkeypatch.setattr(benchmark_module, "_print_timing", lambda *args: None)
    monkeypatch.setattr(
        benchmark_module,
        "_benchmark_results",
        {"started_at": "2026-01-01T00:00:00+00:00", "benchmarks": {}},
    )

    benchmark_module.test_condensation_scaling("cond", 1, 2, 1, False)

    assert generic_calls == [(1, 2, 1)]
    assert coag_calls == []

    benchmark_module.test_coagulation_scaling("coag", 1, 2, 1, False)

    assert generic_calls == [(1, 2, 1)]
    assert coag_calls == [(1, 2, 1)]


def test_coagulation_scaling_reuses_persistent_rng_states_without_seed_drift(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coagulation benchmark keeps a constant seed across repeated GPU steps."""
    monkeypatch.setattr(benchmark_module, "_skip_if_no_cuda", lambda: None)
    monkeypatch.setattr(
        benchmark_module,
        "_preflight_coagulation_case_allocations",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "to_warp_particle_data",
        lambda particles, device: particles,
    )

    rng_state_buffer = object()

    def _zeros_with_guard(shape, **kwargs):
        if kwargs.get("dtype") is benchmark_module.wp.uint32:
            return rng_state_buffer
        return np.zeros(shape, dtype=np.int32)

    monkeypatch.setattr(
        benchmark_module,
        "_wp_zeros_with_guard",
        _zeros_with_guard,
    )
    monkeypatch.setattr(
        benchmark_module,
        "wp",
        types.SimpleNamespace(
            int32=np.int32,
            uint32=np.uint32,
        ),
    )
    gpu_kwargs: list[dict[str, object]] = []
    seed_calls: list[tuple[int, object, int, str]] = []
    monkeypatch.setattr(
        benchmark_module,
        "_seed_coagulation_rng_states_once",
        lambda **kwargs: seed_calls.append(
            (
                kwargs["rng_seed"],
                kwargs["rng_states"],
                kwargs["n_boxes"],
                kwargs["device"],
            )
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "coagulation_step_gpu",
        lambda *args, **kwargs: gpu_kwargs.append(kwargs),
    )

    @contextmanager
    def _profile(_tag: str):
        yield

    monkeypatch.setattr(benchmark_module, "_warp_profiled", _profile)

    def _run_gpu(step_fn, steps, warmup):
        for _ in range(3):
            step_fn()
        return 0.25

    monkeypatch.setattr(benchmark_module, "_time_gpu_loop", _run_gpu)
    monkeypatch.setattr(benchmark_module, "_save_results", lambda: None)
    monkeypatch.setattr(
        benchmark_module,
        "_benchmark_results",
        {"started_at": "2026-01-01T00:00:00+00:00", "benchmarks": {}},
    )

    benchmark_module.test_coagulation_scaling("case-a", 1, 2, 1, False)

    assert len(gpu_kwargs) == 3
    assert [kwargs["rng_seed"] for kwargs in gpu_kwargs] == [42, 42, 42]
    assert seed_calls == [(42, rng_state_buffer, 1, "cuda")]
    assert all(
        kwargs["rng_states"] is rng_state_buffer for kwargs in gpu_kwargs
    )
    assert all(kwargs.get("initialize_rng") is False for kwargs in gpu_kwargs)


def test_wp_func_benchmarks_records_summary_without_cuda(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wp.func benchmark summary records timing metadata via fast stubs."""
    monkeypatch.setattr(benchmark_module, "_skip_if_no_cuda", lambda: None)
    monkeypatch.setattr(
        benchmark_module,
        "_build_wp_func_benchmark_inputs",
        lambda n_evals: {"dummy": np.arange(n_evals, dtype=np.float64)},
    )
    monkeypatch.setattr(
        benchmark_module,
        "_benchmark_cpu_wp_funcs",
        lambda inputs, *, kernel_sample: {
            "diffusion_coefficient": 1.0,
            "mass_transfer_rate": 2.0,
            "particle_radius_from_volume": 3.0,
            "brownian_diffusivity": 4.0,
            "brownian_kernel_pair": 5.0,
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "_benchmark_gpu_wp_funcs",
        lambda inputs: {
            "diffusion_coefficient": 0.5,
            "mass_transfer_rate": 1.0,
            "particle_radius_from_volume": 1.5,
            "brownian_diffusivity": 2.0,
            "brownian_kernel_pair": 2.5,
        },
    )
    monkeypatch.setattr(benchmark_module, "_save_results", lambda: None)
    monkeypatch.setattr(
        benchmark_module,
        "_benchmark_results",
        {"started_at": "2026-01-01T00:00:00+00:00", "benchmarks": {}},
    )

    benchmark_module.test_wp_func_benchmarks()

    result = benchmark_module._benchmark_results["benchmarks"]["wp_func"]
    assert result["n_evals"] == 100_000
    assert result["diffusion_coefficient"]["cpu_us"] > 0.0
    assert result["brownian_kernel_pair"]["gpu_us"] > 0.0
