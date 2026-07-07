"""Unit tests for GPU benchmark safety helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from particula.gpu.tests import benchmark_test


def test_sanitize_benchmark_output_name_keeps_filename_only() -> None:
    """Escaping path fragments collapse to a safe filename."""
    assert (
        benchmark_test._sanitize_benchmark_output_name("../tmp/results.json")
        == "results.json"
    )


def test_sanitize_benchmark_output_name_rejects_empty_filename() -> None:
    """Blank output overrides fail closed."""
    with pytest.raises(ValueError, match="non-empty filename"):
        benchmark_test._sanitize_benchmark_output_name("   ")


def test_get_benchmark_output_path_stays_under_artifact_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved benchmark output stays inside the artifact directory."""
    monkeypatch.setenv("BENCHMARK_OUTPUT", "../../escape.json")
    resolved = benchmark_test._get_benchmark_output_path()
    assert resolved == Path(".artifacts/benchmarks/escape.json")


def test_parse_positive_int_env_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget overrides must be strictly positive integers."""
    monkeypatch.setenv("BENCHMARK_MAX_BYTES", "0")
    with pytest.raises(ValueError, match="must be positive"):
        benchmark_test._parse_positive_int_env("BENCHMARK_MAX_BYTES", 1)


def test_warp_nbytes_uses_warp_dtype_sizes() -> None:
    """Warp byte estimation uses the benchmark dtype map."""
    assert benchmark_test._warp_nbytes((2, 3), benchmark_test.wp.float64) == 48
    assert benchmark_test._warp_nbytes((2, 3), benchmark_test.wp.int32) == 24


def test_condensation_budget_grows_with_cpu_copy_buffers() -> None:
    """CPU-enabled condensation cases budget extra host allocations."""
    gpu_only = benchmark_test._estimate_condensation_budget(
        "case", 1, 32, 3, False
    )
    with_cpu = benchmark_test._estimate_condensation_budget(
        "case", 1, 32, 3, True
    )
    assert with_cpu.cpu_bytes > gpu_only.cpu_bytes
    assert with_cpu.gpu_bytes == gpu_only.gpu_bytes


def test_validate_benchmark_budget_skips_oversized_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized benchmark cases skip before allocation."""
    monkeypatch.setenv("BENCHMARK_MAX_BYTES", "128")
    budget = benchmark_test.BenchmarkMemoryBudget(
        label="oversized",
        cpu_bytes=64,
        gpu_bytes=96,
    )
    with pytest.raises(pytest.skip.Exception, match="exceeding"):
        benchmark_test._validate_benchmark_budget(budget)
