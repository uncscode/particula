"""Unit tests for GPU benchmark safety helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from particula.execution.tests.resident_benchmark_cuda_support import (
    ResidentCaptureBenchmarkBinding,
)
from particula.execution.tests.resident_benchmark_support import (
    ResidentBenchmarkAvailability,
)
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


def test_resident_provenance_uses_fixture_signature_and_device() -> None:
    """Persist the selected CUDA device rather than a synthetic ordinal."""
    device = {"status": "available", "identity": "cuda:7", "memory": 42}
    binding = SimpleNamespace(
        prepared_signature_digest="real-prepared-signature",
        selected_device=device,
    )

    signature, selected_device = benchmark_test.resident_benchmark_provenance(
        cast(ResidentCaptureBenchmarkBinding, binding)
    )

    assert signature == "real-prepared-signature"
    assert selected_device is device


def test_resident_matrix_records_unavailability_with_one_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight all exact rows before CUDA construction and memoize absence."""
    probes: list[str] = []

    def unavailable_after_probe() -> ResidentBenchmarkAvailability:
        probes.append("probe")
        return ResidentBenchmarkAvailability(False, "no CUDA")

    monkeypatch.setattr(
        benchmark_test,
        "cuda_capture_availability",
        unavailable_after_probe,
    )
    monkeypatch.setattr(
        benchmark_test,
        "qualified_cuda_resident_benchmark",
        lambda **_: pytest.fail("CUDA fixture must not be constructed"),
    )

    artifact = benchmark_test._collect_resident_capture_matrix()

    assert probes == ["probe"]
    assert len(artifact.cases) == len(artifact.results) == 4
    assert {result.status for result in artifact.results} == {
        benchmark_test.ResidentBenchmarkStatus.UNAVAILABLE
    }
    assert all(result.reason == "no CUDA" for result in artifact.results)


def test_resident_matrix_forwards_exact_dimensions_and_reuses_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct one exact CUDA binding for each approved matrix request."""
    calls: list[dict[str, object]] = []

    @contextmanager
    def qualified_cuda_resident_benchmark(**kwargs: object):
        calls.append(kwargs)
        dimensions = SimpleNamespace(
            n_boxes=kwargs["n_boxes"],
            n_particles=kwargs["n_particles"],
            n_species=kwargs["n_species"],
        )
        binding = SimpleNamespace(
            loop=SimpleNamespace(
                prepared=SimpleNamespace(
                    signature=SimpleNamespace(dimensions=dimensions)
                )
            ),
            validate_identities=lambda: None,
            enqueue=lambda: None,
            replay=lambda: None,
            synchronize=lambda: None,
            setup_elapsed_seconds=0.0,
            capture_elapsed_seconds=0.0,
            prepared_signature_digest="exact",
            selected_device={
                "status": "available",
                "identity": "cuda:0",
                "memory": 1,
            },
        )
        yield binding

    monkeypatch.setattr(
        benchmark_test,
        "cuda_capture_availability",
        lambda: ResidentBenchmarkAvailability(True),
    )
    monkeypatch.setattr(
        benchmark_test,
        "qualified_cuda_resident_benchmark",
        qualified_cuda_resident_benchmark,
    )
    monkeypatch.setattr(
        benchmark_test,
        "collect_paired_device_timings",
        lambda **_: ((1.0,), (2.0,)),
    )

    artifact = benchmark_test._collect_resident_capture_matrix()

    assert [
        (call["n_boxes"], call["n_particles"], call["n_species"])
        for call in calls
    ] == [(1, 16, 2), (10, 16, 2), (100, 16, 2), (1000, 16, 2)]
    assert all(cast(Any, call["availability"]).available for call in calls)
    assert len(artifact.results) == 8
