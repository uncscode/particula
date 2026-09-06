"""Hardware-free tests for strict profiling evidence support."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from particula.gpu.tests import profiling_support as support


def _workloads() -> tuple[support.ProfilingWorkload, ...]:
    """Return the fixed local profiling matrix."""
    return support.build_default_profiling_workload_matrix()


def _machine() -> support.MachineProvenance:
    """Build valid synthetic machine provenance."""
    return support.MachineProvenance(
        "ci-machine", "linux", "3.12", "12.0", "550", "gpu-0", "abc123"
    )


def _method() -> support.MeasurementMethod:
    """Build one valid synthetic measurement method."""
    return support.MeasurementMethod(
        "profiler", "profiler", "tool --run", "1", "ns"
    )


def _executed() -> support.ExecutedEvidence:
    """Build a complete synthetic executed row."""
    return support.ExecutedEvidence(
        "executed",
        _workloads()[0],
        _machine(),
        _method(),
        (support.RawDurationSample(1, 10),),
        (support.NormalizedMetric("profiler_gpu_duration", 10.0, "ns"),),
        (support.RawReportProvenance("trace.json", 5, "a" * 64),),
    )


def test_default_matrix_preserves_frozen_e8_f6_settings() -> None:
    """Test the exact two-row matrix and all shared settings."""
    workloads = _workloads()
    assert [item.label for item in workloads] == ["small", "medium"]
    assert [item.shape for item in workloads] == [(1, 16, 2), (1000, 16, 2)]
    assert [item.workload_id for item in workloads] == [
        item.workload_id for item in _workloads()
    ]
    for item in workloads:
        assert item.active_fraction == 1.0
        assert item.communication == "gas"
        assert item.diagnostics == ("gas", "saturation")
        assert item.warmup == 2
        assert item.sample_count == 3
        assert item.seed == 1582
        assert item.duration_seconds == 0.5
        assert item.replay_counts == (1, 10, 100, 1000)


def test_artifact_round_trips_to_byte_identical_canonical_json() -> None:
    """Test the executed/unavailable union round-trips canonically."""
    unavailable = support.UnavailableEvidence(
        "unavailable", _workloads()[1], "native profiler unavailable"
    )
    artifact = support.ProfilingArtifact((_executed(), unavailable))
    serialized = support.serialize_profiling_artifact(artifact)
    assert (
        support.serialize_profiling_artifact(
            support.deserialize_profiling_artifact(serialized)
        )
        == serialized
    )


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        '{"schema_version":2,"artifact":{"evidence":[]}}',
        '{"schema_version":1,"artifact":{"rows":[]}}',
        "[" * (support.MAX_ARTIFACT_NESTING_DEPTH + 1),
    ],
)
def test_deserialization_rejects_invalid_envelopes(payload: str) -> None:
    """Test malformed, unsupported, and bounded JSON rejection."""
    with pytest.raises(ValueError):
        support.deserialize_profiling_artifact(payload)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: support.RawDurationSample(True, 1),
        lambda: support.RawDurationSample(1, 0),
        lambda: support.NormalizedMetric("profiler_gpu_duration", -1.0, "ns"),
        lambda: support.NormalizedMetric("percentage", 1.0, "%"),
        lambda: support.RawReportProvenance("report", True, "a" * 64),
        lambda: support.RawReportProvenance("report", 1, "A" * 64),
        lambda: support.MeasurementMethod(
            "profiler", "profiler", "run", "1", "ms"
        ),
    ],
)
def test_records_reject_invalid_scalar_and_closed_vocabularies(
    factory: object,
) -> None:
    """Test Boolean numerics and invalid closed-vocabulary record values."""
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_workloads_and_executed_rows_fail_closed() -> None:
    """Test invalid IDs, shapes, duplicate metrics, and missing samples."""
    workload = _workloads()[0]
    with pytest.raises(ValueError):
        support.ProfilingWorkload(
            "wrong",
            "small",
            (1, 16, 2),
            1.0,
            workload.processes,
            "gas",
            workload.diagnostics,
            2,
            3,
            1582,
            0.5,
            support.REPLAY_COUNTS,
        )
    with pytest.raises(ValueError):
        support.ProfilingWorkload(
            workload.workload_id,
            "small",
            (True, 16, 2),
            1.0,
            workload.processes,
            "gas",
            workload.diagnostics,
            2,
            3,
            1582,
            0.5,
            support.REPLAY_COUNTS,
        )
    metric = support.NormalizedMetric("profiler_gpu_duration", 1.0, "ns")
    with pytest.raises(ValueError):
        support.ExecutedEvidence(
            "executed",
            workload,
            _machine(),
            _method(),
            (),
            (metric,),
            (support.RawReportProvenance("trace", 1, "a" * 64),),
        )
    with pytest.raises(ValueError):
        support.ExecutedEvidence(
            "executed",
            workload,
            _machine(),
            _method(),
            (support.RawDurationSample(1, 1),),
            (metric, metric),
            (support.RawReportProvenance("trace", 1, "a" * 64),),
        )


def test_raw_provenance_is_contained_streamed_and_reverified(
    tmp_path: Path,
) -> None:
    """Test explicit root creation, digest validation, and changed report failure."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    raw_root = support.ensure_profiling_raw_root(root)
    report = raw_root / "trace.json"
    report.write_bytes(b"hello")
    provenance = support.build_raw_report_provenance(root, "trace.json")
    assert provenance.byte_size == 5
    assert provenance.sha256 == hashlib.sha256(b"hello").hexdigest()
    support.verify_raw_report_provenance(root, provenance)
    report.write_bytes(b"changed")
    with pytest.raises(ValueError):
        support.verify_raw_report_provenance(root, provenance)


@pytest.mark.parametrize(
    "filename", ["", ".", "..", "../trace", Path.cwd().anchor + "report", "a/b"]
)
def test_raw_provenance_rejects_unsafe_names(
    tmp_path: Path, filename: str
) -> None:
    """Test report provenance accepts only a single contained filename."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    with pytest.raises((TypeError, ValueError)):
        support.build_raw_report_provenance(root, filename)


def test_raw_provenance_rejects_oversized_and_external_symlink(
    tmp_path: Path,
) -> None:
    """Test raw-report bounds and symlink containment without hardware access."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    raw_root = support.ensure_profiling_raw_root(root)
    large = raw_root / "large"
    large.write_bytes(b"x" * (support.MAX_RAW_REPORT_BYTES + 1))
    with pytest.raises(ValueError):
        support.build_raw_report_provenance(root, "large")
    external = tmp_path / "external"
    external.write_bytes(b"x")
    try:
        (raw_root / "outside").symlink_to(external)
    except OSError:
        pytest.skip("platform does not permit symlink creation")
    with pytest.raises(ValueError):
        support.build_raw_report_provenance(root, "outside")


def test_unavailable_evidence_retains_requested_workload_only() -> None:
    """Test unavailable rows retain no fabricated measurement context."""
    unavailable = support.UnavailableEvidence(
        "unavailable", _workloads()[1], "native profiler unavailable"
    )
    assert unavailable.workload.shape == (1000, 16, 2)
    assert set(unavailable.__dataclass_fields__) == {
        "status",
        "workload",
        "reason",
    }
    with pytest.raises(ValueError):
        support.ProfilingArtifact((unavailable, _executed()))
