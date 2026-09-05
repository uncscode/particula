"""Host-only tests for resident benchmark evidence support."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from particula.execution.tests import resident_benchmark_support
from particula.execution.tests.resident_benchmark_support import (
    MAX_ARTIFACT_CONTAINER_ITEMS,
    MAX_ARTIFACT_NESTING_DEPTH,
    MAX_ARTIFACT_PAYLOAD_BYTES,
    MAX_ARTIFACT_ROWS,
    MAX_TIMING_SAMPLES,
    MAX_WARMUP_SAMPLES,
    RESIDENT_CAPTURE_COMPARISON_DESTINATION,
    ResidentBenchmarkArtifact,
    ResidentBenchmarkCase,
    ResidentBenchmarkResult,
    ResidentBenchmarkStatus,
    build_resident_benchmark_case_id,
    build_resident_benchmark_metadata,
    collect_paired_device_timings,
    deserialize_resident_benchmark_artifact,
    serialize_resident_benchmark_artifact,
    summarize_timing_samples,
    write_json_artifact,
    write_resident_capture_comparison_artifact,
)


def _case() -> ResidentBenchmarkCase:
    """Return a valid canonical host-only benchmark case."""
    return ResidentBenchmarkCase(
        case_id=build_resident_benchmark_case_id(
            requested_shape=(2, 8, 3),
            actual_shape=(1, 4, 2),
            active_fraction=0.5,
            processes=("condensation", "dilution"),
            communication="none",
            diagnostics=("gas",),
            warmup=2,
            timestep_count=5,
            seed=7,
        ),
        requested_shape=(2, 8, 3),
        actual_shape=(1, 4, 2),
        active_fraction=0.5,
        processes=("condensation", "dilution"),
        communication="none",
        diagnostics=("gas",),
        warmup=2,
        timestep_count=5,
        seed=7,
    )


def _metadata():
    """Return complete injected metadata without a device probe."""
    return build_resident_benchmark_metadata(
        timestamp_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        command="pytest resident",
        synchronization_method="explicit",
        warmup=2,
        timestep_count=5,
        seed=7,
        prepared_signature_digest="abc",
        warp_version={"status": "unavailable", "value": None},
        device={"status": "unavailable", "identity": None, "memory": None},
    )


def _artifact() -> ResidentBenchmarkArtifact:
    """Return an artifact with executed and non-executed evidence."""
    case = _case()
    samples = (3.0, 1.0, 2.0)
    return ResidentBenchmarkArtifact(
        metadata=_metadata(),
        cases=(case,),
        results=(
            ResidentBenchmarkResult(
                case_id=case.case_id,
                timing_mode="wall_clock",
                requested_shape=case.requested_shape,
                status=ResidentBenchmarkStatus.EXECUTED,
                reason=None,
                samples=samples,
                summary=summarize_timing_samples(samples),
                provenance={"run": "one"},
            ),
            ResidentBenchmarkResult(
                case_id=case.case_id,
                timing_mode=None,
                requested_shape=case.requested_shape,
                status=ResidentBenchmarkStatus.UNAVAILABLE,
                reason="no device",
                samples=(),
                summary=None,
                provenance={"run": "two"},
            ),
        ),
    )


def test_records_summaries_metadata_and_round_trip_are_deterministic():
    """Validate records and byte-stable serialization with provenance."""
    artifact = _artifact()
    summary = summarize_timing_samples((3.0, 1.0, 2.0, 4.0))
    assert (
        summary.count,
        summary.minimum,
        summary.median,
        summary.mean,
        summary.p95,
    ) == (
        4,
        1.0,
        2.5,
        2.5,
        4.0,
    )
    assert _metadata()["timestamp_utc"] == "2026-01-02T00:00:00Z"
    with pytest.raises(FrozenInstanceError):
        artifact.cases[0].seed = 8
    first = serialize_resident_benchmark_artifact(artifact)
    assert first.endswith("\n")
    assert json.loads(first)["schema_version"] == 2
    assert deserialize_resident_benchmark_artifact(first) == artifact
    assert (
        serialize_resident_benchmark_artifact(
            deserialize_resident_benchmark_artifact(first)
        )
        == first
    )
    isolated_import = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import sys; import particula.execution.tests.resident_benchmark_support; "
            "assert 'warp' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert isolated_import.returncode == 0, isolated_import.stderr


def test_metadata_contains_complete_injected_host_provenance():
    """Keep unavailable device provenance complete without probing a device."""
    metadata = _metadata()

    assert set(metadata) == {
        "timestamp_utc",
        "command",
        "python_version",
        "platform",
        "warp_version",
        "device",
        "synchronization_method",
        "warmup",
        "timestep_count",
        "seed",
        "prepared_signature_digest",
    }
    assert metadata["warp_version"] == {"status": "unavailable", "value": None}
    assert metadata["device"] == {
        "status": "unavailable",
        "identity": None,
        "memory": None,
    }
    with pytest.raises(TypeError):
        metadata["command"] = "changed"
    with pytest.raises(TypeError):
        metadata["device"]["status"] = "changed"


@pytest.mark.parametrize(
    "warp_version, device",
    [
        (
            {"status": "available", "value": "1.9"},
            {"status": "available", "identity": "cuda:0", "memory": 8},
        ),
        (
            {"status": "unavailable", "value": None},
            {"status": "unavailable", "identity": None, "memory": None},
        ),
        (
            {"status": "error", "value": None, "error": "import failed"},
            {
                "status": "error",
                "identity": None,
                "memory": None,
                "error": "probe failed",
            },
        ),
    ],
)
def test_metadata_accepts_exact_status_qualified_provenance(
    warp_version, device
):
    """Accept every documented status-qualified provenance schema."""
    metadata = build_resident_benchmark_metadata(
        timestamp_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        command="pytest resident",
        synchronization_method="explicit",
        warmup=2,
        timestep_count=5,
        seed=7,
        prepared_signature_digest="abc",
        warp_version=warp_version,
        device=device,
    )
    assert metadata["warp_version"] == warp_version
    assert metadata["device"] == device


@pytest.mark.parametrize(
    "warp_version, device, error",
    [
        (
            {"status": "unknown", "value": None},
            {"status": "unavailable", "identity": None, "memory": None},
            ValueError,
        ),
        (
            {"status": "available", "value": None},
            {"status": "unavailable", "identity": None, "memory": None},
            TypeError,
        ),
        (
            {"status": "unavailable", "value": None, "extra": None},
            {"status": "unavailable", "identity": None, "memory": None},
            ValueError,
        ),
        (
            {"status": "unavailable", "value": None},
            {"status": "available", "identity": "cuda:0", "memory": True},
            TypeError,
        ),
        (
            {"status": "unavailable", "value": None},
            {"status": "error", "identity": None, "memory": None},
            ValueError,
        ),
    ],
)
def test_metadata_rejects_malformed_status_qualified_provenance(
    warp_version, device, error
):
    """Reject extra, missing, and mistyped provenance fields deterministically."""
    with pytest.raises(error):
        build_resident_benchmark_metadata(
            timestamp_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
            command="pytest resident",
            synchronization_method="explicit",
            warmup=2,
            timestep_count=5,
            seed=7,
            prepared_signature_digest="abc",
            warp_version=warp_version,
            device=device,
        )


def test_result_statuses_require_their_respective_evidence():
    """Validate executed and explicit non-executed result contracts."""
    case = _case()
    samples = (0.0, 2.0)
    summary = summarize_timing_samples(samples)

    for status in (
        ResidentBenchmarkStatus.UNAVAILABLE,
        ResidentBenchmarkStatus.SKIPPED_BUDGET,
    ):
        result = ResidentBenchmarkResult(
            case_id=case.case_id,
            timing_mode=None,
            requested_shape=case.requested_shape,
            status=status,
            reason="explicit host-only outcome",
            samples=(),
            summary=None,
            provenance={"status": status.value},
        )
        assert result.status is status

    with pytest.raises(ValueError, match="exactly match"):
        ResidentBenchmarkResult(
            case_id=case.case_id,
            timing_mode="wall_clock",
            requested_shape=case.requested_shape,
            status=ResidentBenchmarkStatus.EXECUTED,
            reason=None,
            samples=samples,
            summary=summarize_timing_samples((1.0, 2.0)),
            provenance={"run": "bad-summary"},
        )
    with pytest.raises(ValueError, match="nonempty reason"):
        ResidentBenchmarkResult(
            case_id=case.case_id,
            timing_mode=None,
            requested_shape=case.requested_shape,
            status=ResidentBenchmarkStatus.UNAVAILABLE,
            reason="",
            samples=(),
            summary=None,
            provenance={"run": "missing-reason"},
        )
    with pytest.raises(ValueError, match="cannot contain timing data"):
        ResidentBenchmarkResult(
            case_id=case.case_id,
            timing_mode=None,
            requested_shape=case.requested_shape,
            status=ResidentBenchmarkStatus.SKIPPED_BUDGET,
            reason="budget",
            samples=(),
            summary=summary,
            provenance={"run": "timing-data"},
        )


@pytest.mark.parametrize(
    "samples, error",
    [
        ((), ValueError),
        ((-1.0,), ValueError),
        ((float("nan"),), ValueError),
        ((1.0,) * (MAX_TIMING_SAMPLES + 1), ValueError),
        ([1.0], TypeError),
    ],
)
def test_timing_samples_reject_invalid_values(samples, error):
    """Reject invalid sample containers and values before work."""
    with pytest.raises(error):
        summarize_timing_samples(samples)


def test_timing_summary_uses_nearest_rank_at_boundary_counts():
    """Use the documented nearest-rank p95 for one and twenty samples."""
    assert summarize_timing_samples((4.0,)).p95 == 4.0
    summary = summarize_timing_samples(
        tuple(float(value) for value in range(20))
    )
    assert summary.p95 == 18.0


def test_schema_rejects_noncanonical_and_invalid_result_references():
    """Reject invalid dimensions, case IDs, and artifact result references."""
    with pytest.raises(TypeError, match="requested_shape"):
        ResidentBenchmarkCase(
            "bad",
            (True, 2, 3),
            (1, 1, 1),
            0.0,
            ("condensation",),
            "none",
            (),
            0,
            1,
            0,
        )


def test_canonical_case_id_uses_injective_diagnostic_encoding():
    """Keep distinct hyphen-containing diagnostic selections distinct."""
    common = {
        "requested_shape": (1, 1, 1),
        "actual_shape": (1, 1, 1),
        "active_fraction": 1.0,
        "processes": ("condensation",),
        "communication": "none",
        "warmup": 0,
        "timestep_count": 1,
        "seed": 0,
    }
    first = build_resident_benchmark_case_id(diagnostics=("a-b", "c"), **common)
    second = build_resident_benchmark_case_id(
        diagnostics=("a", "b-c"), **common
    )
    assert first != second
    with pytest.raises(ValueError, match="canonical"):
        ResidentBenchmarkCase(
            "bad",
            (2, 2, 2),
            (1, 1, 1),
            0.0,
            ("dilution", "condensation"),
            "none",
            (),
            0,
            1,
            0,
        )
    case = _case()
    result = ResidentBenchmarkResult(
        case_id="unknown",
        timing_mode=None,
        requested_shape=case.requested_shape,
        status=ResidentBenchmarkStatus.SKIPPED_BUDGET,
        reason="limit",
        samples=(),
        summary=None,
        provenance={"source": "test"},
    )
    with pytest.raises(ValueError, match="unknown"):
        ResidentBenchmarkArtifact(_metadata(), (case,), (result,))

    with pytest.raises(ValueError, match="metadata has invalid fields"):
        ResidentBenchmarkArtifact({"command": "missing fields"}, (case,), ())
    with pytest.raises(ValueError, match="must not exceed"):
        ResidentBenchmarkCase(
            "bad",
            (2, 2, 2),
            (3, 1, 1),
            0.0,
            ("condensation",),
            "none",
            (),
            0,
            1,
            0,
        )


def test_artifact_rejects_duplicate_rows_and_malformed_deserialization():
    """Route duplicate and malformed JSON rows through normal validation."""
    artifact = _artifact()
    executed = artifact.results[0]
    with pytest.raises(ValueError, match="unique case_id/timing_mode"):
        ResidentBenchmarkArtifact(
            artifact.metadata,
            artifact.cases,
            (executed, executed),
        )

    payload = json.loads(serialize_resident_benchmark_artifact(artifact))
    payload["artifact"]["cases"][0]["actual_shape"] = [3, 4, 2]
    with pytest.raises(ValueError, match="must not exceed"):
        deserialize_resident_benchmark_artifact(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid fields"):
        deserialize_resident_benchmark_artifact('{"schema_version": 1}')


def test_artifact_deserialization_enforces_bounded_untrusted_input():
    """Reject excessive bytes, depth, rows, and nested metadata containers."""
    artifact = json.loads(serialize_resident_benchmark_artifact(_artifact()))
    boundary = json.dumps(artifact)
    assert deserialize_resident_benchmark_artifact(boundary)
    with pytest.raises(ValueError, match="byte size"):
        deserialize_resident_benchmark_artifact(
            " " * (MAX_ARTIFACT_PAYLOAD_BYTES + 1)
        )
    with pytest.raises(ValueError, match="nesting depth"):
        deserialize_resident_benchmark_artifact(
            "[" * (MAX_ARTIFACT_NESTING_DEPTH + 1)
        )
    artifact["artifact"]["cases"] = [{}] * (MAX_ARTIFACT_ROWS + 1)
    with pytest.raises(ValueError, match="row count"):
        deserialize_resident_benchmark_artifact(json.dumps(artifact))
    artifact = json.loads(serialize_resident_benchmark_artifact(_artifact()))
    artifact["artifact"]["metadata"]["nested"] = list(
        range(MAX_ARTIFACT_CONTAINER_ITEMS + 1)
    )
    with pytest.raises(ValueError, match="item count"):
        deserialize_resident_benchmark_artifact(json.dumps(artifact))


def test_write_json_artifact_is_contained_deterministic_and_atomic(
    tmp_path: Path,
):
    """Write generic normalized JSON only under a trusted artifact root."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    destination = write_json_artifact(
        root, "nested/result.json", {"b": 2, "a": 1}
    )
    assert destination == root / "nested/result.json"
    assert destination.read_text() == '{\n  "a": 1,\n  "b": 2\n}\n'
    write_json_artifact(root, "nested/result.json", {"a": 3})
    assert destination.read_text() == '{\n  "a": 3\n}\n'
    with pytest.raises(OSError, match="validation"):
        write_json_artifact(root, "../outside.json", {"a": 1})
    with pytest.raises(OSError, match="serialization"):
        write_json_artifact(root, "new/value.json", {"bad": object()})
    assert not (root / "new").exists()


def test_write_json_artifact_preserves_completed_file_on_write_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Retain completed bytes and clean temporary files after write failures."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    destination = write_json_artifact(root, "result.json", {"version": 1})
    original = destination.read_bytes()
    native_fsync = os.fsync

    def raise_fsync(_: int) -> None:
        """Simulate a failed durable temporary-file write."""
        raise OSError("fsync failed")

    monkeypatch.setattr(resident_benchmark_support.os, "fsync", raise_fsync)
    with pytest.raises(OSError, match="(validation|write)"):
        write_json_artifact(root, "result.json", {"version": 2})
    assert destination.read_bytes() == original
    assert not list(root.glob(".resident-benchmark-*"))

    monkeypatch.setattr(resident_benchmark_support.os, "fsync", native_fsync)

    def raise_replace(*_: object, **__: object) -> None:
        """Simulate replacement failure after a complete temporary write."""
        raise OSError("replace failed")

    monkeypatch.setattr(resident_benchmark_support.os, "replace", raise_replace)
    with pytest.raises(OSError, match="artifact write"):
        write_json_artifact(root, "result.json", {"version": 3})
    assert destination.read_bytes() == original
    assert not list(root.glob(".resident-benchmark-*"))


def test_write_json_artifact_rejects_symlink_escapes(tmp_path: Path):
    """Reject untrusted root, parent, and leaf symlinks before writing."""
    root = tmp_path / ".artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError, match="escapes"):
        write_json_artifact(root, "linked/value.json", {"safe": True})
    assert not (outside / "value.json").exists()
    root_link = tmp_path / "root-link" / ".artifacts"
    root_link.parent.mkdir()
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError, match="validation"):
        write_json_artifact(root_link, "value.json", {"safe": True})

    external_file = outside / "existing.json"
    external_file.write_text("outside\n")
    (root / "leaf.json").symlink_to(external_file)
    with pytest.raises(OSError, match="escapes"):
        write_json_artifact(root, "leaf.json", {"safe": True})
    assert external_file.read_text() == "outside\n"


def test_write_json_artifact_rejects_invalid_roots_and_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Reject invalid roots and retain failure context when cleanup also fails."""
    with pytest.raises(OSError, match="validation"):
        write_json_artifact(
            tmp_path / ".artifacts", "result.json", {"safe": True}
        )

    wrong_root = tmp_path / "artifacts"
    wrong_root.mkdir()
    with pytest.raises(OSError, match="validation"):
        write_json_artifact(wrong_root, "result.json", {"safe": True})

    root = tmp_path / ".artifacts"
    root.mkdir()
    destination = write_json_artifact(root, "result.json", {"version": 1})
    original = destination.read_bytes()

    def raise_replace(*_: object, **__: object) -> None:
        """Simulate replacement failure after writing a temporary file."""
        raise OSError("replace failed")

    def raise_unlink(*_: object, **__: object) -> None:
        """Simulate temporary cleanup failure."""
        raise OSError("unlink failed")

    monkeypatch.setattr(resident_benchmark_support.os, "replace", raise_replace)
    monkeypatch.setattr(resident_benchmark_support.os, "unlink", raise_unlink)
    with pytest.raises(OSError, match="cleanup also failed"):
        write_json_artifact(root, "result.json", {"version": 2})
    assert destination.read_bytes() == original

    monkeypatch.undo()
    for temporary in root.glob(".resident-benchmark-*"):
        temporary.unlink()


def test_write_json_artifact_rejects_directory_swap_to_external_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Keep descriptor-relative writes inside root during a directory swap."""
    root = tmp_path / ".artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    native_mkdir = os.mkdir

    def swap_after_mkdir(
        path: str, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        """Replace the newly made destination directory before it is reopened."""
        native_mkdir(path, mode, dir_fd=dir_fd)
        if path == "nested":
            nested = root / path
            nested.rename(root / "nested-original")
            nested.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        resident_benchmark_support.os, "mkdir", swap_after_mkdir
    )
    with pytest.raises(OSError, match="escapes"):
        write_json_artifact(root, "nested/result.json", {"safe": True})
    assert not (outside / "result.json").exists()


def test_paired_collector_alternates_without_warmup_synchronization() -> None:
    """Measure paired operations with exactly one sync per measured path."""
    calls: list[str] = []
    ticks = iter((0.0, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 14.0))
    uncaptured, replay = collect_paired_device_timings(
        uncaptured_operation=lambda: calls.append("uncaptured"),
        replay_operation=lambda: calls.append("replay"),
        synchronize=lambda: calls.append("sync"),
        clock=lambda: next(ticks),
        warmup_count=1,
        sample_count=2,
    )
    assert uncaptured == (1.0, 3.0)
    assert replay == (2.0, 4.0)
    assert calls == [
        "uncaptured",
        "replay",
        "uncaptured",
        "sync",
        "replay",
        "sync",
        "uncaptured",
        "sync",
        "replay",
        "sync",
    ]


@pytest.mark.parametrize(
    "warmup_count,sample_count",
    [
        (MAX_WARMUP_SAMPLES + 1, 1),
        (0, MAX_TIMING_SAMPLES + 1),
        (0, 0),
    ],
)
def test_paired_collector_rejects_counts_before_callbacks(
    warmup_count: int, sample_count: int
) -> None:
    """Reject bounded count failures before an operation, clock, or sync call."""
    calls: list[str] = []
    with pytest.raises(ValueError):
        collect_paired_device_timings(
            uncaptured_operation=lambda: calls.append("uncaptured"),
            replay_operation=lambda: calls.append("replay"),
            synchronize=lambda: calls.append("sync"),
            clock=lambda: calls.append("clock"),
            warmup_count=warmup_count,
            sample_count=sample_count,
        )
    assert calls == []


def test_comparison_writer_emits_one_fixed_schema_envelope(
    tmp_path: Path,
) -> None:
    """Persist only the fixed resident artifact without generic output state."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    artifact = _artifact()
    destination = write_resident_capture_comparison_artifact(root, artifact)
    assert destination == root / RESIDENT_CAPTURE_COMPARISON_DESTINATION
    assert (
        deserialize_resident_benchmark_artifact(destination.read_text())
        == artifact
    )
    assert list(root.rglob("*.json")) == [destination]


def test_schema_v1_decode_populates_absent_timing_provenance_with_none() -> (
    None
):
    """Retain backwards decoding while v2 remains the only emitted schema."""
    payload = json.loads(serialize_resident_benchmark_artifact(_artifact()))
    payload["schema_version"] = 1
    for result in payload["artifact"]["results"]:
        result.pop("setup_elapsed_seconds")
        result.pop("capture_elapsed_seconds")
    decoded = deserialize_resident_benchmark_artifact(json.dumps(payload))
    assert all(
        result.setup_elapsed_seconds is None for result in decoded.results
    )
    assert all(
        result.capture_elapsed_seconds is None for result in decoded.results
    )
